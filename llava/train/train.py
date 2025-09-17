# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image


local_rank = None


def rank0_print(*args):

    print("current file path", "llava/train/train.py")
    print("def rank0_print(*args)")
    print("args\n", args) # ('Formatting inputs...Skip in lazy mode',)
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):

    print("current file path", "llava/train/train.py")
    print("def maybe_zero_3(param, ignore_status=False, name=None)")
    print("param\n", param)
    if hasattr(param, 'shape'):
        print("param.shape\n", param.shape)
    print("ignore_status\n", ignore_status)
    print("name\n", name)
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    print("param (return)\n", param)
    if hasattr(param, 'shape'):
        print("param (return).shape\n", param.shape)
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):

    print("current file path", "llava/train/train.py")
    print("def get_peft_state_maybe_zero_3(named_params, bias)")
    print("named_params\n", named_params)
    print("bias\n", bias)
    print(f"[COND] bias={bias}")
    if bias == "none":
        print("【ENTER】if bias == 'none':")
        to_return = {k: t for k, t in named_params if "lora_" in k}
        print("【EXIT】if bias == 'none':")
    elif bias == "all":
        print("【ENTER】elif bias == 'all':")
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
        print("【EXIT】elif bias == 'all':")
    elif bias == "lora_only":
        print("【ENTER】elif bias == 'lora_only':")
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
        print("【EXIT】elif bias == 'lora_only':")
    else:
        print("【ENTER】else of if bias == ...:")
        raise NotImplementedError
        print("【EXIT】else of if bias == ...:")
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    print("to_return (return)\n", to_return)
    for k, v in to_return.items():
        if hasattr(v, 'shape'):
            print(f"to_return['{k}'].shape\n", v.shape)
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):

    print("current file path", "llava/train/train.py")
    print("def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True)")
    print("named_params\n", named_params)
    print("require_grad_only\n", require_grad_only)
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    print("to_return (return)\n", to_return)
    for k, v in to_return.items():
        if hasattr(v, 'shape'):
            print(f"to_return['{k}'].shape\n", v.shape)
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):

    print("current file path", "llava/train/train.py")
    print("def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match)")
    print("named_params\n", named_params) # <generator object Module.named_parameters at 0x7ed637a75000>
    print("keys_to_match\n", keys_to_match) # ['mm_projector', 'vision_resampler']
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}

    print("to_return (return)\n", to_return)
    """
    to_return
    {'model.mm_projector.0.weight': tensor([[-0.0299, -0.0014,  0.0156,  ...,  0.0070,  0.0259, -0.0071],
            [-0.0011, -0.0205,  0.0048,  ..., -0.0051,  0.0042,  0.0079],
            [ 0.0115, -0.0201,  0.0222,  ..., -0.0037, -0.0095, -0.0099],
            ...,
            [ 0.0006, -0.0216,  0.0028,  ..., -0.0090,  0.0134,  0.0194],
            [-0.0238,  0.0164, -0.0289,  ...,  0.0141,  0.0113,  0.0054],
            [ 0.0259,  0.0069, -0.0188,  ...,  0.0201, -0.0157,  0.0148]],
        dtype=torch.bfloat16), 'model.mm_projector.0.bias': tensor([-0.0119,  0.0082,  0.0117,  ...,  0.0188, -0.0243,  0.0286],
        dtype=torch.bfloat16), 'model.mm_projector.2.weight': tensor([[-1.3123e-02, -1.5442e-02, -1.3733e-02,  ..., -9.8419e-04,
            1.7700e-03, -1.4709e-02],
            [-2.3346e-03, -1.3855e-02,  9.5215e-03,  ...,  7.8125e-03,
            4.2419e-03, -8.4229e-03],
            [-1.1841e-02,  1.4709e-02,  1.3794e-02,  ...,  8.4839e-03,
            -1.1597e-02, -6.6833e-03],
            ...,
            [-1.4282e-02,  1.9073e-03,  5.0049e-03,  ...,  1.3672e-02,
            -4.3945e-03,  1.3000e-02],
            [-8.7280e-03,  1.4893e-02, -7.4768e-03,  ...,  5.2185e-03,
            -5.2452e-05,  1.0910e-03],
            [-8.9722e-03,  3.6011e-03, -1.4587e-02,  ..., -1.0803e-02,
            -5.3711e-03,  9.6436e-03]], dtype=torch.bfloat16), 'model.mm_projector.2.bias': tensor([ 0.0039, -0.0108, -0.0074,  ...,  0.0064, -0.0076, -0.0034],
        dtype=torch.bfloat16)}
    """
    for k, v in to_return.items():
        if hasattr(v, 'shape'):
            print(f"to_return['{k}'].shape\n", v.shape)
    return to_return


def find_all_linear_names(model):

    print("current file path", "llava/train/train.py")
    print("def find_all_linear_names(model)")
    print("model\n", model)
    if hasattr(model, 'shape'):
        print("model.shape\n", model.shape)
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    result = list(lora_module_names)
    print("result (return)\n", result)
    return result


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):

    print("current file path", "llava/train/train.py")
    print("def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str)")
    print("trainer\n", type(trainer)) # <class 'llava.train.llava_trainer.LLaVATrainer'>
    print("output_dir\n", output_dir) # ./checkpoints/llava-v1.5-7b-pretrain
    """Collects the state dict and dump to disk."""

    print("trainer.args\n", trainer.args) # TrainingArguments(...
    print("【COND】tune_mm_mlp_adapter=", getattr(trainer.args, "tune_mm_mlp_adapter", False)) # False
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        print("【ENTER】if getattr(trainer.args, 'tune_mm_mlp_adapter', False):")
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            print("【ENTER】if getattr(trainer.args, 'use_im_start_end', False):")
            keys_to_match.extend(['embed_tokens', 'embed_in'])
            print("【EXIT】if getattr(trainer.args, 'use_im_start_end', False):")

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        if hasattr(weight_to_save, 'shape'):
            print("【ENTER】if hasattr(weight_to_save, 'shape'):")
            print("weight_to_save.shape\n", weight_to_save.shape)
            print("【EXIT】if hasattr(weight_to_save, 'shape'):")
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        print("current_folder\n", current_folder) # checkpoint-xxx or llava-v1.5-7b-pretrain
        print("parent_folder\n", parent_folder) # ./checkpoints
        print("【COND】trainer.args.local_rank=", trainer.args.local_rank) # 0 or -1
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            print("【ENTER】if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:")
            if current_folder.startswith('checkpoint-'):
                print("【ENTER】if current_folder.startswith('checkpoint-'):")
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
                print(f"Adapter weights saved to {os.path.join(mm_projector_folder, f'{current_folder}.bin')}")
                print("【EXIT】if current_folder.startswith('checkpoint-'):")
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
                print(f"Adapter weights saved to {os.path.join(output_dir, f'mm_projector.bin')}")
            print("【EXIT】if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:")
        print("【EXIT】if getattr(trainer.args, 'tune_mm_mlp_adapter', False):")
        return

    if trainer.deepspeed:
        print("【ENTER】if trainer.deepspeed:")
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        print("【EXIT】if trainer.deepspeed:")
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        print("【ENTER】if trainer.args.should_save:")
        cpu_state_dict = {
            key: value.cpu() for key, value in state_dict.items()
        }
        for key, value in cpu_state_dict.items():
            if hasattr(value, 'shape'):
                print(f"cpu_state_dict['{key}'].shape\n", value.shape)
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        print("【EXIT】if trainer.args.should_save:")


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):

    print("current file path", "llava/train/train.py")
    print("def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)")
    print("special_tokens_dict\n", special_tokens_dict)
    print("tokenizer\n", type(tokenizer))
    print("model\n", type(model))
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        print("input_embeddings.shape\n", input_embeddings.shape)
        print("output_embeddings.shape\n", output_embeddings.shape)
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:

    print("current file path", "llava/train/train.py")
    print("def _tokenize_fn(strings, tokenizer)")
    print("strings\n", strings)
    print("tokenizer\n", type(tokenizer))
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    for idx, tensor in enumerate(input_ids):
        if hasattr(tensor, 'shape'):
            print(f"input_ids[{idx}].shape\n", tensor.shape)
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):

    print("current file path", "llava/train/train.py")
    print("def _mask_targets(target, tokenized_lens, speakers)")
    print("target\n", target)
    print("tokenized_lens\n", tokenized_lens)
    print("speakers\n", speakers)
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):

    print("current file path", "llava/train/train.py")
    print("def _add_speaker_and_signal(header, source, get_conversation=True)")
    print("header\n", header)
    print("source\n", source)
    print("get_conversation\n", get_conversation)
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:

    print("current file path", "llava/train/train.py")
    print("def preprocess_multimodal(sources, data_args)")
    print("sources\n", sources) # [[{'from': 'human', 'value': 'Give a brief description of the image.\n<image>'}, {'from': 'gpt', 'value': 'the divine queen in her elaborate masks canvas print featuring the face and hands of a woman with red hair'}]]
    print("data_args\n", data_args) # DataArguments(data_path='/content/LLaVA/blip_laion_cc_sbu_1.json', lazy_preprocess=True, is_multimodal=True, image_folder='/content/LLaVA/images', image_aspect_ratio='square')
    is_multimodal = data_args.is_multimodal 
    print("is_multimodal\n", is_multimodal) # True
    if not is_multimodal:
        # 【SKIP】
        return sources

    for source in sources:
        print("source current loop\n", source)
        for sentence in source:
            print("sentence current loop\n", sentence)
            print("【COND】 if DEFAULT_IMAGE_TOKEN in sentence['value']:", DEFAULT_IMAGE_TOKEN in sentence['value'])
            print("sentence['value']\n", sentence['value'])
            print("DEFAULT_IMAGE_TOKEN\n", DEFAULT_IMAGE_TOKEN)
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                print("【ENTER】if DEFAULT_IMAGE_TOKEN in sentence['value']:")
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    print("sources (final return)\n", sources)
    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:

    print("current file path", "llava/train/train.py")
    print("def preprocess_llama_2(sources, tokenizer, has_image=False)")
    print("sources\n", sources)
    print("tokenizer\n", type(tokenizer))
    print("has_image\n", has_image)
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    print("input_ids.shape\n", input_ids.shape)

    targets = input_ids.clone()
    print("targets.shape\n", targets.shape)

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:

    print("current file path", "llava/train/train.py")
    print("def preprocess_v1(sources, tokenizer, has_image=False)")
    print("sources\n", sources)
    print("tokenizer\n", type(tokenizer))
    print("has_image\n", has_image)
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    print("input_ids.shape\n", input_ids.shape)

    targets = input_ids.clone()
    print("targets.shape\n", targets.shape)

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:

    print("current file path", "llava/train/train.py")
    print("def preprocess_mpt(sources, tokenizer)")
    print("sources\n", sources)
    print("tokenizer\n", type(tokenizer))
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    print("input_ids.shape\n", input_ids.shape)
    targets = input_ids.clone()
    print("targets.shape\n", targets.shape)
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:

    print("current file path", "llava/train/train.py")
    print("def preprocess_plain(sources, tokenizer)")
    print("sources\n", sources) # [[{'from': 'human', 'value': '<image>\nGive a brief description of the image.'}, {'from': 'gpt', 'value': 'the divine queen in her elaborate masks canvas print featuring the face and hands of a woman with red hair'}]]
    print("tokenizer\n", type(tokenizer)) # <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>
    # add end signal and concatenate together
    conversations = []
    print("conversations initial\n", conversations) # []
    for source in sources:
        print("source current loop\n", source) 
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        print("conversation current loop\n", conversation)
        conversations.append(conversation)
    print("conversations (final)\n", conversations) 
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    print("input_ids\n", input_ids)
    for idx, tensor in enumerate(input_ids):
        if hasattr(tensor, 'shape'):
            print(f"input_ids[{idx}].shape\n", tensor.shape) # torch.Size([24])
    targets = copy.deepcopy(input_ids)
    print("targets\n", targets)
    for idx, tensor in enumerate(targets):
        if hasattr(tensor, 'shape'):
            print(f"targets[{idx}].shape\n", tensor.shape) # torch.Size([24])
    print("sources\n", sources)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer)) # prompt <image>
        target[:tokenized_len] = IGNORE_INDEX

    print("input_ids (return)\n", input_ids) # [1, -200]
    print("targets (return)\n", targets)
    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:

    print("current file path", "llava/train/train.py")
    print("def preprocess(sources, tokenizer, has_image=False)")
    print("sources\n", sources)
    print("tokenizer\n", type(tokenizer))
    print("has_image\n", has_image)
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
        for idx, tensor in enumerate(input_ids):
            if hasattr(tensor, 'shape'):
                print(f"input_ids[{idx}].shape\n", tensor.shape)
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    if isinstance(targets, list):
        for idx, tensor in enumerate(targets):
            if hasattr(tensor, 'shape'):
                print(f"targets[{idx}].shape\n", tensor.shape)
    elif hasattr(targets, 'shape'):
        print("targets.shape\n", targets.shape)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):

        print("current file path", "llava/train/train.py")
        print("def LazySupervisedDataset.__init__(self, data_path, tokenizer, data_args)")
        print("data_path\n", data_path) # /content/LLaVA/blip_laion_cc_sbu_1.json
        print("tokenizer\n", type(tokenizer)) # <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>
        print("data_args\n", data_args) # DataArguments(data_path='/content/LLaVA/blip_laion_cc_sbu_1.json', lazy_preprocess=True, is_multimodal=True, image_folder='/content/LLaVA/images', image_aspect_ratio='square')
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        # 今回は1サンプルだけなのでprintしても危険ではない
        print("list_data_dict", list_data_dict)

        rank0_print("Formatting inputs...Skip in lazy mode") # Formatting inputs...Skip in lazy mode
        self.tokenizer = tokenizer
        print("self.tokenizer\n", self.tokenizer)
        self.list_data_dict = list_data_dict
        print("self.list_data_dict\n", self.list_data_dict)
        self.data_args = data_args
        print("self.data_args\n", self.data_args)

    def __len__(self):

        print("current file path", "llava/train/train.py")
        print("def LazySupervisedDataset.__len__(self)")
        return len(self.list_data_dict)

    @property
    def lengths(self):

        print("current file path", "llava/train/train.py")
        print("def LazySupervisedDataset.lengths(self)")
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):

        print("current file path", "llava/train/train.py")
        print("def LazySupervisedDataset.modality_lengths(self)")
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        print("current file path", "llava/train/train.py")
        print("def LazySupervisedDataset.__getitem__(self, i)")
        print("i\n", i) # 0
        sources = self.list_data_dict[i]
        print("sources\n", sources)
        print("【COND】 isinstance(i, int):", isinstance(i, int))
        if isinstance(i, int):
            print("【ENTER】if isinstance(i, int):")
            sources = [sources]
            print("sources (after)\n", sources)
            print("【EXIT】if isinstance(i, int):")
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        print("【COND】 'image' in sources[0]:", 'image' in sources[0])
        if 'image' in sources[0]:
            print("【ENTER】if 'image' in sources[0]:")
            image_file = self.list_data_dict[i]['image']
            print("image_file\n", image_file)
            image_folder = self.data_args.image_folder
            print("image_folder\n", image_folder)
            processor = self.data_args.image_processor
            print("processor\n", processor)
            image_path = os.path.join(image_folder, image_file)
            print("image_path\n", image_path)
            try:
                print("Trying to open image...")
                image = Image.open(image_path).convert('RGB')
                print("Image opened successfully.")
            except Exception as e:
                print(f"Error opening image: {e}")
                # 画像がなければこのサンプルはスキップ
                print("Skipping this sample due to image loading error.")
                return None 
            print("【COND】 self.data_args.image_aspect_ratio", self.data_args.image_aspect_ratio) # square
            if self.data_args.image_aspect_ratio == 'pad':
                # 【SKIP】
                print("【ENTER】if self.data_args.image_aspect_ratio == 'pad':")
                def expand2square(pil_img, background_color):
                    print("def expand2square(pil_img, background_color)")
                    width, height = pil_img.size
                    print("width\n", width)
                    print("height\n", height)
                    if width == height:
                        print("【ENTER】if width == height:")
                        print("pil_img (return)\n", pil_img)
                        print("【EXIT】if width == height:")
                        return pil_img
                    elif width > height:
                        print("【ENTER】elif width > height:")
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        print("result (after)\n", result)
                        print("【EXIT】elif width > height:")
                        return result
                    else:
                        print("【ENTER】width < height:")
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        print("result (after)\n", result)
                        print("【EXIT】width < height:")
                        return result
                print("image (before)\n", image)
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                print("image (after expand2square)\n", image)
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                print("image (after processor.preprocess)\n", image)
                print("【EXIT】if self.data_args.image_aspect_ratio == 'pad':")
            else:
                print("【ENTER】else (self.data_args.image_aspect_ratio != 'pad')")
                print("image (before)\n", image)
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                print("image (after processor.preprocess)\n", image)
            print("sources (before preprocess_multimodal)\n", sources)
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
            print("sources (after preprocess_multimodal)\n", sources)
        else:
            # 【ENTER】
            print("【ENTER】else ('image' not in sources[0])")
            sources = copy.deepcopy([e["conversations"] for e in sources])
            print("sources (after deepcopy)\n", sources)
            print("【EXIT】else ('image' not in sources[0])")
        print("Calling preprocess...")
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        print("data_dict (after preprocess)\n", data_dict)
        print("【COND】 isinstance(i, int):", isinstance(i, int))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        print("current file path", "llava/train/train.py")
        print("def DataCollatorForSupervisedDataset.__call__(self, instances)")
        print("instances\n", instances)
        """
        [{'input_ids': tensor([    1,  -200,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
        10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
          411,  2654, 11315,    13]), 'labels': tensor([ -100,  -100,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
        10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
          411,  2654, 11315,    13]), 'image': tensor([[[ 0.0325,  0.0325,  0.0325,  ..., -0.7120, -0.3616, -0.1280],
         [ 0.0325,  0.0325,  0.0325,  ..., -0.3908, -0.1718, -0.0259],
         [ 0.0325,  0.0325,  0.0325,  ..., -0.0113,  0.0471,  0.0909],
         ...,
         [-1.0331, -1.0331, -1.0331,  ..., -1.0623, -1.0623, -1.0623],
         [-1.0477, -1.0331, -1.0331,  ..., -1.0623, -1.0623, -1.0623],
         [-1.0477, -1.0331, -1.0331,  ..., -1.0623, -1.0623, -1.0623]],

        [[ 0.3190,  0.3190,  0.3190,  ..., -0.3864, -0.0112,  0.2139],
         [ 0.3190,  0.3190,  0.3190,  ..., -0.0712,  0.1539,  0.3190],
         [ 0.3190,  0.3190,  0.3190,  ...,  0.2890,  0.3640,  0.4390],
         ...,
         [-1.0167, -1.0167, -1.0167,  ..., -1.0017, -1.0017, -1.0017],
         [-1.0317, -1.0167, -1.0167,  ..., -1.0017, -1.0017, -1.0017],
         [-1.0317, -1.0167, -1.0167,  ..., -1.0017, -1.0017, -1.0017]],

        [[ 0.9656,  0.9656,  0.9656,  ...,  0.0982,  0.4537,  0.6670],
         [ 0.9656,  0.9656,  0.9656,  ...,  0.3968,  0.6101,  0.7523],
         [ 0.9656,  0.9656,  0.9656,  ...,  0.7523,  0.8092,  0.8377],
         ...,
         [-0.3711, -0.3853, -0.3995,  ..., -0.4279, -0.4279, -0.4279],
         [-0.3711, -0.3711, -0.3853,  ..., -0.4279, -0.4279, -0.4279],
         [-0.3853, -0.3711, -0.3711,  ..., -0.4279, -0.4279, -0.4279]]])}]
        """
        # Noneを除外
        instances = [x for x in instances if x is not None]
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:

    print("current file path", "llava/train/train.py")
    print("def make_supervised_data_module(tokenizer, data_args)")
    print("tokenizer\n", type(tokenizer))
    print("data_args\n", data_args) #  DataArguments(data_path='/content/LLaVA/blip_laion_cc_sbu_1.json', lazy_preprocess=True, is_multimodal=True, image_folder='/content/LLaVA/images', image_aspect_ratio='square')
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    print("train_dataset\n", train_dataset) # <llava.train.train.LazySupervisedDataset object at 0x7ed6341f4880>
    print("len(train_dataset)\n", len(train_dataset)) # 1
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    print("data_collator\n", data_collator) # DataCollatorForSupervisedDataset(tokenizer=LlamaTokenizer(name_or_path='lmsys/vicuna-7b-v1.5', vocab_size=32000, model_max_length=2048, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'bos_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'eos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'unk_token': AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'pad_token': '<unk>'}, clean_up_tokenization_spaces=False))
    result = dict(train_dataset=train_dataset,
                  eval_dataset=None,
                  data_collator=data_collator)
    print("def make_supervised_data_module: result (return)\n", result) # {'train_dataset': <llava.train.train.LazySupervisedDataset object at 0x7ed6341f4880>, 'eval_dataset': None, 'data_collator': DataCollatorForSupervisedDataset(tokenizer=LlamaTokenizer(name_or_path='lmsys/vicuna-7b-v1.5', vocab_size=32000, model_max_length=2048, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'bos_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'eos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'unk_token': AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'pad_token': '<unk>'}, clean_up_tokenization_spaces=False))}
    return result

def train():

    print("current file path", "llava/train/train.py")
    print("def train()")
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    print("original parser\n", parser)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("model_args\n", model_args)
    print("data_args\n", data_args)
    print("training_args\n", training_args)
    local_rank = training_args.local_rank
    print("local_rank\n", local_rank)
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    print("compute_dtype\n", compute_dtype)
    bnb_model_from_pretrained_args = {}
    print("bnb_model_from_pretrained_args\n", bnb_model_from_pretrained_args)
    # 【SKIP】bfloat16 なので 以下の if 文はスキップされる
    print(f"[COND] bits={training_args.bits}")
    if training_args.bits in [4, 8]:
        print("【ENTER】if training_args.bits in [4, 8]:")
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
        print("【EXIT】if training_args.bits in [4, 8]:")

    print(f"[COND] vision_tower={model_args.vision_tower}")
    # 【ENTER】 vision_tower=openai/clip-vit-large-patch14-336 なので、この分岐に入る
    if model_args.vision_tower is not None:
        print("【ENTER】if model_args.vision_tower is not None:")
        print(f"[COND] mpt_in_model_name_or_path={'mpt' in model_args.model_name_or_path}")
        #【SKIP】model_args.model_name_or_path に mptは含まれていないので、この分岐はskipされる
        if 'mpt' in model_args.model_name_or_path:
            print("【ENTER】if 'mpt' in model_args.model_name_or_path:")
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            print("config\n", config)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            print("config.attn_config['attn_impl']\n", config.attn_config['attn_impl'])
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
            print("model\n", model)
            print("【EXIT】if 'mpt' in model_args.model_name_or_path:")
        #【ENTER】 model_args.model_name_or_path に mptは含まれていないので、この分岐に入る
        else:
            print("[COND] not_mpt_in_model_name_or_path={'mpt' not in model_args.model_name_or_path}")
            print("【ENTER】else of if 'mpt' in model_args.model_name_or_path:")
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
            print("model defined as LlavaLlamaForCausalLM \n", model)
            print("【EXIT】else of if 'mpt' in model_args.model_name_or_path:")
        print("【EXIT】if model_args.vision_tower is not None:")
    # 【SKIP】 vision_tower=clip-vit-large-patch14-336 なので、この分岐には入らない
    else:
        print("[COND] vision_tower=None")
        print("【ENTER】else of if model_args.vision_tower is not None:")
        # modelのロード（ここがたいへん重要）
        # model_name_or_path は lmsys/vicuna-7b-v1.5; https://huggingface.co/lmsys/vicuna-7b-v1.5/blob/main/config.json
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
        print("model defined as LlamaForCausalLM \n", model)
        print("【EXIT】else of if model_args.vision_tower is not None:")
    model.config.use_cache = False
    print("model.config.use_cache\n", model.config.use_cache)

    print(f"[COND] freeze_backbone={model_args.freeze_backbone}")
    # 【SKIP】 freeze_backbone=False なので、この分岐はskipされる
    if model_args.freeze_backbone:
        print("【ENTER】if model_args.freeze_backbone:")
        model.model.requires_grad_(False)
        print("【EXIT】if model_args.freeze_backbone:")

    # 【SKIP】 bfloat16 なので 以下の if 文はスキップされる
    print(f"[COND] bits={training_args.bits}")
    if training_args.bits in [4, 8]:
        print("【ENTER】if training_args.bits in [4, 8]:")
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        print("【EXIT】if training_args.bits in [4, 8]:")

    print(f"[COND] gradient_checkpointing={training_args.gradient_checkpointing}")
    # 【ENTER】 gradient_checkpointing=True なので、この分岐に入る
    if training_args.gradient_checkpointing:
        print("【ENTER】if training_args.gradient_checkpointing:")
        print(f"[COND] has_enable_input_require_grads={hasattr(model, 'enable_input_require_grads')}")
        # 【ENTER】 model に enable_input_require_grads メソッドがあるので、この分岐に入る
        if hasattr(model, "enable_input_require_grads"):
            print("【ENTER】if hasattr(model, 'enable_input_require_grads'):")
            model.enable_input_require_grads()
            print("【EXIT】if hasattr(model, 'enable_input_require_grads'):")
        # 【SKIP】 model に enable_input_require_grads メソッドがあるので、この分岐はskipされる
        else:
            print("【ENTER】else of if hasattr(model, 'enable_input_require_grads'):")
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            print("【EXIT】else of if hasattr(model, 'enable_input_require_grads'):")
        print("【EXIT】if training_args.gradient_checkpointing:")

    print(f"[COND] lora_enable={training_args.lora_enable}")
    # 【SKIP】 lora_enable=False なので、この分岐はskipされる
    if training_args.lora_enable:
        print("【ENTER】if training_args.lora_enable:")
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        print("LoraConfig =\n", lora_config)
        print(f"[COND] bits={training_args.bits}")
        if training_args.bits == 16:
            print("【ENTER】if training_args.bits == 16:")
            print(f"[COND] bf16={training_args.bf16}")
            if training_args.bf16:
                model.to(torch.bfloat16)
                print("【EXIT】if training_args.bf16:")
            print(f"[COND] fp16={training_args.fp16}")
            if training_args.fp16:
                print("【ENTER】if training_args.fp16:")
                model.to(torch.float16)
                print("【EXIT】if training_args.fp16:")
            print("【EXIT】if training_args.bits == 16:")
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        print("【EXIT】if training_args.lora_enable:")

    print(f"[COND] mpt_in_model_name_or_path={'mpt' in model_args.model_name_or_path}")
    # 【SKIP】model_args.model_name_or_path に mptは含まれていないので、この分岐はskipされる
    if 'mpt' in model_args.model_name_or_path:
        print("【ENTER】if 'mpt' in model_args.model_name_or_path:")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
        print("【EXIT】if 'mpt' in model_args.model_name_or_path:")
    #【ENTER】 model_args.model_name_or_path に mptは含まれていないので、この分岐に入る
    else:
        print("[COND] not_mpt_in_model_name_or_path={'mpt' not in model_args.model_name_or_path}")
        print("【ENTER】else of if 'mpt' in model_args.model_name_or_path:")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        print("tokenizer defined by AutoTokenizer.from_pretrained \n", tokenizer)
        print("【EXIT】else of if 'mpt' in model_args.model_name_or_path:")

    print(f"[COND] version={model_args.version}")
    # 【SKIP】 version=plain なので、この分岐はskipされる
    if model_args.version == "v0":
        print("【ENTER】if model_args.version == 'v0':")
        print(f"[COND] pad_token_is_None={tokenizer.pad_token is None}")
        if tokenizer.pad_token is None:
            print("【ENTER】if tokenizer.pad_token is None:")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
            print("【EXIT】if tokenizer.pad_token is None:")
        print("【EXIT】if model_args.version == 'v0':")
    # 【SKIP】 version=plain なので、この分岐はskipされる
    elif model_args.version == "v0.5":
        print("【ENTER】elif model_args.version == 'v0.5':")
        tokenizer.pad_token = tokenizer.unk_token
        print("【EXIT】elif model_args.version == 'v0.5':")
    # 【ENTER】 version=plain なので、この分岐に入る
    else:
        print("【ENTER】else of if model_args.version == 'v0' and elif 'v0.5':")
        tokenizer.pad_token = tokenizer.unk_token
        print(f"[COND] version_in_conv_templates={model_args.version in conversation_lib.conv_templates}")
        # 【ENTER】 model_args.version=plain は conversation_lib.conv_templates に含まれている（"plain": conv_llava_plain）ので、この分岐に入る
        if model_args.version in conversation_lib.conv_templates:
            print("【ENTER】if model_args.version in conversation_lib.conv_templates:")
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
            print(f"conversation_lib.default_conversation set to {model_args.version}")
            print("【EXIT】if model_args.version in conversation_lib.conv_templates:")
        # 【SKIP】 model_args.version=plain は conversation_lib.conv_templates に含まれているので、この分岐はskipされる
        else:
            print("【ENTER】else of if model_args.version in conversation_lib.conv_templates:")
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
            print("【EXIT】else of if model_args.version in conversation_lib.conv_templates:")
        print("【EXIT】else of if model_args.version == 'v0' and elif 'v0.5':")

    print(f"[COND] vision_tower={model_args.vision_tower}")
    # 【ENTER】 vision_tower=openai/clip-vit-large-patch14-336 なので、この分岐に入る
    if model_args.vision_tower is not None:
        print("【ENTER】if model_args.vision_tower is not None:")
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        print(f"[COND] tune_mm_mlp_adapter={model_args.tune_mm_mlp_adapter}") # True
        if model_args.tune_mm_mlp_adapter:
            # 【ENTER】 tune_mm_mlp_adapter=True なので、この分岐に入る
            print("【ENTER】if model_args.tune_mm_mlp_adapter:")
            # モデル全体の全パラメータを「学習不可（requires_grad=False）」にする
            # これで通常の重みは全て凍結される
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                # mm_projector（画像特徴量→テキスト特徴量への変換層）の全パラメータだけを「学習可能（requires_grad=True）」に戻す
                # これで mm_projector のみ学習されることになる
                print("model.get_model().mm_projector.parameters()", model.get_model().mm_projector.parameters())
                p.requires_grad = True
            print("【EXIT】if model_args.tune_mm_mlp_adapter:")

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        print(f"[COND] freeze_mm_mlp_adapter={training_args.freeze_mm_mlp_adapter}") # False
        if training_args.freeze_mm_mlp_adapter:
            # 【SKIP】 freeze_mm_mlp_adapter=False なので、この分岐はskipされる
            print("【ENTER】if training_args.freeze_mm_mlp_adapter:")
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
            print("【EXIT】if training_args.freeze_mm_mlp_adapter:")

        print(f"[COND] bits={training_args.bits}") # 16
        if training_args.bits in [4, 8]:
            # 【SKIP】 bfloat16 なので 以下の if 文はスキップされる
            print("【ENTER】if training_args.bits in [4, 8]:")
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
            print("【EXIT】if training_args.bits in [4, 8]:")

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        print("model_args.mm_use_im_start_end", model_args.mm_use_im_start_end)
        model.config.mm_projector_lr = training_args.mm_projector_lr
        print("training_args.mm_projector_lr", training_args.mm_projector_lr)
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        print("training_args.use_im_start_end", training_args.use_im_start_end)
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        print("model_args.mm_use_im_patch_token", model_args.mm_use_im_patch_token)
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        print("【EXIT】if model_args.vision_tower is not None:")

    print(f"[COND] bits={training_args.bits}") # 16
    if training_args.bits in [4, 8]:
        # 【SKIP】 bfloat16 なので 以下の if 文はスキップされる
        print("【ENTER】if training_args.bits in [4, 8]:")
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            print(f"[COND] is_LoraLayer={isinstance(module, LoraLayer)} name={name}")
            if isinstance(module, LoraLayer):
                print(f"[COND] bf16={training_args.bf16}")
                if training_args.bf16:
                    print("【ENTER】if training_args.bf16:")
                    module = module.to(torch.bfloat16)
                    print("【EXIT】if training_args.bf16:")
            print(f"[COND] 'norm'_in_name={'norm' in name}")
            if 'norm' in name:
                print("【ENTER】if 'norm' in name:")
                module = module.to(torch.float32)
                print("【EXIT】if 'norm' in name:")
            print(f"[COND] 'lm_head'_or_'embed_tokens'_in_name={('lm_head' in name or 'embed_tokens' in name)}")
            if 'lm_head' in name or 'embed_tokens' in name:
                print("【ENTER】if 'lm_head' in name or 'embed_tokens' in name:")
                if hasattr(module, 'weight'):
                    print(f"[COND] bf16={training_args.bf16} weight_dtype_is_float32={module.weight.dtype == torch.float32}")
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        print("【ENTER】if training_args.bf16 and module.weight.dtype == torch.float32:")
                        module = module.to(torch.bfloat16)
                        print("【EXIT】if training_args.bf16 and module.weight.dtype == torch.float32:")
                print("【EXIT】if 'lm_head' in name or 'embed_tokens' in name:")
        print("【EXIT】if training_args.bits in [4, 8]:")


    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    print("data_module\n", data_module) # {'train_dataset': <llava.train.train.LazySupervisedDataset object at 0x7ed6341f4880>, 'eval_dataset': None, 'data_collator': DataCollatorForSupervisedDataset(tokenizer=LlamaTokenizer(name_or_path='lmsys/vicuna-7b-v1.5', vocab_size=32000, model_max_length=2048, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'bos_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'eos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'unk_token': AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'pad_token': '<unk>'}, clean_up_tokenization_spaces=False))}

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    print("trainer\n", trainer) # <llava.train.llava_trainer.LLaVATrainer object at 0x7ed6341f4490>

    print("【COND】list(pathlib.Path(training_args.output_dir).glob('checkpoint-*'))\n", list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))) # [PosixPath('checkpoints/llava-v1.5-7b-pretrain/checkpoint-250'), PosixPath('checkpoints/llava-v1.5-7b-pretrain/checkpoint-1')]
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        # 【ENTER】 
        print("【ENTER】if list(pathlib.Path(training_args.output_dir).glob(checkpoint-*)):")
        trainer.train(resume_from_checkpoint=False)
        print("【EXIT】if list(pathlib.Path(training_args.output_dir).glob(checkpoint-*)):")
    else:
        print("【ENTER】else of if list(pathlib.Path(training_args.output_dir).glob(checkpoint-*)):")
        trainer.train()
        print("【EXIT】else of if list(pathlib.Path(training_args.output_dir).glob(checkpoint-*)):")
    trainer.save_state()

    model.config.use_cache = True
    print("model.config.use_cache = True", model.config.use_cache) # True

    print(f"【COND】lora_enable={training_args.lora_enable}") # False
    if training_args.lora_enable:
        # 【SKIP】
        print("【ENTER】if training_args.lora_enable:")
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        print("state_dict for LoRA:\n", state_dict)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        print("non_lora_state_dict:\n", non_lora_state_dict)
        print("【EXIT】if training_args.lora_enable:")
        print(f"[COND] local_rank={training_args.local_rank} (0 or -1)")
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            print("【ENTER】if training_args.local_rank == 0 or training_args.local_rank == -1:")
            print("training_args.output_dir", training_args.output_dir)
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            print("【EXIT】if training_args.local_rank == 0 or training_args.local_rank == -1:")
    else:
        # 【ENTER】
        print("【ENTER】else of if training_args.lora_enable:")
        print("trainer", trainer) # <class 'llava.train.llava_trainer.LLaVATrainer'>
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
        print("【EXIT】else of if training_args.lora_enable:")


if __name__ == "__main__":
    train()
