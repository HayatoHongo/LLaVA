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

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image


local_rank = None

def rank0_print(*args):
    """
    rank0_print(*args)
    rank0プロセスのみprintするユーティリティ関数。
    
    入力例:
        args = ("学習開始",)
    出力:
        rank0プロセスならprint("学習開始")、それ以外は何もしない
    """
    print("current file path", "llava/train/train.py")
    print("def rank0_print(*args)")
    print("args", "print(args) disabled for safety")
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    def __post_init__(self):
        print("current file path", "llava/train/train.py")
        print("def ModelArguments(...) (dataclass init)")
    """
    ModelArguments
    モデル設定用の引数を管理するデータクラス。
    
    入力例:
        ModelArguments(model_name_or_path="facebook/opt-125m", version="v0", ...)
    出力:
        モデル設定値を属性として持つインスタンス
    """
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")   # モデル名またはパス
    version: Optional[str] = field(default="v0")   # 会話テンプレートや前処理のバージョン指定
    freeze_backbone: bool = field(default=False)   # モデルのバックボーンを凍結するか
    tune_mm_mlp_adapter: bool = field(default=False)   # マルチモーダルMLPアダプタのみ学習するか
    vision_tower: Optional[str] = field(default=None)   # 画像特徴抽出用のビジョンモデル名またはパス
    mm_vision_select_layer: Optional[int] = field(default=-1)   # ビジョンタワーのどの層の特徴を使うか（-1で最終層）
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)   # 事前学習済みマルチモーダルMLPアダプタのパス
    mm_projector_type: Optional[str] = field(default='linear')   # マルチモーダルプロジェクタの種類（linear等）
    mm_use_im_start_end: bool = field(default=False)   # 画像トークンの前後に特殊トークンを付与するか
    mm_use_im_patch_token: bool = field(default=True)   # 画像パッチトークンを使うか
    mm_patch_merge_type: Optional[str] = field(default='flat')   # パッチ統合方法（flat等）
    mm_vision_select_feature: Optional[str] = field(default="patch")   # 画像特徴の種類（patch等）


@dataclass
class DataArguments:
    def __post_init__(self):
        print("current file path", "llava/train/train.py")
        print("def DataArguments(...) (dataclass init)")
    """
    DataArguments
    データセット設定用の引数を管理するデータクラス。
    
    入力例:
        DataArguments(data_path="train.json", image_folder="images/", ...)
    出力:
        データセット設定値を属性として持つインスタンス
    """
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})   # 学習データのパス
    lazy_preprocess: bool = False   # データの遅延前処理を行うか
    is_multimodal: bool = False   # マルチモーダルデータかどうか
    image_folder: Optional[str] = field(default=None)   # 画像フォルダのパス
    image_aspect_ratio: str = 'square'   # 画像のアスペクト比（square/pad等）


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    def __post_init__(self):
        print("current file path", "llava/train/train.py")
        print("def TrainingArguments(...) (dataclass init)")
    """
    TrainingArguments
    学習設定用の引数を管理するデータクラス。
    transformers.TrainingArgumentsを継承。
    
    入力例:
        TrainingArguments(output_dir="./output", fp16=True, ...)
    出力:
        学習設定値を属性として持つインスタンス
    """
    cache_dir: Optional[str] = field(default=None)   # モデルやデータのキャッシュディレクトリ
    optim: str = field(default="adamw_torch")   # オプティマイザの種類
    remove_unused_columns: bool = field(default=False)   # 未使用カラムを削除するか
    freeze_mm_mlp_adapter: bool = field(default=False)   # マルチモーダルMLPアダプタを凍結するか
    mpt_attn_impl: Optional[str] = field(default="triton")   # MPTモデルのAttention実装方式
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )   # モデルの最大シーケンス長
    deepspeed_plugin: any = None   # これを追加することで、deepspeedの引数を受け取れるようにする
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )   # 量子化時のdouble quantizationを有効にするか
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )   # 量子化データ型（fp4/nf4）
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )   # 量子化ビット数（16/8/4など）
    lora_enable: bool = False   # LoRAによる微調整を有効にするか
    lora_r: int = 64   # LoRAのランク
    lora_alpha: int = 16   # LoRAのalpha値
    lora_dropout: float = 0.05   # LoRAのドロップアウト率
    lora_weight_path: str = ""   # LoRA重みのパス
    lora_bias: str = "none"   # LoRAのバイアス設定
    mm_projector_lr: Optional[float] = None   # マルチモーダルプロジェクタの学習率
    group_by_modality_length: bool = field(default=False)   # モダリティごとにバッチをグループ化するか


def maybe_zero_3(param, ignore_status=False, name=None):

    print("current file path", "llava/train/train.py")
    print("def maybe_zero_3(param, ignore_status=False, name=None)")
    print("param\n", param)
    print("ignore_status\n", ignore_status)
    print("name\n", name)
    """
    maybe_zero_3(param, ignore_status=False, name=None)
    DeepSpeed Zero3対応でパラメータを安全に取得する関数。
    
    入力例:
        param: torch.nn.Parameter (DeepSpeedラップ済み)
    出力:
        CPU上のtorch.Tensor (パラメータ値)
    """
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
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):

    print("current file path", "llava/train/train.py")
    print("def get_peft_state_maybe_zero_3(named_params, bias)")
    print("named_params\n", named_params)
    print("bias\n", bias)
    """
    get_peft_state_maybe_zero_3(named_params, bias)
    LoRAパラメータのstate_dictを取得する関数。
    
    入力例:
        named_params: model.named_parameters()のイテレータ
        bias: "none"/"all"/"lora_only"
    出力:
        LoRA層のstate_dict (dict)
    """
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
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
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    print("to_return (return)\n", to_return)
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):

    print("current file path", "llava/train/train.py")
    print("def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True)")
    print("named_params\n", named_params)
    print("require_grad_only\n", require_grad_only)
    """
    get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True)
    LoRA以外のパラメータstate_dictを取得する関数。
    
    入力例:
        named_params: model.named_parameters()のイテレータ
        require_grad_only: True
    出力:
        LoRA以外のパラメータstate_dict (dict)
    """
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    print("to_return (return)\n", to_return)
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):

    print("current file path", "llava/train/train.py")
    print("def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match)")
    print("named_params\n", named_params)
    print("keys_to_match\n", keys_to_match)
    """
    get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match)
    マルチモーダルアダプタのstate_dict取得。
    
    入力例:
        named_params: model.named_parameters()のイテレータ
        keys_to_match: ["mm_projector"]
    出力:
        指定キーワードを含むパラメータのstate_dict (dict)
    """
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    print("to_return (return)\n", to_return)
    return to_return


def find_all_linear_names(model):

    print("current file path", "llava/train/train.py")
    print("def find_all_linear_names(model)")
    print("model\n", model)
    """
    find_all_linear_names(model)
    モデル内のLinear層名を抽出。
    
    入力例:
        model: nn.Module (例: LlamaForCausalLM)
    出力:
        Linear層名のリスト (例: ["q_proj", "k_proj", ...])
    """
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
    print("def safe_save_model_for_hf_trainer(trainer, output_dir)")
    print("trainer\n", "print(trainer) disabled for safety.") 
    print("output_dir\n", output_dir)
    """
    safe_save_model_for_hf_trainer(trainer, output_dir)
    Trainerのモデルを安全に保存。
    
    入力例:
        trainer: transformers.Trainer
        output_dir: "./output"
    出力:
        なし (output_dirにモデル保存)
    """
    """Collects the state dict and dump to disk."""

    """
    【safe_save_model_for_hf_trainer内の主な保存処理 実行可否まとめ】

    | 保存処理                                         | pretraining                  | finetuning            |
    |--------------------------------------------------|------------------------------|-----------------------|
    | tune_mm_mlp_adapter                               | True                         | False                 |
    | mm_use_im_start_end                               | False                        | False                 |
    | trainer.model.config.save_pretrained              | 実行（config.json保存）       | 実行（config.json保存） |
    | torch.save(weight_to_save, mm_projector.bin)      | 実行（mm_projector.bin保存） | 実行されない           |
    | torch.save(weight_to_save, mm_projector_folder/...) | チェックポイント保存時のみ    | 実行されない           |
    | trainer.save_model                                | 実行されない                 | DeepSpeed時のみ        |
    | trainer._save                                     | 実行されない                 | 実行（全重み保存）      |

    【補足】
    - pretraining では mm_projector 等の adapter 層と config のみ保存される  
    - finetuning では通常の全パラメータ保存（trainer._save）が行われる  
    - DeepSpeed 利用時は trainer.save_model で全重み保存  
    - torch.save(weight_to_save, mm_projector_folder/...) は checkpoint 保存時のみ実行  
    """

    # In pretrain.sh --tune_mm_mlp_adapter True \, in finetune.sh, --tune_mm_mlp_adapter False \
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save mm_projector
        keys_to_match = ['mm_projector']
        # both in pretraning and finetuning, use_im_start_end is set to False. This code is usually not executed.
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        # mm_projectorやembed_tokens等のマルチモーダルアダプタ層の重みのみを保存
        # pretraining では weight_to_save の中身は「mm_projector層の全パラメータ（重み・バイアス）」のみです。
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        print("[INFO] Weights to save:", weight_to_save.keys())
        print("weight_to_save\n", weight_to_save)
        # モデルの設定(config)のみ保存
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                # チェックポイントごとにmm_projector等の重みのみ保存
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        print("return (None)")
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        # DeepSpeedを使う場合はtrainer.save_modelを使って保存
        trainer.save_model(output_dir)
    print("return (None)")
    return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        # 全パラメータ(通常のPyTorch state_dict)を保存
        print("cpu_state_dict\n", cpu_state_dict)
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
    print("return (None)")


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    特殊トークン追加時に埋め込みをリサイズ。
    
    入力例:
        special_tokens_dict = {"pad_token": "[PAD]"}
        tokenizer: AutoTokenizer
        model: LlamaForCausalLM
    出力:
        なし (モデルの埋め込み層がリサイズされる)
    """
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    # pretrainingの時のみ、pad_token を追加する。
    # finetuningでは tokenizer に pad_token が既に存在するので、追加しない。
    # finetuningで万が一、pad_tokenが設定されていなかった際の安全策として、tokenizer_unk_token を pad_token に設定する。
    print("current file path", "llava/train/train.py")
    print("def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)")
    print("special_tokens_dict\n", special_tokens_dict)
    print("tokenizer\n", tokenizer)
    print("model\n", model)
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
    print("return (None)")


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """
    _tokenize_fn(strings, tokenizer)
    テキストリストをトークナイズして、学習用の形式に変換。

    入力例:
        strings = ["こんにちは", "さようなら"]

    内部処理イメージ:
        "こんにちは" → [101, 2001, 2002, 2003, 2004, 102]
        "さようなら" → [101, 3001, 3002, 3003, 102, 0]   # PAD 追加

    出力例:
        {
            "input_ids": [
                tensor([101, 2001, 2002, 2003, 2004, 102]),
                tensor([101, 3001, 3002, 3003, 102, 0])
            ],
            "labels": [
                tensor([101, 2001, 2002, 2003, 2004, 102]),
                tensor([101, 3001, 3002, 3003, 102, 0])
            ],
            "input_ids_lens": [6, 5],   # PADを除いた系列長。1文目は6トークン、2文目は5トークン
            "labels_lens":    [6, 5]
        }
    """
    """Tokenize a list of strings."""
    print("current file path", "llava/train/train.py")
    print("def _tokenize_fn(strings, tokenizer)")
    print("strings\n", strings)
    print("tokenizer\n", tokenizer)
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
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    result = dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )
    print("result (return)\n", result)
    return result


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
    print("masked target\n", target)
    return target


def _add_speaker_and_signal(header, source, get_conversation=True):

    print("current file path", "llava/train/train.py")
    print("def _add_speaker_and_signal(header, source, get_conversation=True)")
    print("header\n", header)
    print("source\n", source)
    print("get_conversation\n", get_conversation)
    """
    _add_speaker_and_signal(header, source, get_conversation=True)
    会話データに話者・信号を付与。
    
    入力例:
        header = ""
        source = [{"from": "human", "value": "こんにちは"}, {"from": "gpt", "value": "こんにちは！"}]
    出力:
        付与済み会話テキスト (str)
    """
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
    print("conversation (return)\n", conversation)
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    """
    preprocess_multimodal(sources, data_args)
    マルチモーダルデータの前処理。
    
    入力例:
        sources = [[{"from": "human", "value": "<image> こんにちは"}]]
        data_args: DataArguments
    出力:
        前処理済みsources (list)
    """
    print("current file path", "llava/train/train.py")
    print("def preprocess_multimodal(sources, data_args)")
    print("sources\n", sources)
    print("data_args\n", data_args)
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        print("sources (return)\n", sources)
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    print("sources (return)\n", sources)
    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    preprocess_llama_2(sources, tokenizer, has_image=False)
    Llama2形式の会話データ前処理。
    
    入力例:
        sources = [[{"from": "human", "value": "こんにちは"}, {"from": "gpt", "value": "やあ！"}]]
        tokenizer: AutoTokenizer
        has_image: False
    出力:
        {"input_ids": Tensor, "labels": Tensor}
    """
    print("current file path", "llava/train/train.py")
    print("def preprocess_llama_2(sources, tokenizer, has_image=False)")
    print("sources\n", sources)
    print("tokenizer\n", tokenizer)
    print("has_image\n", has_image)
    """
    preprocess_llama_2(sources, tokenizer, has_image=False)
    Llama2形式の会話データ前処理。
    
    入力例:
        sources = [[{"from": "human", "value": "こんにちは"}, {"from": "gpt", "value": "やあ！"}]]
        tokenizer: AutoTokenizer
        has_image: False
    出力:
        {"input_ids": Tensor, "labels": Tensor}
    """
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

    targets = input_ids.clone()

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

    result = dict(
        input_ids=input_ids,
        labels=targets,
    )
    print("result (return)\n", result)
    return result


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    preprocess_v1(sources, tokenizer, has_image=False)
    v1系会話データ前処理。
    
    入力例:
        sources = [[{"from": "human", "value": "こんにちは"}, {"from": "gpt", "value": "やあ！"}]]
        tokenizer: AutoTokenizer
        has_image: False
    出力:
        {"input_ids": Tensor, "labels": Tensor}
    """
    print("current file path", "llava/train/train.py")
    print("def preprocess_v1(sources, tokenizer, has_image=False)")
    print("sources\n", sources)
    print("tokenizer\n", tokenizer)
    print("has_image\n", has_image)
    """
    preprocess_v1(sources, tokenizer, has_image=False)
    v1系会話データ前処理。
    
    入力例:
        sources = [[{"from": "human", "value": "こんにちは"}, {"from": "gpt", "value": "やあ！"}]]
        tokenizer: AutoTokenizer
        has_image: False
    出力:
        {"input_ids": Tensor, "labels": Tensor}
    """
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

    targets = input_ids.clone()

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

    result = dict(
        input_ids=input_ids,
        labels=targets,
    )
    print("result (return)\n", result)
    return result


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    preprocess_mpt(sources, tokenizer)
    MPT形式の会話データ前処理。
    
    入力例:
        sources = [[{"from": "human", "value": "こんにちは"}, {"from": "gpt", "value": "やあ！"}]]
        tokenizer: AutoTokenizer
    出力:
        {"input_ids": Tensor, "labels": Tensor}
    """
    print("current file path", "llava/train/train.py")
    print("def preprocess_mpt(sources, tokenizer)")
    print("sources\n", sources)
    print("tokenizer\n", tokenizer)
    """
    preprocess_mpt(sources, tokenizer)
    MPT形式の会話データ前処理。
    
    入力例:
        sources = [[{"from": "human", "value": "こんにちは"}, {"from": "gpt", "value": "やあ！"}]]
        tokenizer: AutoTokenizer
    出力:
        {"input_ids": Tensor, "labels": Tensor}
    """
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
    targets = input_ids.clone()
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

    result = dict(
        input_ids=input_ids,
        labels=targets,
    )
    print("result (return)\n", result)
    return result


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    preprocess_plain(sources, tokenizer)
    プレーン形式の会話データ前処理。
    
    入力例:
        sources = [[{"from": "human", "value": "<image>"}, {"from": "gpt", "value": "説明文"}]]
        tokenizer: AutoTokenizer
    出力:
        {"input_ids": [Tensor], "labels": [Tensor]}
    """
    print("current file path", "llava/train/train.py")
    print("def preprocess_plain(sources, tokenizer)")
    print("sources\n", sources)
    print("tokenizer\n", tokenizer)
    """
    preprocess_plain(sources, tokenizer)
    プレーン形式の会話データ前処理。
    
    入力例:
        sources = [[{"from": "human", "value": "<image>"}, {"from": "gpt", "value": "説明文"}]]
        tokenizer: AutoTokenizer
    出力:
        {"input_ids": [Tensor], "labels": [Tensor]}
    """
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    result = dict(input_ids=input_ids, labels=targets)
    print("result (return)\n", result)
    return result


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    preprocess(sources, tokenizer, has_image=False)
    会話データの前処理を形式ごとに分岐。
    
    入力例:
        sources = [[{"from": "human", "value": "こんにちは"}, {"from": "gpt", "value": "やあ！"}]]
        tokenizer: AutoTokenizer
        has_image: False
    出力:
        {"input_ids": [Tensor], "labels": [Tensor]}
    """
    print("current file path", "llava/train/train.py")
    print("def preprocess(sources, tokenizer, has_image=False)")
    print("sources\n", sources)
    print("tokenizer\n", tokenizer)
    print("has_image\n", has_image)
    """
    preprocess(sources, tokenizer, has_image=False)
    会話データの前処理を形式ごとに分岐。
    
    入力例:
        sources = [[{"from": "human", "value": "こんにちは"}, {"from": "gpt", "value": "やあ！"}]]
        tokenizer: AutoTokenizer
        has_image: False
    出力:
        {"input_ids": [Tensor], "labels": [Tensor]}
    """
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        result = preprocess_plain(sources, tokenizer)
        print("result (return)\n", result)
        return result
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        result = preprocess_llama_2(sources, tokenizer, has_image=has_image)
        print("result (return)\n", result)
        return result
    if conversation_lib.default_conversation.version.startswith("v1"):
        result = preprocess_v1(sources, tokenizer, has_image=has_image)
        print("result (return)\n", result)
        return result
    if conversation_lib.default_conversation.version == "mpt":
        result = preprocess_mpt(sources, tokenizer)
        print("result (return)\n", result)
        return result
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
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    result = dict(input_ids=input_ids, labels=targets)
    print("result (return)\n", result)
    return result


class LazySupervisedDataset(Dataset):
    """
    LazySupervisedDataset(Dataset)
    SFT用データセットクラス。
    
    入力例:
        data_path="train.json"
        tokenizer=AutoTokenizer
        data_args=DataArguments
    出力:
        __getitem__でdict(input_ids=Tensor, labels=Tensor, image=Tensor)などを返す
    """
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image_path = os.path.join(image_folder, image_file)
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
            # 画像がなければこのサンプルはスキップ
                return None
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
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
    """
    DataCollatorForSupervisedDataset
    SFT用データコラトラ。
    
    入力例:
        tokenizer=AutoTokenizer
        __call__でinstances=[{"input_ids":Tensor, "labels":Tensor, "image":Tensor}, ...]
    出力:
        バッチ化されたdict(input_ids=Tensor, labels=Tensor, images=Tensor, attention_mask=Tensor)
    """
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
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
    """
    make_supervised_data_module(tokenizer, data_args)
    SFT用データセット・コラトラ生成。
    
    入力例:
        tokenizer=AutoTokenizer
        data_args=DataArguments
    出力:
        {"train_dataset": Dataset, "eval_dataset": None, "data_collator": DataCollatorForSupervisedDataset}
    """
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    """
    train()
    学習メイン関数。
    
    入力例:
        なし (コマンドライン引数で設定)
    出力:
        なし (学習・保存が実行される)
    """
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
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

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
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
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=False) # changed to False
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
