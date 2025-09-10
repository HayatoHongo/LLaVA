#    Copyright 2023 Haotian Liu
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
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", **kwargs):

    print("current file path", "llava/model/builder.py")
    print("def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map=\"auto\", device=\"cuda\", **kwargs)")
    print("model_path\n", model_path)
    print("model_base\n", model_base)
    print("model_name\n", model_name)
    print("load_8bit\n", load_8bit)
    print("load_4bit\n", load_4bit)
    print("device_map\n", device_map)
    print("device\n", device)
    print("kwargs\n", kwargs)

    kwargs = {"device_map": device_map, **kwargs}
    print("kwargs\n", kwargs)

    print(f"[COND] device={device}")
    if device != "cuda":
        print("【ENTER】if device != 'cuda':")
        kwargs['device_map'] = {"": device}
        print("【EXIT】if device != 'cuda':")

    print(f"[COND] load_8bit={load_8bit} load_4bit={load_4bit}")
    if load_8bit:
        print("【ENTER】if load_8bit:")
        kwargs['load_in_8bit'] = True
        print("【EXIT】if load_8bit:")
    elif load_4bit:
        print("【ENTER】elif load_4bit:")
        kwargs['load_in_4bit'] = True
        print("kwargs['load_in_4bit']\n", kwargs['load_in_4bit'])
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        print("kwargs['quantization_config']\n", kwargs['quantization_config'])
        print("【EXIT】elif load_4bit:")
    else:
        print("【ENTER】else of if load_8bit/elif load_4bit:")
        kwargs['torch_dtype'] = torch.float16
        print("【EXIT】else of if load_8bit/elif load_4bit:")

    print(f"[COND] 'llava'_in_model_name={('llava' in model_name.lower())}")
    if 'llava' in model_name.lower():
        print("【ENTER】if 'llava' in model_name.lower():")
        # Load LLaVA model
        print(f"[COND] 'lora'_in_model_name={'lora' in model_name.lower()} model_base_is_None={model_base is None}")
        if 'lora' in model_name.lower() and model_base is None:
            print("【ENTER】if 'lora' in model_name.lower() and model_base is None:")
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
            print("【EXIT】if 'lora' in model_name.lower() and model_base is None:")
        print(f"[COND] 'lora'_in_model_name={'lora' in model_name.lower()} model_base_is_not_None={model_base is not None}")
        if 'lora' in model_name.lower() and model_base is not None:
            print("【ENTER】if 'lora' in model_name.lower() and model_base is not None:")
            lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
            print("lora_cfg_pretrained\n", lora_cfg_pretrained)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print("tokenizer\n", tokenizer)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            print("model\n", model)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            print("token_num\n", token_num)
            print("tokem_dim\n", tokem_dim)
            print(f"[COND] lm_head_shape_match={model.lm_head.weight.shape[0] == token_num}")
            if model.lm_head.weight.shape[0] != token_num:
                print("【ENTER】if model.lm_head.weight.shape[0] != token_num:")
                print("original model.lm_head.weight\n", type(model.lm_head.weight))
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                print("after process model.lm_head.weight\n", type(model.lm_head.weight))
                print("original model.model.embed_tokens.weight\n", type(model.model.embed_tokens.weight))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                print("after process model.model.embed_tokens.weight\n", type(model.model.embed_tokens.weight))
                print("【EXIT】if model.lm_head.weight.shape[0] != token_num:")

            print('Loading additional LLaVA weights...')
            print(f"[COND] non_lora_trainables_exists={os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin'))}")
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                print("【ENTER】if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):")
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
                print("non_lora_trainables (keys)\n", list(non_lora_trainables.keys()))
                print("【EXIT】if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):")
            else:
                print("【ENTER】else of if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):")
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    print("current file path", "llava/model/builder.py")
                    print("def load_from_hf(repo_id, filename, subfolder=None)")
                    print("repo_id\n", repo_id)
                    print("filename\n", filename)
                    print("subfolder\n", subfolder)
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    print("cache_file\n", cache_file)
                    result = torch.load(cache_file, map_location='cpu')
                    print("result (return)\n", result)
                    return result
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
                print("non_lora_trainables (keys)\n", list(non_lora_trainables.keys()))
                print("【EXIT】else of if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):")
            print("original non_lora_trainables (keys)\n", list(non_lora_trainables.keys()))
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            print("after process non_lora_trainables (keys)\n", list(non_lora_trainables.keys()))
            print(f"[COND] any_model_model_in_non_lora_trainables={any(k.startswith('model.model.') for k in non_lora_trainables)}")
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                print("【ENTER】if any(k.startswith('model.model.') for k in non_lora_trainables):")
                print("original non_lora_trainables (keys)\n", list(non_lora_trainables.keys()))
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                print("after process non_lora_trainables (keys)\n", list(non_lora_trainables.keys()))
                print("【EXIT】if any(k.startswith('model.model.') for k in non_lora_trainables):")
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print("model (after PeftModel.from_pretrained)\n", model)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print("model (after merge_and_unload)\n", model)
            print('Model is loaded...')
        elif model_base is not None:
            print(f"[COND] model_base_is_not_None={model_base is not None}")
            print("【ENTER】elif model_base is not None:")
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            print(f"[COND] 'mpt'_in_model_name={'mpt' in model_name.lower()}")
            if 'mpt' in model_name.lower():
                print("【ENTER】if 'mpt' in model_name.lower():")
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    print("【ENTER】if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):")
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                    print("【EXIT】if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):")
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                print("tokenizer\n", tokenizer)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                print("cfg_pretrained\n", cfg_pretrained)
                model = LlavaMPTForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                print("model\n", model)
                print("【EXIT】if 'mpt' in model_name.lower():")
            else:
                print("【ENTER】else of if 'mpt' in model_name.lower():")
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                print("tokenizer\n", tokenizer)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                print("cfg_pretrained\n", cfg_pretrained)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                print("model\n", model)
                print("【EXIT】else of if 'mpt' in model_name.lower():")

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            print("mm_projector_weights (keys)\n", list(mm_projector_weights.keys()))
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            print("mm_projector_weights (after process, keys)\n", list(mm_projector_weights.keys()))
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            print(f"[COND] model_base_is_None={model_base is None}")
            print("【ENTER】else of if model_base is not None:")
            print(f"[COND] 'mpt'_in_model_name={'mpt' in model_name.lower()} 'mistral'_in_model_name={'mistral' in model_name.lower()}")
            if 'mpt' in model_name.lower():
                print("【ENTER】if 'mpt' in model_name.lower():")
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                print("tokenizer\n", tokenizer)
                model = LlavaMPTForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                print("model\n", model)
                print("【EXIT】if 'mpt' in model_name.lower():")
            elif 'mistral' in model_name.lower():
                print("【ENTER】elif 'mistral' in model_name.lower():")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                print("tokenizer\n", tokenizer)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    use_flash_attention_2=False,
                    **kwargs
                )
                print("model\n", model)
                print("【EXIT】elif 'mistral' in model_name.lower():")
            else:
                print("【ENTER】else of if/elif 'mpt'/'mistral' in model_name.lower():")
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                print("tokenizer\n", tokenizer)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                print("model\n", model)
                print("【EXIT】else of if/elif 'mpt'/'mistral' in model_name.lower():")
            print("【EXIT】else of if model_base is not None:")
    else:
        print(f"[COND] 'llava'_not_in_model_name={not ('llava' in model_name.lower())}")
        print("【ENTER】else of if 'llava' in model_name.lower():")
        # Load language model
        if model_base is not None:
            print(f"[COND] model_base_is_not_None={model_base is not None}")
            print("【ENTER】if model_base is not None:")
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print("tokenizer\n", tokenizer)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print("model\n", model)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print("model (after PeftModel.from_pretrained)\n", model)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("model (after merge_and_unload)\n", model)
            print('Convert to FP16...')
            print("original model dtype\n", model.dtype)
            model.to(torch.float16)
            print("after process model dtype\n", model.dtype)
        else:
            use_fast = False
            print(f"[COND] 'mpt'_in_model_name={'mpt' in model_name.lower()}")
            if 'mpt' in model_name.lower():
                print("【ENTER】if 'mpt' in model_name.lower():")
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                print("tokenizer\n", tokenizer)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
                print("model\n", model)
                print("【EXIT】if 'mpt' in model_name.lower():")
            else:
                print("【ENTER】else of if 'mpt' in model_name.lower():")
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                print("tokenizer\n", tokenizer)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
                print("model\n", model)
                print("【EXIT】else of if 'mpt' in model_name.lower():")
            print("【EXIT】if model_base is not None:")

    image_processor = None
    print("image_processor\n", image_processor)
    
    print(f"[COND] 'llava'_in_model_name={('llava' in model_name.lower())}")
    if 'llava' in model_name.lower():
        print("【ENTER】if 'llava' in model_name.lower():")
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        print("mm_use_im_start_end\n", mm_use_im_start_end)
        print("mm_use_im_patch_token\n", mm_use_im_patch_token)
        print("【COND】mm_use_im_patch_token={mm_use_im_patch_token}")
        if mm_use_im_patch_token:
            print("【ENTER】if mm_use_im_patch_token:")
            print("original tokenizer vocab size\n", len(tokenizer))
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            print("tokenizer\n", tokenizer)
            print("after process tokenizer vocab size\n", len(tokenizer))
            print("【EXIT】if mm_use_im_patch_token:")
        print("【COND】mm_use_im_start_end={mm_use_im_start_end}")
        if mm_use_im_start_end:
            print("【ENTER】if mm_use_im_start_end:")
            print("original tokenizer vocab size\n", len(tokenizer))
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            print("tokenizer\n", tokenizer)
            print("after process tokenizer vocab size\n", len(tokenizer))
            print("【EXIT】if mm_use_im_start_end:")
        model.resize_token_embeddings(len(tokenizer))
        print("model.resize_token_embeddings(len(tokenizer)) executed", model)
        print("【EXIT】if 'llava' in model_name.lower():")

        vision_tower = model.get_vision_tower()
        print("vision_tower\n", vision_tower)
        print(f"[COND] vision_tower_is_loaded={not vision_tower.is_loaded}")
        if not vision_tower.is_loaded:
            print("【ENTER】if not vision_tower.is_loaded:")
            vision_tower.load_model()
            print("vision_tower (after load_model)\n", vision_tower)
            print("【EXIT】if not vision_tower.is_loaded:")
        print("original vision_tower dtype\n", getattr(vision_tower, 'dtype', 'N/A'))
        vision_tower.to(device=device, dtype=torch.float16)
        print("after process vision_tower dtype\n", getattr(vision_tower, 'dtype', 'N/A'))
        print("vision tower after to()\n", vision_tower)
        image_processor = vision_tower.image_processor
        print("image_processor\n", image_processor)
        print("【EXIT】if 'llava' in model_name.lower():")

    if hasattr(model.config, "max_sequence_length"):
        print("【ENTER】if hasattr(model.config, 'max_sequence_length'):")
        context_len = model.config.max_sequence_length
        print("context_len\n", context_len)
        print("【EXIT】if hasattr(model.config, 'max_sequence_length'):")
    else:
        print("【ENTER】else of if hasattr(model.config, 'max_sequence_length'):")
        context_len = 2048
        print("context_len\n", context_len)
        print("【EXIT】else of if hasattr(model.config, 'max_sequence_length'):")

    print("tokenizer\n", tokenizer)
    print("model\n", model)
    print("image_processor\n", image_processor)
    print("context_len\n", context_len)
    return tokenizer, model, image_processor, context_len
