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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):

        print("current file path", "llava/llava/model/language_model/llava_llama.py")
        print("def LlavaLlamaModel.__init__(self, config: LlamaConfig)")
        print("self\n", type(self))
        print("config\n", config)
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):

        print("current file path", "llava/llava/model/language_model/llava_llama.py")
        print("def LlavaLlamaForCausalLM.__init__(self, config)")
        print("self\n", type(self))
        # config は https://huggingface.co/lmsys/vicuna-7b-v1.5/blob/main/config.json
        print("config\n", config)
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        # LlavaLlamaModelの初期化あと、LlavaMetaModelの初期化も呼ばれる。
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        print("self.model\n", self.model)
        """
        self.model
        LlavaLlamaModel(
        (embed_tokens): Embedding(32000, 4096, padding_idx=0)
        (layers): ModuleList(
            (0-31): 32 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
                (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
                (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
                (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
                (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
                (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): LlamaMLP(
                (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
                (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
                (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
                (act_fn): SiLUActivation()
            )
            (input_layernorm): LlamaRMSNorm()
            (post_attention_layernorm): LlamaRMSNorm()
            )
        )
        (norm): LlamaRMSNorm()
        )
        """
        print("self.pretraining_tp\n", self.pretraining_tp) # 1
        print("self.vocab_size\n", self.vocab_size) # 32_000
        print("self.lm_head\n", self.lm_head) # Linear(in_features=4096, out_features=32000, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):

        print("current file path", "llava/llava/model/language_model/llava_llama.py")
        print("def LlavaLlamaForCausalLM.get_model(self)")
        print("self\n", type(self))
        print("self.model (return)\n", self.model)
        """
        LlavaLlamaModel(
        (embed_tokens): Embedding(32000, 4096, padding_idx=0)
        (layers): ModuleList(
            (0-31): 32 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
                (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
                (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
                (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
                (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
                (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): LlamaMLP(
                (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
                (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
                (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
                (act_fn): SiLUActivation()
            )
            (input_layernorm): LlamaRMSNorm()
            (post_attention_layernorm): LlamaRMSNorm()
            )
        )
        (norm): LlamaRMSNorm()
        (vision_tower): CLIPVisionTower(
            (vision_tower): CLIPVisionModel(
            (vision_model): CLIPVisionTransformer(
                (embeddings): CLIPVisionEmbeddings(
                (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
                (position_embedding): Embedding(577, 1024)
                )
                (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (encoder): CLIPEncoder(
                (layers): ModuleList(
                    (0-23): 24 x CLIPEncoderLayer(
                    (self_attn): CLIPAttention(
                        (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
                        (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
                        (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
                        (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
                    )
                    (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                    (mlp): CLIPMLP(
                        (activation_fn): QuickGELUActivation()
                        (fc1): Linear(in_features=1024, out_features=4096, bias=True)
                        (fc2): Linear(in_features=4096, out_features=1024, bias=True)
                    )
                    (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                    )
                )
                )
                (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            )
            )
        )
        (mm_projector): Sequential(
            (0): Linear(in_features=1024, out_features=4096, bias=True)
            (1): GELU(approximate='none')
            (2): Linear(in_features=4096, out_features=4096, bias=True)
        )
        )
        """
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        print("current file path", "llava/llava/model/language_model/llava_llama.py")
        print("def LlavaLlamaForCausalLM.forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, images, image_sizes, return_dict)")
        print("input_ids\n", input_ids)
        if hasattr(input_ids, 'shape'):
            print("input_ids.shape\n", input_ids.shape)
        print("attention_mask\n", attention_mask)
        print("position_ids\n", position_ids)
        print("past_key_values\n", past_key_values)
        print("inputs_embeds\n", inputs_embeds)
        if hasattr(inputs_embeds, 'shape'):
            print("inputs_embeds.shape\n", inputs_embeds.shape)
        print("labels\n", labels)
        print("use_cache\n", use_cache)
        print("output_attentions\n", output_attentions)
        print("output_hidden_states\n", output_hidden_states)
        print("images\n", images)
        if hasattr(images, 'shape'):
            print("images.shape\n", images.shape)
        print("image_sizes\n", image_sizes)
        print("return_dict\n", return_dict)

        print(f"[COND] inputs_embeds_is_None={inputs_embeds is None}")
        if inputs_embeds is None:
            print("【ENTER】if inputs_embeds is None:")
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
            print("【EXIT】if inputs_embeds is None:")

        result = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        print("Return of \ndef LlavaLlamaForCausalLM.forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, images, image_sizes, return_dict)")
        print("result (return)\n", result)
        return result

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        print("current file path", "llava/llava/model/language_model/llava_llama.py")
        print("def LlavaLlamaForCausalLM.generate(self, inputs, images, image_sizes, **kwargs)")
        print("inputs\n", inputs)
        if hasattr(inputs, 'shape'):
            print("inputs.shape\n", inputs.shape)
        print("images\n", images)
        if hasattr(images, 'shape'):
            print("images.shape\n", images.shape)
        print("image_sizes\n", image_sizes)
        print("kwargs\n", kwargs)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        print(f"[COND] 'inputs_embeds'_in_kwargs={'inputs_embeds' in kwargs}")
        if "inputs_embeds" in kwargs:
            print("【ENTER】if 'inputs_embeds' in kwargs:")
            print("print(risk): print(inputs_embeds) disabled for safety")
            raise NotImplementedError("`inputs_embeds` is not supported")
            print("【EXIT】if 'inputs_embeds' in kwargs:")

        print(f"[COND] images_is_not_None={images is not None}")
        if images is not None:
            print("【ENTER】if images is not None:")
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
            print("【EXIT】if images is not None:")
        else:
            print(f"[COND] images_is_None={images is None}")
            print("【ENTER】else of if images is not None:")
            inputs_embeds = self.get_model().embed_tokens(inputs)
            print("【EXIT】else of if images is not None:")

        result = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        print("result (return)\n", result)
        return result

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):

        print("current file path", "llava/llava/model/language_model/llava_llama.py")
        print("def LlavaLlamaForCausalLM.prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs)")
        print("input_ids\n", input_ids)
        if hasattr(input_ids, 'shape'):
            print("input_ids.shape\n", input_ids.shape)
        print("past_key_values\n", past_key_values)
        print("inputs_embeds\n", inputs_embeds)
        if hasattr(inputs_embeds, 'shape'):
            print("inputs_embeds.shape\n", inputs_embeds.shape)
        print("kwargs\n", kwargs)
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        print(f"[COND] images_is_not_None={images is not None}")
        if images is not None:
            print("【ENTER】if images is not None:")
            inputs['images'] = images
            print("【EXIT】if images is not None:")
        print(f"[COND] image_sizes_is_not_None={image_sizes is not None}")
        if image_sizes is not None:
            print("【ENTER】if image_sizes is not None:")
            inputs['image_sizes'] = image_sizes
            print("【EXIT】if image_sizes is not None:")
        print("inputs (return)\n", inputs)
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
