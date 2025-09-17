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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):

        print("current file path", "llava/model/llava_arch.py")
        print("LlavaMetaModel.__init__(self, config)")
        print("config\n", config)
        # LlamaModelの__init_を呼び出す 
        super(LlavaMetaModel, self).__init__(config)

        print(f"[COND] mm_vision_tower={hasattr(config, 'mm_vision_tower')}")
        if hasattr(config, "mm_vision_tower"):
            print("【ENTER】if hasattr(config, 'mm_vision_tower'):")
            self.vision_tower = build_vision_tower(config, delay_load=True)
            print("self.vision_tower\n", self.vision_tower)
            self.mm_projector = build_vision_projector(config)
            print("self.mm_projector\n", self.mm_projector)

            print(f"[COND] unpad_in_mm_patch_merge_type={'unpad' in getattr(config, 'mm_patch_merge_type', '')}")
            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                print("【ENTER】if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):")
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )
                print("【EXIT】if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):")
            print("【EXIT】if hasattr(config, 'mm_vision_tower'):")

    def get_vision_tower(self):

        print("current file path", "llava/model/llava_arch.py")
        print("def get_vision_tower(self)")
        vision_tower = getattr(self, 'vision_tower', None)
        print("vision_tower (raw)\n", vision_tower)
        """
        CLIPVisionTower(
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
        """
        print("type(vision_tower)\n", type(vision_tower))
        print(f"[COND] type_vision_tower_is_list={type(vision_tower) is list}")  # False
        if type(vision_tower) is list:
            # 【SKIP】
            print("【ENTER】if type(vision_tower) is list:")
            vision_tower = vision_tower[0]
            print("【EXIT】if type(vision_tower) is list:")
        print("vision_tower (return)\n", vision_tower)
        """
        vision_tower (return)
        CLIPVisionTower(
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
        """
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):

        print("current file path", "llava/model/llava_arch.py")
        print("def initialize_vision_modules(self, model_args, fsdp=None)")
        print("model_args\n", model_args)
        print("fsdp\n", fsdp)
        vision_tower = model_args.vision_tower
        print("vision_tower from model_args\n", vision_tower)
        mm_vision_select_layer = model_args.mm_vision_select_layer
        print("mm_vision_select_layer from model_args\n", mm_vision_select_layer)
        mm_vision_select_feature = model_args.mm_vision_select_feature
        print("mm_vision_select_feature from model_args\n", mm_vision_select_feature)
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        print("pretrain_mm_mlp_adapter from model_args\n", pretrain_mm_mlp_adapter)
        mm_patch_merge_type = model_args.mm_patch_merge_type
        # 下記はself.config.mm_vision_towerに関するもの。self.vision_towerは依然としてNone
        self.config.mm_vision_tower = vision_tower
        print("self.config.mm_vision_tower\n", self.config.mm_vision_tower)

        print("[COND] self.get_vision_tower()\n", self.get_vision_tower())
        print(f"[COND] get_vision_tower_is_None={self.get_vision_tower() is None}")
        if self.get_vision_tower() is None:
            #【ENTER】self.vision_tower, self.get_vision_towerはNoneなのでこの分岐に入る。
            print("【ENTER】if self.get_vision_tower() is None:")
            print("[ENTER] self.get_vision_tower() is None")
            # build_vision_tower(model_args) はちょっと奥の依存関係が深い
            vision_tower = build_vision_tower(model_args)
            print("vision_tower after build_vision_tower\n", vision_tower)
            """
            CLIPVisionTower(
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
            """
            # 分散学習(FSDP)を使うかどうか. 今回は [] 空のリストとなるので、Noneではないが、len(fsdp) == 0
            print("[COND] fsdp\n", fsdp) # []
            print(f"[COND] fsdp_is_not_None={fsdp is not None} len_fsdp={len(fsdp) if fsdp is not None else 'N/A'}") # fsdp_is_not_None=True len_fsdp=0
            if fsdp is not None and len(fsdp) > 0:
                # 【SKIP】
                print("【ENTER】if fsdp is not None and len(fsdp) > 0:")
                print("[COND] len(fsdp)\n", len(fsdp))
                print("[ENTER] if fsdp is not None and len(fsdp) > 0:")
                self.vision_tower = [vision_tower]
                print("self.vision_tower\n", self.vision_tower)
                print("【EXIT】if fsdp is not None and len(fsdp) > 0:")
            else:
                # 【ENTER】else of if fsdp is not None and len(fsdp) > 0:
                print("[COND] else_fsdp_is_not_None_and_len_fsdp_gt_0=True")
                print("【ENTER】else of if fsdp is not None and len(fsdp) > 0:")
                self.vision_tower = vision_tower
                print("self.vision_tower\n", self.vision_tower)
                """
                CLIPVisionTower(
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
                """
                print("【EXIT】else of if fsdp is not None and len(fsdp) > 0:")

            print("【EXIT】if self.get_vision_tower() is None:")
        else:
            # 【SKIP】
            print("[COND] else_get_vision_tower_is_None=True")
            print("【ENTER】else of if self.get_vision_tower() is None:")
            print("vision_tower before load_model\n", self.get_vision_tower())
            print(f"[COND] fsdp_is_not_None={fsdp is not None} len_fsdp={len(fsdp) if fsdp is not None else 'N/A'}")
            if fsdp is not None and len(fsdp) > 0:
                print("【ENTER】if fsdp is not None and len(fsdp) > 0:")
                vision_tower = self.vision_tower[0]
                print("vision_tower\n", vision_tower)
                print("【EXIT】if fsdp is not None and len(fsdp) > 0:")
            else:
                print("[COND] else_fsdp_is_not_None_and_len_fsdp_gt_0=True")
                print("【ENTER】else of if fsdp is not None and len(fsdp) > 0:")
                vision_tower = self.vision_tower
                print("vision_tower\n", vision_tower)
                print("【EXIT】else of if fsdp is not None and len(fsdp) > 0:")
            vision_tower.load_model()
            print("vision_tower after load_model\n", vision_tower)
            print("【EXIT】else of if self.get_vision_tower() is None:")

        self.config.use_mm_proj = True
        print("self.config.use_mm_proj set to True") # True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        print("self.config.mm_projector_type\n", self.config.mm_projector_type) # mlp2x_gelu
        self.config.mm_hidden_size = vision_tower.hidden_size 
        print("self.config.mm_hidden_size\n", self.config.mm_hidden_size) # 1024
        self.config.mm_vision_select_layer = mm_vision_select_layer 
        print("self.config.mm_vision_select_layer\n", self.config.mm_vision_select_layer) # -2
        self.config.mm_vision_select_feature = mm_vision_select_feature
        print("self.config.mm_vision_select_feature\n", self.config.mm_vision_select_feature) # patch
        self.config.mm_patch_merge_type = mm_patch_merge_type
        print("self.config.mm_patch_merge_type\n", self.config.mm_patch_merge_type) # flat

        # mm_projector_is_None=True
        print(f"[COND] mm_projector_is_None={getattr(self, 'mm_projector', None) is None}")
        if getattr(self, 'mm_projector', None) is None:
            # 【ENTER】
            print("【ENTER】if getattr(self, 'mm_projector', None) is None:")
            self.mm_projector = build_vision_projector(self.config)
            print("self.mm_projector after build_vision_projector\n", self.mm_projector)
            print("mm_patch_merge_type\n", mm_patch_merge_type) # flat
            print(f"[COND] unpad_in_mm_patch_merge_type={'unpad' in mm_patch_merge_type}")
            if 'unpad' in mm_patch_merge_type:
                # 【SKIP】
                print("【ENTER】if 'unpad' in mm_patch_merge_type:")
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
                print("【EXIT】if 'unpad' in mm_patch_merge_type:")
            print("【EXIT】if getattr(self, 'mm_projector', None) is None:")
        else:
            # 【SKIP】
            # In case it is frozen by LoRA
            print("[COND] else_mm_projector_is_None=True")
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        print(f"[COND] pretrain_mm_mlp_adapter_is_not_None={pretrain_mm_mlp_adapter is not None}")
        if pretrain_mm_mlp_adapter is not None:
            print("【ENTER】if pretrain_mm_mlp_adapter is not None:")
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                print("current file path", "llava/model/llava_arch.py")
                print("def get_w(weights, keyword)")
                print("weights\n", weights)
                print("keyword\n", keyword)
                result = {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                print("result (return)\n", result)
                return result

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
            print("【EXIT】if pretrain_mm_mlp_adapter is not None:")


def unpad_image(tensor, original_size):
    print("current file path", "llava/model/llava_arch.py")
    print("def unpad_image(tensor, original_size)")
    print("tensor\n", tensor)
    print("original_size\n", original_size)
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    print(f"[COND] original_aspect_ratio_gt_current_aspect_ratio={original_aspect_ratio > current_aspect_ratio}")
    if original_aspect_ratio > current_aspect_ratio:
        print("【ENTER】if original_aspect_ratio > current_aspect_ratio:")
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        print("【ENTER】else of if original_aspect_ratio > current_aspect_ratio:")
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]
        print("【EXIT】else of if original_aspect_ratio > current_aspect_ratio:")

    print("unpadded_tensor (return)\n", unpadded_tensor)
    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        print("current file path", "llava/model/llava_arch.py")
        print("class LlavaMetaForCausalLM(ABC).get_model(self)")
        pass

    def get_vision_tower(self):
        print("current file path", "llava/model/llava_arch.py")
        print("class LlavaMetaForCausalLM(ABC).get_vision_tower(self)")
        result = self.get_model().get_vision_tower()
        print("LlavaMetaForCausalLM(ABC).get_vision_tower(self) result (return)\n", result)
        """
        CLIPVisionTower(
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
        """
        return result

    def encode_images(self, images):
        print("current file path", "llava/model/llava_arch.py")
        print("def LlavaMetaForCausalLM(ABC).encode_images(self, images)")
        print("images\n", images)
        """
        tensor([[[[ 0.0325,  0.0325,  0.0325,  ..., -0.7109, -0.3613, -0.1279],
                [ 0.0325,  0.0325,  0.0325,  ..., -0.3906, -0.1719, -0.0259],
                [ 0.0325,  0.0325,  0.0325,  ..., -0.0112,  0.0471,  0.0908],
                ...,
                [-1.0312, -1.0312, -1.0312,  ..., -1.0625, -1.0625, -1.0625],
                [-1.0469, -1.0312, -1.0312,  ..., -1.0625, -1.0625, -1.0625],
                [-1.0469, -1.0312, -1.0312,  ..., -1.0625, -1.0625, -1.0625]],

                [[ 0.3184,  0.3184,  0.3184,  ..., -0.3867, -0.0112,  0.2139],
                [ 0.3184,  0.3184,  0.3184,  ..., -0.0713,  0.1543,  0.3184],
                [ 0.3184,  0.3184,  0.3184,  ...,  0.2891,  0.3633,  0.4395],
                ...,
                [-1.0156, -1.0156, -1.0156,  ..., -1.0000, -1.0000, -1.0000],
                [-1.0312, -1.0156, -1.0156,  ..., -1.0000, -1.0000, -1.0000],
                [-1.0312, -1.0156, -1.0156,  ..., -1.0000, -1.0000, -1.0000]],

                [[ 0.9648,  0.9648,  0.9648,  ...,  0.0981,  0.4531,  0.6680],
                [ 0.9648,  0.9648,  0.9648,  ...,  0.3965,  0.6094,  0.7539],
                [ 0.9648,  0.9648,  0.9648,  ...,  0.7539,  0.8086,  0.8359],
                ...,
                [-0.3711, -0.3848, -0.4004,  ..., -0.4277, -0.4277, -0.4277],
                [-0.3711, -0.3711, -0.3848,  ..., -0.4277, -0.4277, -0.4277],
                [-0.3848, -0.3711, -0.3711,  ..., -0.4277, -0.4277, -0.4277]]]],
            device='cuda:0', dtype=torch.bfloat16)
        """

        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        print("image_features (return)\n", image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        print("current file path", "llava/model/llava_arch.py")
        print("def LlavaMetaForCausalLM(ABC).prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=None)")
        print("input_ids\n", input_ids)
        """
        tensor([[    1,  -200,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
        10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
        411,  2654, 11315,    13]], device='cuda:0')
        """
        print("position_ids\n", position_ids) # None
        print("attention_mask\n", attention_mask)
        """
        tensor([[True, True, True, True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True, True, True, True]],
            device='cuda:0')     
        """
        print("past_key_values\n", past_key_values) # None
        print("labels\n", labels)
        """
        tensor([[ -100,  -100,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
                10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
                411,  2654, 11315,    13]], device='cuda:0')
        """
        print("images\n", images)
        """
        tensor([[[[ 0.0325,  0.0325,  0.0325,  ..., -0.7109, -0.3613, -0.1279],
                [ 0.0325,  0.0325,  0.0325,  ..., -0.3906, -0.1719, -0.0259],
                [ 0.0325,  0.0325,  0.0325,  ..., -0.0112,  0.0471,  0.0908],
                ...,
                [-1.0312, -1.0312, -1.0312,  ..., -1.0625, -1.0625, -1.0625],
                [-1.0469, -1.0312, -1.0312,  ..., -1.0625, -1.0625, -1.0625],
                [-1.0469, -1.0312, -1.0312,  ..., -1.0625, -1.0625, -1.0625]],

                [[ 0.3184,  0.3184,  0.3184,  ..., -0.3867, -0.0112,  0.2139],
                [ 0.3184,  0.3184,  0.3184,  ..., -0.0713,  0.1543,  0.3184],
                [ 0.3184,  0.3184,  0.3184,  ...,  0.2891,  0.3633,  0.4395],
                ...,
                [-1.0156, -1.0156, -1.0156,  ..., -1.0000, -1.0000, -1.0000],
                [-1.0312, -1.0156, -1.0156,  ..., -1.0000, -1.0000, -1.0000],
                [-1.0312, -1.0156, -1.0156,  ..., -1.0000, -1.0000, -1.0000]],

                [[ 0.9648,  0.9648,  0.9648,  ...,  0.0981,  0.4531,  0.6680],
                [ 0.9648,  0.9648,  0.9648,  ...,  0.3965,  0.6094,  0.7539],
                [ 0.9648,  0.9648,  0.9648,  ...,  0.7539,  0.8086,  0.8359],
                ...,
                [-0.3711, -0.3848, -0.4004,  ..., -0.4277, -0.4277, -0.4277],
                [-0.3711, -0.3711, -0.3848,  ..., -0.4277, -0.4277, -0.4277],
                [-0.3848, -0.3711, -0.3711,  ..., -0.4277, -0.4277, -0.4277]]]],
            device='cuda:0', dtype=torch.bfloat16)
        """
        print("image_sizes\n", image_sizes) # None
        vision_tower = self.get_vision_tower()
        print("vision_tower\n", vision_tower)
        """
        LlavaMetaForCausalLM(ABC).get_vision_tower(self) result (return)
        CLIPVisionTower(
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
        """
        print(f"[COND] vision_tower_is_None={vision_tower is None} images_is_None={images is None} input_ids_shape_1_eq_1={input_ids.shape[1] == 1}")
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            # 【SKIP】
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        print("【COND】type(images)\n", type(images))
        print("【COND】images.ndim\n", images.ndim)
        if type(images) is list or images.ndim == 5:
            # 【SKIP】
            print("【ENTER】if type(images) is list or images.ndim == 5:")
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None]
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            print("image_features after processing\n", image_features)
            print("【EXIT】if type(images) is list or images.ndim == 5:")
        else:
            # 【ENTER】
            print("【ENTER】else of if type(images) is list or images.ndim == 5:")
            image_features = self.encode_images(images)
            print("image_features after encode_images\n", image_features)
            print("【EXIT】else of if type(images) is list or images.ndim == 5:")

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        print("position_ids (return)\n", position_ids)
        print("attention_mask (return)\n", attention_mask)
        print("past_key_values (return)\n", past_key_values)
        print("new_input_embeds (return)\n", new_input_embeds)
        print("new_labels (return)\n", new_labels)
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        print("current file path", "llava/model/llava_arch.py")
        print("def initialize_vision_tokenizer(self, model_args, tokenizer)")
        print("model_args\n", model_args) # ModelArguments(model_name_or_path='lmsys/vicuna-7b-v1.5', version='plain', freeze_backbone=False, tune_mm_mlp_adapter=True, vision_tower='openai/clip-vit-large-patch14-336', mm_vision_select_layer=-2, pretrain_mm_mlp_adapter=None, mm_projector_type='mlp2x_gelu', mm_use_im_start_end=False, mm_use_im_patch_token=False, mm_patch_merge_type='flat', mm_vision_select_feature='patch')
        print("tokenizer\n", tokenizer) # LlamaTokenizer(name_or_path='lmsys/vicuna-7b-v1.5', vocab_size=32000, model_max_length=2048, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'bos_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'eos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'unk_token': AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False), 'pad_token': '<unk>'}, clean_up_tokenization_spaces=False)

        print(f"[COND] mm_use_im_patch_token={model_args.mm_use_im_patch_token}") # False
        if model_args.mm_use_im_patch_token:
            # 【SKIP】
            print("【ENTER】if mm_use_im_patch_token:")
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            print("【EXIT】if mm_use_im_patch_token:")

        if model_args.mm_use_im_start_end: # False
            # 【SKIP】
            print("【ENTER】if model_args.mm_use_im_start_end:")
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
            print("【EXIT】if model_args.mm_use_im_start_end:")

        elif model_args.mm_use_im_patch_token: # False
            # 【SKIP】
            print("【ENTER】elif mm_use_im_patch_token:")
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            print("【EXIT】elif mm_use_im_patch_token:")
