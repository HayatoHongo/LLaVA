import transformers
import torch
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None) # default to None
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

from transformers import HfArgumentParser

args_dict = {
    #"deepspeed": "./scripts/zero2.json",
    "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "version": "plain",
    "data_path": "/workspaces/LLaVA/CC3M_1.json",
    "image_folder": "/workspaces/LLaVA/images/",
    "vision_tower": "openai/clip-vit-large-patch14-336",
    "mm_projector_type": "mlp2x_gelu",
    "tune_mm_mlp_adapter": True,
    "mm_vision_select_layer": -2,
    "mm_use_im_start_end": False,
    "mm_use_im_patch_token": False,
    "bf16": True,
    "output_dir": "./checkpoints/llava-TinyLlama-1.1B-Chat-v1.0",

    # TrainingArguments 相当
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "evaluation_strategy": "no",
    "save_strategy": "steps",
    "save_steps": 1,
    "save_total_limit": 1,
    "learning_rate": 1e-3,
    "weight_decay": 0.0, # I don't know why 0.0
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "logging_steps": 1,
    "tf32": False, # switched from True for TinyLlama
    "model_max_length": 2048,
    "gradient_checkpointing": True,
    "dataloader_num_workers": 2,
    "lazy_preprocess": True,
    "report_to": "none",
}

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import torch.nn as nn
# __init__
# load_model

# result = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            # 【ENTER】
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):            
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            
    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    # image_forward_outs から、指定した層の特徴量 (B, 577, 1024) を取り出したのち、パッチ特徴量 (B, 576, 1024) のみを返す。
    def feature_select(self, image_forward_outs):

        image_features = image_forward_outs.hidden_states[self.select_layer]
        if hasattr(image_features, 'shape'):
            pass
        if self.select_feature == 'patch':
            """
            tensor([[[ 0.2236,  0.2432, -0.5938,  ...,  0.4863, -0.5273, -0.2041],
                    [-0.0469, -0.1836, -0.0273,  ...,  0.3535,  0.3750,  0.3047],
                    [-0.2598,  1.1484,  0.4844,  ...,  0.4961, -0.1719, -0.5117],
                    ...,
                    [ 1.7188,  0.9688,  0.8828,  ..., -0.2441, -0.8672,  1.3047],
                    [ 0.7891, -0.3984,  0.6797,  ..., -0.3594, -0.9922,  0.3164],
                    [ 1.5000,  0.6250,  0.3672,  ..., -0.5469, -0.4902,  0.9766]]],
                device='cuda:0', dtype=torch.bfloat16)
            """
            image_features = image_features[:, 1:]
            """
            tensor([[[-0.0469, -0.1836, -0.0273,  ...,  0.3535,  0.3750,  0.3047],
                    [-0.2598,  1.1484,  0.4844,  ...,  0.4961, -0.1719, -0.5117],
                    [ 1.0625, -0.0635, -0.3730,  ...,  0.0220,  0.0820,  0.4805],
                    ...,
                    [ 1.7188,  0.9688,  0.8828,  ..., -0.2441, -0.8672,  1.3047],
                    [ 0.7891, -0.3984,  0.6797,  ..., -0.3594, -0.9922,  0.3164],
                    [ 1.5000,  0.6250,  0.3672,  ..., -0.5469, -0.4902,  0.9766]]],
                device='cuda:0', dtype=torch.bfloat16)
            """
        elif self.select_feature == 'cls_patch':
            pass
        else:
            pass
        """
        image_features (return)
        tensor([[[-0.0469, -0.1836, -0.0273,  ...,  0.3535,  0.3750,  0.3047],
                [-0.2598,  1.1484,  0.4844,  ...,  0.4961, -0.1719, -0.5117],
                [ 1.0625, -0.0635, -0.3730,  ...,  0.0220,  0.0820,  0.4805],
                ...,
                [ 1.7188,  0.9688,  0.8828,  ..., -0.2441, -0.8672,  1.3047],
                [ 0.7891, -0.3984,  0.6797,  ..., -0.3594, -0.9922,  0.3164],
                [ 1.5000,  0.6250,  0.3672,  ..., -0.5469, -0.4902,  0.9766]]],
            device='cuda:0', dtype=torch.bfloat16)
        """
        if hasattr(image_features, 'shape'):
            pass
        return image_features


    @torch.no_grad() 
    def forward(self, images):

        
        if hasattr(images, 'shape'):
            pass
        if type(images) is list:
            pass
        else:
            # 【ENTER】
            image_forward_outs = self.vision_tower(images.to(device=self.vision_tower.device, dtype=self.vision_tower.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        """
        image_features (return)
        tensor([[[-0.0469, -0.1836, -0.0273,  ...,  0.3535,  0.3750,  0.3047],
                [-0.2598,  1.1484,  0.4844,  ...,  0.4961, -0.1719, -0.5117],
                [ 1.0625, -0.0635, -0.3730,  ...,  0.0220,  0.0820,  0.4805],
                ...,
                [ 1.7188,  0.9688,  0.8828,  ..., -0.2441, -0.8672,  1.3047],
                [ 0.7891, -0.3984,  0.6797,  ..., -0.3594, -0.9922,  0.3164],
                [ 1.5000,  0.6250,  0.3672,  ..., -0.5469, -0.4902,  0.9766]]],
            device='cuda:0', dtype=torch.bfloat16)
        """
        if hasattr(image_features, 'shape'):
            pass
        return image_features

    @property
    def config(self):
        if self.is_loaded:
            # 【ENTER】
            result = self.vision_tower.config
        else:
            pass
        return result

    @property
    def hidden_size(self):
        result = self.config.hidden_size
        return result

    
        
import os

def build_vision_tower(vision_tower_cfg, **kwargs):
    # vision_tower = build_vision_tower(model_args)
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    # ローカルに存在しない場合はFalse。存在する場合の例: /ubuntu/home/user/model/openai/clip-vit-large-patch14-336
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        # 【ENTER】
        result = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        return result
    raise ValueError(f'Unknown vision tower: {vision_tower}')

import re

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    if projector_type == 'linear':
      pass

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        #【ENTER】if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        
        result = nn.Sequential(*modules) # * はリストをアンパックして引数に展開する
        """
        Sequential(
        (0): Linear(in_features=1024, out_features=4096, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=4096, out_features=4096, bias=True)
        )
        """
        return result

    if projector_type == 'identity':
      pass

    raise ValueError(f'Unknown projector type: {projector_type}')

# LlavaMetaModel
# __init__
# get_vision_tower
# initialize_vision_modules
# unpad_image

class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
              pass

    def initialize_vision_modules(self, model_args, fsdp=None):

        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type
        # 下記はself.config.mm_vision_towerに関するもの。self.vision_towerは依然としてNone
        self.config.mm_vision_tower = vision_tower
        
        if self.get_vision_tower() is None:
            #【ENTER】self.vision_tower, self.get_vision_towerはNoneなのでこの分岐に入る。

            # build_vision_tower(model_args) はちょっと奥の依存関係が深い
            vision_tower = build_vision_tower(model_args)
            # 分散学習(FSDP)を使うかどうか. 今回は [] 空のリストとなるので、Noneではないが、len(fsdp) == 0
            if fsdp is not None and len(fsdp) > 0:
                pass
            else:
                # 【ENTER】else of if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = vision_tower
        else:
            pass

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type    

        # mm_projector_is_None=True
        if getattr(self, 'mm_projector', None) is None:
            # 【ENTER】
            self.mm_projector = build_vision_projector(self.config)
            """
            Sequential(
            (0): Linear(in_features=1024, out_features=2048, bias=True)
            (1): GELU(approximate='none')
            (2): Linear(in_features=2048, out_features=2048, bias=True)
            )
            """
            if 'unpad' in mm_patch_merge_type:
                pass
        else:
            pass
        if pretrain_mm_mlp_adapter is not None:
            pass

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            # 【SKIP】
            vision_tower = vision_tower[0]
        return vision_tower



from transformers import LlamaConfig, LlamaModel

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

# LlavaMetaForCausalLM
# get_vision_tower
# encode_images
# prepare_inputs_labels_for_multimodal
# initialize_vision_tokenizer

class LlavaMetaForCausalLM:

    def get_vision_tower(self):
        result = self.get_model().get_vision_tower()
        return result

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            pass

        if model_args.mm_use_im_start_end: # False
            pass

        elif model_args.mm_use_im_patch_token: # False
            pass

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features


    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        """
        llava/llava/model/language_model/llava_llama.py
        """
        """
        tensor([[    1,  -200,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
                10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
                411,  2654, 11315,    13]])
        """

        """
        tensor([[True, True, True, True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True, True, True, True]])
        """

        """
        tensor([[ -100,  -100,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
                10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
                411,  2654, 11315,    13]])
        """


        vision_tower = self.get_vision_tower()

        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            pass

        if type(images) is list or images.ndim == 5:
            pass
        else:
            # 【ENTER】
            image_features = self.encode_images(images)
            """
            tensor([[[-0.1943,  0.1157, -0.0747,  ...,  0.0027, -0.1691, -0.3439],
                    [ 0.0437,  0.1717, -0.0998,  ...,  0.0930, -0.1386, -0.0731],
                    [-0.0505,  0.1592, -0.0982,  ...,  0.0866, -0.1123, -0.2177],
                    ...,
                    [-0.0182,  0.0850, -0.0556,  ...,  0.0622, -0.1969,  0.0129],
                    [-0.0651,  0.0586, -0.1218,  ..., -0.0614, -0.1158, -0.0104],
                    [ 0.0863,  0.0081, -0.1651,  ..., -0.2040, -0.0455,  0.0618]]],
                grad_fn=<ViewBackward0>)
            """

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.

        """
        tensor([[ -100,  -100,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
                10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
                411,  2654, 11315,    13]])
        """


        """
        tensor([[True, True, True, True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True, True, True, True]])
        """

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            pass
        else:
            # 【ENTER】
            attention_mask = attention_mask.bool()
    
            """
            tensor([[True, True, True, True, True, True, True, True, True, True, True, True,
                    True, True, True, True, True, True, True, True, True, True, True, True]])
            """
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

            """
            tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
                    18, 19, 20, 21, 22, 23])
            """
        if labels is None:
            pass

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        """
        [tensor([    1,  -200,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
                10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
                411,  2654, 11315,    13])]
        """

        """
        [tensor([ -100,  -100,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
                10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
                411,  2654, 11315,    13])]
        """


        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            """
            tensor([    1,  -200,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
                    10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
                    411,  2654, 11315,    13])
            """
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
            """
            tensor([ -100,  -100,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
                    10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
                    411,  2654, 11315,    13])
            """
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1): # 2回ループ。1回目 START から IMAGE_TOKEN_INDEXの手前まで、2回目はIMAGE_TOKEN_INDEX より先から 最後まで
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            """
            [tensor([1]), tensor([  278, 25616, 26624,   297,   902, 19430, 11105, 29879, 10508,  1596,
                    23425,   278,  3700,   322,  6567,   310,   263,  6114,   411,  2654,
                    11315,    13])]
            """
            """
            [tensor([-100]), tensor([  278, 25616, 26624,   297,   902, 19430, 11105, 29879, 10508,  1596,
                    23425,   278,  3700,   322,  6567,   310,   263,  6114,   411,  2654,
                    11315,    13])]
            """
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

            """
            [torch.Size([1, 2048]), torch.Size([576, 2048]), torch.Size([22, 2048])]
            """




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
                pass
            else:
                #【ENTER】
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    # :cur_len に、代入
                    new_labels_padded[i, :cur_len] = cur_new_labels 
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)





        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)


        if _labels is None:
            #【SKIP】
            new_labels = None
        else:
            # 【ENTER】
            new_labels = new_labels_padded

        if _attention_mask is None:
            # 【SKIP】
            attention_mask = None
        else:
            # 【ENTER】
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels



from typing import List, Optional, Tuple, Union
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaForCausalLM

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        # LlavaLlamaModelの初期化あと、LlavaMetaModelの初期化も呼ばれる。
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
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


        if hasattr(input_ids, 'shape'):
            pass
        if hasattr(inputs_embeds, 'shape'):
            pass
        if hasattr(images, 'shape'):
            pass

        if inputs_embeds is None:
            # 【ENTER】
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








        #  LlamaForCausalLM.forward(self, ...)で明示
        # Trainer > def train > def inner_training_loop > def training_step > model(**inputs) > model.forward
        result = LlamaForCausalLM.forward(
            self,
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
        return result

def maybe_zero_3(param, ignore_status=False, name=None):

    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"): # TinyLLaVAではdeepspeedを使用しないのでSKIP
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                pass
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):

    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}

    for k, v in to_return.items():
        if hasattr(v, 'shape'):
            pass
            
    return to_return

import dataclasses
from typing import List
from enum import auto, Enum

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False


conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)


conv_templates = {
    "plain": conv_llava_plain,
}

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

from torch.utils.data import Dataset
import json
import copy
from PIL import Image
from typing import Sequence
from typing import Dict

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        # 今回は1サンプルだけなのでprintしても危険ではない
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):

        return len(self.list_data_dict)

    # Trainer > def _get_dataloader > dataloader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))
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
                pass
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            pass

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





# Trainer > def _get_dataloader > dataloader_params = {..."collate_fn": data_collator,...}
# self.accelerator.prepare(DataLoader(dataset, **dataloader_params)) で呼ばれる

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        #  [(torch.Size([24]), torch.Size([24]), torch.Size([3, 336, 336]))]
        # データローダーが None を返すことがあるので、Noneのサンプルを除外。
        instances = [x for x in instances if x is not None]
        # input_idsとlabelsのそれぞれについてリストを作成。タプルをつくる。
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # input_idsはtokenizerのpad_token_id(0)でパディング
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        # labelsはIGNORE_INDEX(-100)でパディング
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        # .ne() は "not equal" → pad_token_id(=0) じゃない部分を 1、pad 部分を 0 にする。モデルが pad 部分を読まないように制御するマスクです。
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
    
from typing import Dict

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:

    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    result = dict(train_dataset=train_dataset,
                  eval_dataset=None,
                  data_collator=data_collator)
    return result

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    logger,
)

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):

    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

import copy

def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer
) -> Dict:

    default_conversation = conv_templates["plain"] # hard coding
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    for idx, tensor in enumerate(input_ids):
        if hasattr(tensor, 'shape'):
            pass
    targets = copy.deepcopy(input_ids)
    for idx, tensor in enumerate(targets):
        if hasattr(tensor, 'shape'):
            pass
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer)) # prompt <image>
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

def _add_speaker_and_signal(header, source, get_conversation=True):
    default_conversation = conv_templates["plain"] # hard coding
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:

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
            pass
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

    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    
    default_conversation = conv_templates["plain"] # hard coding
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if default_conversation.sep_style == SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer) # True
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
        for idx, tensor in enumerate(input_ids):
            if hasattr(tensor, 'shape'):
                pass
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    if isinstance(targets, list):
        for idx, tensor in enumerate(targets):
            if hasattr(tensor, 'shape'):
                pass
    elif hasattr(targets, 'shape'):
        pass
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:

    default_conversation = conv_templates["plain"] # hard coding    
    is_multimodal = data_args.is_multimodal

    if not is_multimodal:
        pass

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources

class LLaVATrainer(Trainer):
    # Trainer > _inner_training_loop > _maybe_log_save_evaluate > self._save_checkpoint(model, trial)
    def _save_checkpoint(self, model, trial, metrics=None):

        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # 【ENTER】
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector'] # 'vision_resampler'
            if getattr(self.args, "use_im_start_end", False):
                pass

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                # 【ENTER】
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            pass

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):

    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        if hasattr(weight_to_save, 'shape'):
            pass
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu() for key, value in state_dict.items()
        }
        for key, value in cpu_state_dict.items():
            if hasattr(value, 'shape'):
                pass
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

import pathlib

def train():

    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(args_dict)
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = {}
    # 【SKIP】bfloat16 なので 以下の if 文はスキップされる
    if training_args.bits in [4, 8]:
      pass

    # 【ENTER】 vision_tower=openai/clip-vit-large-patch14-336 なので、この分岐に入る
    if model_args.vision_tower is not None:
        #【SKIP】model_args.model_name_or_path に mptは含まれていないので、この分岐はskipされる
        if 'mpt' in model_args.model_name_or_path:
          pass

        #【ENTER】 model_args.model_name_or_path に mptは含まれていないので、この分岐に入る
        else:
            # PreTrainedModel.from_pretrained
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    # 【SKIP】 vision_tower=clip-vit-large-patch14-336 なので、この分岐には入らない
    else:
      pass

    # 【SKIP】 freeze_backbone=False なので、この分岐はskipされる
    if model_args.freeze_backbone:
        pass

    # 【SKIP】 bfloat16 なので 以下の if 文はスキップされる
    if training_args.bits in [4, 8]:
      pass

    # 【ENTER】 gradient_checkpointing=True なので、この分岐に入る
    if training_args.gradient_checkpointing:
        # 【ENTER】 model に enable_input_require_grads メソッドがあるので、この分岐に入る
        if hasattr(model, "enable_input_require_grads"):
            # PreTrainedModel.enable_input_require_grads
            # 元々 全ての重みについて True
            model.enable_input_require_grads()
        # 【SKIP】 model に enable_input_require_grads メソッドがあるので、この分岐はskipされる
        else:
          pass


    # 【SKIP】 lora_enable=False なので、この分岐はskipされる
    if training_args.lora_enable:
      pass

    # 【SKIP】model_args.model_name_or_path に mptは含まれていないので、この分岐はskipされる
    if 'mpt' in model_args.model_name_or_path:
      pass

    #【ENTER】 model_args.model_name_or_path に mptは含まれていないので、この分岐に入る
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    # 【SKIP】 version=plain なので、この分岐はskipされる
    if model_args.version == "v0":
      pass

    # 【SKIP】 version=plain なので、この分岐はskipされる
    elif model_args.version == "v0.5":
      pass
    # 【ENTER】 version=plain なので、この分岐に入る
    else:
        tokenizer.pad_token = tokenizer.unk_token
        # 【ENTER】 model_args.version=plain は conversation_lib.conv_templates に含まれている（"plain": conv_llava_plain）ので、この分岐に入る
        if model_args.version in conv_templates:
            default_conversation = conv_templates[model_args.version]
        # 【SKIP】 model_args.version=plain は conversation_lib.conv_templates に含まれているので、この分岐はskipされる
        else:
          pass

    # 【ENTER】 vision_tower=openai/clip-vit-large-patch14-336 なので、この分岐に入る
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
            # 【ENTER】 tune_mm_mlp_adapter=True なので、この分岐に入る
            # モデル全体の全パラメータを「学習不可（requires_grad=False）」にする
            # これで通常の重みは全て凍結される
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                # mm_projector（画像特徴量→テキスト特徴量への変換層）の全パラメータだけを「学習可能（requires_grad=True）」に戻す
                # これで mm_projector のみ学習されることになる
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
          pass

        if training_args.bits in [4, 8]:
          pass

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        pass

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        # 【ENTER】
        trainer.train(resume_from_checkpoint=False)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
      pass
    else:
        # 【ENTER】
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()