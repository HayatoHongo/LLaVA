import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):

        print("current file path", "llava/llava/model/multimodal_encoder/clip_encoder.py")
        print("def CLIPVisionTower.__init__(self, vision_tower, args, delay_load=False)")
        print("self\n", type(self)) 
        print("vision_tower\n", vision_tower) # openai/clip-vit-large-patch14-336
        print("args\n", args) #  ModelArguments(model_name_or_path='lmsys/vicuna-7b-v1.5', version='plain', freeze_backbone=False, tune_mm_mlp_adapter=True, vision_tower='openai/clip-vit-large-patch14-336', mm_vision_select_layer=-2, pretrain_mm_mlp_adapter=None, mm_projector_type='mlp2x_gelu', mm_use_im_start_end=False, mm_use_im_patch_token=False, mm_patch_merge_type='flat', mm_vision_select_feature='patch')
        print("delay_load\n", delay_load) # False
        super().__init__()

        self.is_loaded = False
        print("self.is_loaded\n", self.is_loaded) # False

        self.vision_tower_name = vision_tower
        print("self.vision_tower_name\n", self.vision_tower_name) # openai/clip-vit-large-patch14-336
        self.select_layer = args.mm_vision_select_layer
        print("self.select_layer\n", self.select_layer) # -2
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch') 
        print("self.select_feature\n", self.select_feature) # patch
        
        print(f"【COND】 delay_load={delay_load}")
        if not delay_load:
            print("【ENTER】if not delay_load:")
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            print("【ENTER】elif getattr(args, 'unfreeze_mm_vision_tower', False):")
            self.load_model()
            print("【EXIT】elif getattr(args, 'unfreeze_mm_vision_tower', False):")
        else:
            print("【ENTER】else of if not delay_load/elif getattr(args, 'unfreeze_mm_vision_tower', False):")
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            print("self.cfg_only\n", self.cfg_only)
            print("【EXIT】else of if not delay_load/elif getattr(args, 'unfreeze_mm_vision_tower', False):")

    def load_model(self):

        print("current file path", "llava/llava/model/multimodal_encoder/clip_encoder.py")
        print("def CLIPVisionTower.load_model(self)")
        print("self\n", type(self))
        print("self.vision_tower_name\n", self.vision_tower_name) # openai/clip-vit-large-patch14-336
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        print("self.image_processor\n", self.image_processor)
        """
        CLIPImageProcessor {
        "crop_size": {
            "height": 336,
            "width": 336
        },
        "do_center_crop": true,
        "do_convert_rgb": true,
        "do_normalize": true,
        "do_rescale": true,
        "do_resize": true,
        "feature_extractor_type": "CLIPFeatureExtractor",
        "image_mean": [
            0.48145466,
            0.4578275,
            0.40821073
        ],
        "image_processor_type": "CLIPImageProcessor",
        "image_std": [
            0.26862954,
            0.26130258,
            0.27577711
        ],
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "size": {
            "shortest_edge": 336
        }
        }
        """
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        print("self.vision_tower\n", self.vision_tower)
        """
        CLIPVisionModel(
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
        """
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True
        print("self.is_loaded\n", self.is_loaded) # True

    def feature_select(self, image_forward_outs):

        print("current file path", "llava/llava/model/multimodal_encoder/clip_encoder.py")
        print("def CLIPVisionTower.feature_select(self, image_forward_outs)")
        print("image_forward_outs\n", image_forward_outs) # 24層のtuple
        image_features = image_forward_outs.hidden_states[self.select_layer]
        print("image_features (after select_layer)\n", type(image_features))
        if hasattr(image_features, 'shape'):
            print("image_features.shape\n", image_features.shape) # torch.Size([1, 577, 1024])
        print(f"【COND】 select_feature={self.select_feature}") # patch
        if self.select_feature == 'patch':
            print("【ENTER】if self.select_feature == 'patch':")
            print("original image_features\n", image_features)
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
            print("after process\n", image_features)
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
            print("【EXIT】if self.select_feature == 'patch':")
        elif self.select_feature == 'cls_patch':
            # 【SKIP】
            print("【ENTER】elif self.select_feature == 'cls_patch':")
            image_features = image_features
            print("【EXIT】elif self.select_feature == 'cls_patch':")
        else:
            # 【SKIP】
            print(f"【COND】 select_feature={self.select_feature}")
            print("【ENTER】else (unexpected select_feature):")
            print("print(risk): print(self.select_feature) disabled for safety")
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
            print("【EXIT】else (unexpected select_feature):")
        print("image_features (return)\n", image_features)
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
            print("image_features.shape\n", image_features.shape) # torch.Size([1, 576, 1024])
        return image_features

    # requires_grad_(False) は パラメータの勾配計算を止める。
    # @torch.no_grad() は 全てのテンソルの勾配記録を止める（forward実行時のみ）。
    @torch.no_grad() 
    def forward(self, images):

        print("current file path", "llava/llava/model/multimodal_encoder/clip_encoder.py")
        print("def CLIPVisionTower.forward(self, images)")
        print("images\n", images)
        """
        images
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
        if hasattr(images, 'shape'):
            print("images.shape\n", images.shape) # torch.Size([1, 3, 336, 336])
        print(f"【COND】 type_images_is_list={type(images) is list}") # False
        if type(images) is list:
            # 【SKIP】
            print("【ENTER】if type(images) is list:")
            image_features = []
            print("original images\n", images)
            for image in images:
                print("original image\n", image)
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                print("after process image_forward_out\n", type(image_forward_out))
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                print("after process image_feature\n", type(image_feature))
                image_features.append(image_feature)
            print("after process image_features (list)\n", type(image_features))
            print("【EXIT】if type(images) is list:")
        else:
            # 【ENTER】
            print("【ENTER】else (type(images) is not list):")
            print("original images\n", images)
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            print("after process image_forward_outs\n", type(image_forward_outs)) # 24層のtuple
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            print("after process image_features\n", type(image_features)) # <class 'torch.Tensor'>
            print("【EXIT】else (type(images) is not list):")

        print("image_features (return)\n", image_features)
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
            print("image_features.shape\n", image_features.shape) # 
        return image_features

    @property
    def dummy_feature(self):

        print("current file path", "llava/llava/model/multimodal_encoder/clip_encoder.py")
        print("def CLIPVisionTower.dummy_feature(self)")
        print("self\n", type(self))
        result = torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
        print("result (return)\n", result)
        if hasattr(result, 'shape'):
            print("result.shape\n", result.shape)
        return result

    @property
    def dtype(self):

        print("current file path", "llava/llava/model/multimodal_encoder/clip_encoder.py")
        print("def CLIPVisionTower.dtype(self)")
        print("self\n", type(self))
        result = self.vision_tower.dtype
        print("result (return)\n", result)
        return result

    @property
    def device(self):

        print("current file path", "llava/llava/model/multimodal_encoder/clip_encoder.py")
        print("def CLIPVisionTower.device(self)")
        print("self\n", type(self))
        result = self.vision_tower.device
        print("result (return)\n", result)
        return result

    @property
    def config(self):

        print("current file path", "llava/llava/model/multimodal_encoder/clip_encoder.py")
        print("def CLIPVisionTower.config(self)")
        print("self\n", type(self)) 
        print("self.is_loaded\n", self.is_loaded) # True
        print(f"【COND】 is_loaded={self.is_loaded}")
        if self.is_loaded:
            # 【ENTER】
            print("【ENTER】if self.is_loaded:")
            result = self.vision_tower.config
            print("result (return)\n", type(result))
            print("【EXIT】if self.is_loaded:")
        else:
            # 【SKIP】
            print("【ENTER】else (not is_loaded):")
            result = self.cfg_only
            print("result (return)\n", type(result))
            print("【EXIT】else (not is_loaded):")
        print("result (return)\n", result)
        """
        CLIPVisionConfig {
        "_name_or_path": "openai/clip-vit-large-patch14-336",
        "attention_dropout": 0.0,
        "dropout": 0.0,
        "hidden_act": "quick_gelu",
        "hidden_size": 1024,
        "image_size": 336,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "layer_norm_eps": 1e-05,
        "model_type": "clip_vision_model",
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": 24,
        "patch_size": 14,
        "projection_dim": 768,
        "transformers_version": "4.31.0"
        }
        """
        return result

    @property
    def hidden_size(self):

        print("current file path", "llava/llava/model/multimodal_encoder/clip_encoder.py")
        print("def CLIPVisionTower.hidden_size(self)")
        print("self\n", type(self))
        result = self.config.hidden_size
        print("result (return), self.config.hidden_size\n", result) # 1024
        return result

    @property
    def num_patches_per_side(self):

        print("current file path", "llava/llava/model/multimodal_encoder/clip_encoder.py")
        print("def CLIPVisionTower.num_patches_per_side(self)")
        print("self\n", type(self))
        result = self.config.image_size // self.config.patch_size
        print("result (return)\n", result)
        return result

    @property
    def num_patches(self):

        print("current file path", "llava/llava/model/multimodal_encoder/clip_encoder.py")
        print("def CLIPVisionTower.num_patches(self)")
        print("self\n", type(self))
        result = (self.config.image_size // self.config.patch_size) ** 2
        print("result (return)\n", result)
        return result
