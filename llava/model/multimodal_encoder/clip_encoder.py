import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):

        print("current file path", "llava/llava/model/multimodal_encoder/clip_encoder.py")
        print("def CLIPVisionTower.__init__(self, vision_tower, args, delay_load=False)")
        print("self\n", type(self))
        print("vision_tower\n", vision_tower)
        print("args\n", args)
        print("delay_load\n", delay_load)
        super().__init__()

        self.is_loaded = False
        print("self.is_loaded\n", self.is_loaded)

        self.vision_tower_name = vision_tower
        print("self.vision_tower_name\n", self.vision_tower_name)
        self.select_layer = args.mm_vision_select_layer
        print("self.select_layer\n", self.select_layer)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        print("self.select_feature\n", self.select_feature)
        
        print(f"[COND] delay_load={delay_load}")
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
        print("self.vision_tower_name\n", self.vision_tower_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        print("self.image_processor\n", self.image_processor)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        print("self.vision_tower\n", self.vision_tower)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):

        print("current file path", "llava/llava/model/multimodal_encoder/clip_encoder.py")
        print("def CLIPVisionTower.feature_select(self, image_forward_outs)")
        print("image_forward_outs\n", image_forward_outs)
        image_features = image_forward_outs.hidden_states[self.select_layer]
        print("image_features (after select_layer)\n", type(image_features))
        if hasattr(image_features, 'shape'):
            print("image_features.shape\n", image_features.shape)
        print(f"[COND] select_feature={self.select_feature}")
        if self.select_feature == 'patch':
            print("【ENTER】if self.select_feature == 'patch':")
            print("original image_features\n", image_features)
            image_features = image_features[:, 1:]
            print("after process\n", image_features)
            print("【EXIT】if self.select_feature == 'patch':")
        elif self.select_feature == 'cls_patch':
            print("【ENTER】elif self.select_feature == 'cls_patch':")
            image_features = image_features
            print("【EXIT】elif self.select_feature == 'cls_patch':")
        else:
            print(f"[COND] select_feature={self.select_feature}")
            print("【ENTER】else (unexpected select_feature):")
            print("print(risk): print(self.select_feature) disabled for safety")
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
            print("【EXIT】else (unexpected select_feature):")
        print("image_features (return)\n", image_features)
        if hasattr(image_features, 'shape'):
            print("image_features.shape\n", image_features.shape)
        return image_features

    @torch.no_grad()
    def forward(self, images):

        print("current file path", "llava/llava/model/multimodal_encoder/clip_encoder.py")
        print("def CLIPVisionTower.forward(self, images)")
        print("images\n", images)
        if hasattr(images, 'shape'):
            print("images.shape\n", images.shape)
        print(f"[COND] type_images_is_list={type(images) is list}")
        if type(images) is list:
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
            print("【ENTER】else (type(images) is not list):")
            print("original images\n", images)
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            print("after process image_forward_outs\n", type(image_forward_outs))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            print("after process image_features\n", type(image_features))
            print("【EXIT】else (type(images) is not list):")

        print("image_features (return)\n", image_features)
        if hasattr(image_features, 'shape'):
            print("image_features.shape\n", image_features.shape)
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
        # おそらく is_loaded は True
        print("self.is_loaded\n", self.is_loaded)
        print(f"[COND] is_loaded={self.is_loaded}")
        if self.is_loaded:
            print("【ENTER】if self.is_loaded:")
            result = self.vision_tower.config
            print("result (return)\n", type(result))
            print("【EXIT】if self.is_loaded:")
        else:
            print("【ENTER】else (not is_loaded):")
            result = self.cfg_only
            print("result (return)\n", type(result))
            print("【EXIT】else (not is_loaded):")
        print("result (return)\n", result)
        return result

    @property
    def hidden_size(self):

        print("current file path", "llava/llava/model/multimodal_encoder/clip_encoder.py")
        print("def CLIPVisionTower.hidden_size(self)")
        print("self\n", type(self))
        result = self.config.hidden_size
        print("result (return), self.config.hidden_size\n", result)
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
