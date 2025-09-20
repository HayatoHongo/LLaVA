import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):

        print("current file path", "llava/llava/model/multimodal_projector/builder.py")
        print("def IdentityMap.__init__(self)")
        print("self\n", type(self))
        super().__init__()

    def forward(self, x, *args, **kwargs):

        print("current file path", "llava/llava/model/multimodal_projector/builder.py")
        print("def IdentityMap.forward(self, x, *args, **kwargs)")
        print("【COND】x\n", x)
        if hasattr(x, 'shape'):
            print("x.shape\n", x.shape)
        print("args\n", args)
        print("kwargs\n", kwargs)
        result = x
        print("result (return)\n", result)
        if hasattr(result, 'shape'):
            print("result.shape\n", result.shape)
        return result

    @property
    def config(self):

        print("current file path", "llava/llava/model/multimodal_projector/builder.py")
        print("def IdentityMap.config(self)")
        print("self\n", type(self))
        result = {"mm_projector_type": 'identity'}
        print("result (return)\n", result)
        return result


class SimpleResBlock(nn.Module):
    def __init__(self, channels):

        print("current file path", "llava/llava/model/multimodal_projector/builder.py")
        print("def SimpleResBlock.__init__(self, channels)")
        print("self\n", type(self))
        print("channels\n", channels)
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):

        print("current file path", "llava/llava/model/multimodal_projector/builder.py")
        print("def SimpleResBlock.forward(self, x)")
        print("x\n", x)
        if hasattr(x, 'shape'):
            print("x.shape\n", x.shape)
        x = self.pre_norm(x)
        result = x + self.proj(x)
        print("result (return)\n", result)
        if hasattr(result, 'shape'):
            print("result.shape\n", result.shape)
        return result


def build_vision_projector(config, delay_load=False, **kwargs):
    # self.mm_projector = build_vision_projector(self.config)
    print("current file path", "llava/llava/model/multimodal_projector/builder.py")
    print("def build_vision_projector(config, delay_load=False, **kwargs)")
    print("config\n", config)
    """
    config
    LlavaConfig {
    "_name_or_path": "lmsys/vicuna-7b-v1.5",
    "architectures": [
        "LlamaForCausalLM"
    ],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 4096,
    "mm_hidden_size": 1024,
    "mm_patch_merge_type": "flat",
    "mm_projector_type": "mlp2x_gelu",
    "mm_vision_select_feature": "patch",
    "mm_vision_select_layer": -2,
    "mm_vision_tower": "openai/clip-vit-large-patch14-336",
    "model_type": "llava_llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 32,
    "pad_token_id": 0,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "tie_word_embeddings": false,
    "torch_dtype": "float16",
    "transformers_version": "4.31.0",
    "use_cache": false,
    "use_mm_proj": true,
    "vocab_size": 32000
    }
    """
    print("delay_load\n", delay_load) # False
    print("kwargs\n", kwargs) # {}
    projector_type = getattr(config, 'mm_projector_type', 'linear') 
    print("projector_type from config\n", projector_type) # mlp2x_gelu

    print("【COND】 projector_type\n", projector_type) # mlp2x_gelu
    if projector_type == 'linear':
        #【SKIP】
        result = nn.Linear(config.mm_hidden_size, config.hidden_size)
        print("result (return)\n", result)
        return result
        
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    print("【COND】mlp_gelu_match\n", mlp_gelu_match)
    if mlp_gelu_match:
        #【ENTER】if mlp_gelu_match:
        print("【ENTER】if mlp_gelu_match:")
        mlp_depth = int(mlp_gelu_match.group(1))
        print("mlp_depth from mlp_gelu_match.group(1)\n", mlp_depth)
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        print("modules after first Linear\n", modules)
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        print("modules before Sequential\n", modules)
        result = nn.Sequential(*modules) # * はリストをアンパックして引数に展開する
        print("result (return)\n", result)
        """
        Sequential(
        (0): Linear(in_features=1024, out_features=4096, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=4096, out_features=4096, bias=True)
        )
        """
        print("【EXIT】if mlp_gelu_match:")
        return result

    print("【COND】projector_type\n", projector_type)
    if projector_type == 'identity':
        # 【SKIP】
        print("【ENTER】if projector_type == 'identity':")
        result = IdentityMap()
        print("result (return)\n", result)
        print("【EXIT】if projector_type == 'identity':")
        return result

    print("print(risk): print(projector_type) disabled for safety")
    raise ValueError(f'Unknown projector type: {projector_type}')
