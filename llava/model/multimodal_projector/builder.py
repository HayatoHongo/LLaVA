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
        print("x\n", x)
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

    print("current file path", "llava/llava/model/multimodal_projector/builder.py")
    print("def build_vision_projector(config, delay_load=False, **kwargs)")
    print("config\n", config)
    print("delay_load\n", delay_load)
    print("kwargs\n", kwargs)
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    print("projector_type\n", projector_type)
    if projector_type == 'linear':
        result = nn.Linear(config.mm_hidden_size, config.hidden_size)
        print("result (return)\n", result)
        return result
        
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    print("mlp_gelu_match\n", mlp_gelu_match)
    if mlp_gelu_match:
        print("【ENTER】if mlp_gelu_match:")
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        result = nn.Sequential(*modules)
        print("result (return)\n", result)
        print("【EXIT】if mlp_gelu_match:")
        return result

    print("projector_type\n", projector_type)
    if projector_type == 'identity':
        print("【ENTER】if projector_type == 'identity':")
        result = IdentityMap()
        print("result (return)\n", result)
        print("【EXIT】if projector_type == 'identity':")
        return result

    print("print(risk): print(projector_type) disabled for safety")
    raise ValueError(f'Unknown projector type: {projector_type}')
