import os
from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):

    print("current file path", "llava/llava/model/multimodal_encoder/builder.py")
    print("def build_vision_tower(vision_tower_cfg, **kwargs)")
    print("vision_tower_cfg\n", vision_tower_cfg)
    print("kwargs\n", kwargs)
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    print(f"[COND] is_absolute_path_exists={is_absolute_path_exists} vision_tower={vision_tower}")
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        print("【ENTER】if is_absolute_path_exists or vision_tower.startswith('openai') or vision_tower.startswith('laion') or 'ShareGPT4V' in vision_tower:")
        result = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        print("result (return)\n", result)
        print("【EXIT】if is_absolute_path_exists or vision_tower.startswith('openai') or vision_tower.startswith('laion') or 'ShareGPT4V' in vision_tower:")
        return result

    print("print(risk): print(vision_tower) disabled for safety")
    raise ValueError(f'Unknown vision tower: {vision_tower}')
