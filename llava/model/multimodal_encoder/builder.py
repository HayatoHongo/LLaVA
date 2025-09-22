import os
from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):

    print("current file path", "llava/llava/model/multimodal_encoder/builder.py")
    print("def build_vision_tower(vision_tower_cfg, **kwargs)")
    print("vision_tower_cfg\n", vision_tower_cfg) # ModelArguments(model_name_or_path='lmsys/vicuna-7b-v1.5', version='plain', freeze_backbone=False, tune_mm_mlp_adapter=True, vision_tower='openai/clip-vit-large-patch14-336', mm_vision_select_layer=-2, pretrain_mm_mlp_adapter=None, mm_projector_type='mlp2x_gelu', mm_use_im_start_end=False, mm_use_im_patch_token=False, mm_patch_merge_type='flat', mm_vision_select_feature='patch')
    print("kwargs\n", kwargs) # {}
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    print("vision_tower from vision_tower_cfg\n", vision_tower) # openai/clip-vit-large-patch14-336
    # ローカルに存在しない場合はFalse。存在する場合の例: /ubuntu/home/user/model/openai/clip-vit-large-patch14-336
    is_absolute_path_exists = os.path.exists(vision_tower)
    print("is_absolute_path_exists\n", is_absolute_path_exists) # False
    print(f"【COND】 is_absolute_path_exists={is_absolute_path_exists} vision_tower={vision_tower}") # is_absolute_path_exists=False vision_tower=openai/clip-vit-large-patch14-336
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        # 【ENTER】
        print("【ENTER】if is_absolute_path_exists or vision_tower.startswith('openai') or vision_tower.startswith('laion') or 'ShareGPT4V' in vision_tower:")
        result = CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
        print("result (return)\n", result)
        """
        result (return)
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
        print("【EXIT】if is_absolute_path_exists or vision_tower.startswith('openai') or vision_tower.startswith('laion') or 'ShareGPT4V' in vision_tower:")
        return result

    print("print(risk): print(vision_tower) disabled for safety")
    raise ValueError(f'Unknown vision tower: {vision_tower}')
