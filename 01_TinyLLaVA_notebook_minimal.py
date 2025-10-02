# -*- coding: utf-8 -*-
# Refactored: minimal, readable, responsibility-focused

import os
import re
import json
import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import transformers
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments as HFTrainingArguments,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
    CLIPVisionModel,
    CLIPImageProcessor,
    CLIPVisionConfig,
)
from dataclasses import dataclass, field
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from PIL import Image

# -----------------------------
# Args (kept but trimmed)
# -----------------------------

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="plain")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)     # last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    # 固定でpatchを使う前提に簡素化するが、互換のため残す
    mm_vision_select_feature: Optional[str] = field(default="patch")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    # set at runtime
    image_processor: Optional[CLIPImageProcessor] = None

@dataclass
class TrainingArguments(HFTrainingArguments):
    # 余計な拡張は削除し、必要最小限のみ保持
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=512)
    # LoRAや量子化のフィールドは使っていないため削除
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    # Adapter関連の最小限
    freeze_mm_mlp_adapter: bool = False
    use_im_start_end: bool = False


# -----------------------------
# Constants (kept)
# -----------------------------
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"
SEP = "\n"  # 会話テンプレートは1定数に縮約


# -----------------------------
# Vision Tower (fixed bug, simplified)
# -----------------------------
class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower_name: str, select_layer: int, delay_load: bool = False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower_name
        self.select_layer = select_layer

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def _feature_select(self, hidden_states):
        # patch固定: CLS(先頭)除外で[:, 1:]
        return hidden_states[:, 1:]

    @torch.no_grad()
    def forward(self, images: torch.FloatTensor) -> torch.FloatTensor:
        outs = self.vision_tower(
            images.to(device=self.vision_tower.device, dtype=self.vision_tower.dtype),
            output_hidden_states=True,
        )
        feats = outs.hidden_states[self.select_layer]
        feats = self._feature_select(feats).to(images.dtype)  # (B, 576, C)
        return feats

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        # 未ロード時でも設定を返す
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size


def build_vision_tower(vision_tower_name: str, select_layer: int, delay_load: bool = False) -> CLIPVisionTower:
    if not vision_tower_name:
        raise ValueError("vision_tower_name is required")
    # 1種類の実装のみ想定のためシンプルに
    return CLIPVisionTower(vision_tower_name, select_layer=select_layer, delay_load=delay_load)


def build_vision_projector(config) -> nn.Module:
    """mlpNx_gelu or linear のみサポート（実運用で使うものに限定）"""
    proj_type = getattr(config, "mm_projector_type", "linear")
    if proj_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    m = re.match(r"^mlp(\d+)x_gelu$", proj_type)
    if not m:
        raise ValueError(f"Unsupported projector type: {proj_type}")
    depth = int(m.group(1))
    layers: List[nn.Module] = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
    for _ in range(1, depth):
        layers += [nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size)]
    return nn.Sequential(*layers)


# -----------------------------
# Multimodal base model (merged)
# -----------------------------
class LlavaLlamaModel(LlamaModel):
    """LlamaModel + (vision tower, projector) を保持する最小構成"""

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # 必要になったときに initialize_vision_modules で埋める
        self.vision_tower: Optional[CLIPVisionTower] = None
        self.mm_projector: Optional[nn.Module] = None

    def initialize_vision_modules(self, model_args: ModelArguments):
        # Vision tower
        self.config.mm_vision_tower = model_args.vision_tower
        vt = build_vision_tower(
            vision_tower_name=model_args.vision_tower,
            select_layer=model_args.mm_vision_select_layer,
            delay_load=False,
        )
        self.vision_tower = vt

        # Config side-effects
        self.config.use_mm_proj = True
        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.mm_hidden_size = vt.hidden_size
        self.config.mm_vision_select_layer = model_args.mm_vision_select_layer
        self.config.mm_patch_merge_type = model_args.mm_patch_merge_type
        self.config.mm_use_im_start_end = model_args.mm_use_im_start_end
        self.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

        # Projector
        self.mm_projector = build_vision_projector(self.config)


# -----------------------------
# CausalLM (merged helpers)
# -----------------------------
class LlavaLlamaForCausalLM(LlamaForCausalLM):
    """LlamaForCausalLM + 画像エンコード/埋め込み挿入処理（プライベートメソッド化）"""

    def __init__(self, config: LlamaConfig):
        super().__init__(config)   # ← 可読性重視で素直に
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    # --- small helpers ---
    def get_model(self) -> LlavaLlamaModel:
        return self.model

    def _encode_images(self, images: torch.FloatTensor) -> torch.FloatTensor:
        """VisionTower→Projectorで画像特徴を取得"""
        vt = self.get_model().vision_tower
        assert vt is not None, "Vision tower is not initialized"
        feats = vt(images)  # (B, P, C_v)
        proj = self.get_model().mm_projector
        assert proj is not None, "Projector is not initialized"
        return proj(feats)  # (B, P, C_text)

    def _prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
    ):
        """
        前提:
          - images/attention_mask は None でない（画像あり経路）
        概要:
          1) マスクで右パディング除去
          2) <image> トークン位置に画像埋め込みを差し込む
          3) 右パディングで復元（長さ合わせ）
        """
        assert images is not None, "images must be provided for multimodal path"
        assert attention_mask is not None, "attention_mask must be provided"

        image_features = self._encode_images(images)  # (B_img, P, C)
        attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        # 1) remove padding
        input_ids_list = [ids[mask] for ids, mask in zip(input_ids, attention_mask)]
        labels_list = [lab[mask] for lab, mask in zip(labels, attention_mask)]

        new_embeds_list: List[torch.Tensor] = []
        new_labels_list: List[torch.Tensor] = []
        cur_img_idx = 0

        for b, ids in enumerate(input_ids_list):
            num_images = (ids == IMAGE_TOKEN_INDEX).sum()
            # トークンを埋め込みへ
            tok_embeds = self.get_model().embed_tokens(ids)

            if num_images == 0:
                # 画像なしサンプル（想定外だが安全側処理）
                new_embeds_list.append(tok_embeds)
                new_labels_list.append(labels_list[b])
                continue

            # <image> 位置で分割し、間に画像埋め込みを挟む
            split_points = [-1] + torch.where(ids == IMAGE_TOKEN_INDEX)[0].tolist() + [ids.shape[0]]
            labels_b = labels_list[b]
            text_chunks = []
            label_chunks = []
            for i in range(len(split_points) - 1):
                sl, sr = split_points[i] + 1, split_points[i + 1]
                text_chunks.append(ids[sl:sr])
                label_chunks.append(labels_b[sl:sr])

            # 連結前に埋め込み化
            split_sizes = [len(x) for x in text_chunks]
            text_cat = torch.cat(text_chunks) if split_sizes and sum(split_sizes) > 0 else ids.new_empty((0,))
            text_embeds = self.get_model().embed_tokens(text_cat) if text_cat.numel() > 0 else self.get_model().embed_tokens(ids[0:0])
            text_embeds_split = torch.split(text_embeds, split_sizes, dim=0) if split_sizes else []

            assembled_embeds: List[torch.Tensor] = []
            assembled_labels: List[torch.Tensor] = []
            for i in range(num_images + 1):
                # テキスト部分
                if text_embeds_split:
                    assembled_embeds.append(text_embeds_split[i])
                    assembled_labels.append(label_chunks[i])
                # 画像部分（末尾以外に挿入）
                if i < num_images:
                    img_feat = image_features[cur_img_idx]  # (P, C)
                    cur_img_idx += 1
                    assembled_embeds.append(img_feat)
                    assembled_labels.append(
                        torch.full((img_feat.shape[0],), IGNORE_INDEX, device=labels_b.device, dtype=labels_b.dtype)
                    )

            new_embeds = torch.cat(assembled_embeds) if assembled_embeds else text_embeds
            new_labels = torch.cat(assembled_labels) if assembled_labels else labels_b
            new_embeds_list.append(new_embeds.to(self.device))
            new_labels_list.append(new_labels)

        # 2) 長さ制限（オプショナル）
        max_len_cfg = getattr(self.config, "tokenizer_model_max_length", None)
        if max_len_cfg is not None:
            new_embeds_list = [x[:max_len_cfg] for x in new_embeds_list]
            new_labels_list = [x[:max_len_cfg] for x in new_labels_list]

        # 3) 右パディング復元
        max_len = max(x.shape[0] for x in new_embeds_list)
        bsz = len(new_embeds_list)
        padded_embeds = []
        padded_labels = torch.full(
            (bsz, max_len), IGNORE_INDEX, dtype=new_labels_list[0].dtype, device=new_labels_list[0].device
        )
        new_attn = torch.zeros((bsz, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        new_pos = torch.zeros((bsz, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (emb, lab) in enumerate(zip(new_embeds_list, new_labels_list)):
            L = emb.shape[0]
            pad = torch.zeros((max_len - L, emb.shape[1]), dtype=emb.dtype, device=emb.device)
            padded_embeds.append(torch.cat([emb, pad], dim=0))
            if L > 0:
                padded_labels[i, :L] = lab
                new_attn[i, :L] = True
                new_pos[i, :L] = torch.arange(0, L, dtype=new_pos.dtype, device=new_pos.device)

        inputs_embeds = torch.stack(padded_embeds, dim=0)
        if position_ids is None:
            new_pos = None

        return None, new_pos, new_attn, past_key_values, inputs_embeds, padded_labels

    # --- forward ---
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
    ) -> Union[Tuple, CausalLMOutputWithPast, GenerateOutput]:
        if inputs_embeds is None and images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self._prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )

        return LlamaForCausalLM.forward(
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
            return_dict=return_dict,
        )


# -----------------------------
# Data utils (plain only)
# -----------------------------
def tokenizer_image_token(prompt: str, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def interleave(X, sep):
        return [e for pair in zip(X, [sep] * len(X)) for e in pair][:-1]

    input_ids = []
    offset = 0
    if chunks and chunks[0] and chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(chunks[0][0])

    for x in interleave(chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    return input_ids


def preprocess_plain(sources: Sequence[str], tokenizer) -> Dict:
    """plain固定: <image> + テキストを結合し、<image>位置に特殊IDを差し込む"""
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        # 画像トークンは単独に
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversations.append(source[0]["value"] + source[1]["value"] + SEP)

    input_ids = [tokenizer_image_token(p, tokenizer, return_tensors="pt") for p in conversations]
    targets = copy.deepcopy(input_ids)
    for tgt, src in zip(targets, sources):
        prompt_len = len(tokenizer_image_token(src[0]["value"], tokenizer))  # "<image>" 部分
        tgt[:prompt_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=targets)


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    """必要最小限の置換のみ"""
    for source in sources:
        for sent in source:
            if DEFAULT_IMAGE_TOKEN in sent["value"]:
                sent["value"] = sent["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sent["value"] = (DEFAULT_IMAGE_TOKEN + "\n" + sent["value"]).strip()
            if data_args.mm_use_im_start_end:
                sent["value"] = sent["value"].replace(
                    DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
    return sources


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning (必要最小限)"""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super().__init__()
        self.tokenizer = tokenizer
        self.list_data_dict = json.load(open(data_path, "r"))
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Optional[Dict[str, torch.Tensor]]:
        ex = self.list_data_dict[i]
        sources = [ex] if isinstance(i, int) else ex
        assert len(sources) == 1, "unexpected wrap"

        image = None
        if "image" in sources[0]:
            image_file = sources[0]["image"]
            image_path = os.path.join(self.data_args.image_folder, image_file)
            try:
                img = Image.open(image_path).convert("RGB")
            except Exception:
                return None  # 画像欠損時はスキップ
            image = self.data_args.image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        data = preprocess(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
        if isinstance(i, int):
            data = dict(input_ids=data["input_ids"][0], labels=data["labels"][0])

        if image is not None:
            data["image"] = image
        elif self.data_args.is_multimodal:
            crop = self.data_args.image_processor.crop_size
            data["image"] = torch.zeros(3, crop["height"], crop["width"])
        return data


class DataCollatorForSupervisedDataset:
    """関数型に近い、シンプルなコラレータ"""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Optional[Dict]]) -> Dict[str, torch.Tensor]:
        instances = [x for x in instances if x is not None]
        if not instances:
            # dataloader側で再試行される想定
            return dict(
                input_ids=torch.empty(0, dtype=torch.long),
                labels=torch.empty(0, dtype=torch.long),
                attention_mask=torch.empty(0, dtype=torch.bool),
            )

        input_ids, labels = tuple([inst[k] for inst in instances] for k in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [inst["image"] for inst in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch["images"] = torch.stack(images)
            else:
                batch["images"] = images
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments) -> Dict:
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


# -----------------------------
# Saving (final-only; no DeepSpeed branches)
# -----------------------------
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """mm_projectorのみ学習する前提では、最終的にそれだけ保存"""
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        keys = ["mm_projector"]
        named = dict(trainer.model.named_parameters())
        weight_to_save = {k: v.detach().cpu().clone() for k, v in named.items() if any(k.startswith(key) or (("." + key + ".") in k) for key in keys)}
        trainer.model.config.save_pretrained(output_dir)
        torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
        return

    # 通常保存（CPUへ移してから）
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state = {k: v.cpu() for k, v in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state)  # noqa


# -----------------------------
# Glue
# -----------------------------
def preprocess(sources: Sequence[str], tokenizer, has_image: bool = False) -> Dict:
    # plain固定なので分岐不要
    return preprocess_plain(sources, tokenizer)


def train():
    args_dict = {
        "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "version": "plain",
        "data_path": "/home/ubuntu/llava-virginia/LLaVA-CC3M-Pretrain-595K/chat.json",
        "image_folder": "/home/ubuntu/llava-virginia/LLaVA-CC3M-Pretrain-595K/images",
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
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "evaluation_strategy": "no",
        "save_strategy": "steps",
        "save_steps": 500,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "logging_steps": 1,
        "tf32": False,
        "model_max_length": 2048,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 2,
        "lazy_preprocess": True,
        "report_to": "none",
    }

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(args_dict)

    compute_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)

    # モデル本体（事前学習重み読み込み）
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    # 勾配チェックポイント有効化時の標準パターン
    if training_args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # plain固定のためpad_tokenはunkで代用
    tokenizer.pad_token = tokenizer.unk_token

    # Visionの初期化
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)
        vt = model.get_model().vision_tower
        vt.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vt.image_processor
        data_args.is_multimodal = True

        # Configへ必要情報
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        # Adapter学習（mm_projectorのみ学習）
        training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
