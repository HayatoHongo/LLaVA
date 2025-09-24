import os
import torch

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    logger,
)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):

    print("current file path", "llava/train/llava_trainer.py")
    print("def maybe_zero_3(param, ignore_status=False, name=None)")
    print("param\n", param)
    print("ignore_status\n", ignore_status)
    print("name\n", name)
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    print(f"【COND】 hasattr_ds_id={hasattr(param, 'ds_id')}")
    if hasattr(param, "ds_id"):
        print("【ENTER】if hasattr(param, 'ds_id'):")
        print(f"【COND】 ds_status={getattr(param, 'ds_status', None)}, ignore_status={ignore_status}")
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            print("【ENTER】if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:")
            print(f"【COND】 ignore_status={ignore_status}")
            if not ignore_status:
                print("【ENTER】if not ignore_status:")
                print(name, 'no ignore status')
                print("【EXIT】if not ignore_status:")
            print("【EXIT】if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
        print("【EXIT】if hasattr(param, 'ds_id'):")
    else:
        print("【ENTER】else (not hasattr(param, 'ds_id')):")
        param = param.detach().cpu().clone()
        print("【EXIT】else (not hasattr(param, 'ds_id')):")
    print("param (to return)\n", param)
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):

    print("current file path", "llava/train/llava_trainer.py")
    print("def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match)")
    print("named_params\n", named_params)
    print("keys_to_match\n", keys_to_match)
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    print("to_return\n", to_return)
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):

    print("current file path", "llava/train/llava_trainer.py")
    print("def split_to_even_chunks(indices, lengths, num_chunks)")
    print("indices\n", indices)
    print("lengths\n", lengths)
    print("num_chunks\n", num_chunks)
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """


    print(f"【COND】 len_indices={len(indices)}, num_chunks={num_chunks}")
    if len(indices) % num_chunks != 0:
        print("【ENTER】if len(indices) % num_chunks != 0:")
        result = [indices[i::num_chunks] for i in range(num_chunks)]
        print("【EXIT】if len(indices) % num_chunks != 0:")
        return result

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    print("chunks\n", chunks)
    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):

    print("current file path", "llava/train/llava_trainer.py")
    print("def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None)")
    print("lengths\n", lengths)
    print("batch_size\n", batch_size)
    print("world_size\n", world_size)
    print("generator\n", generator)
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    print(f"【COND】 all_gt0={all(l > 0 for l in lengths)}, all_lt0={all(l < 0 for l in lengths)}")
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        print("【ENTER】if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):")
        # all samples are in the same modality
        result = get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
        print("【EXIT】if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):")
        return result
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]
    # print of return value is not needed as it's a direct return


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):

    print("current file path", "llava/train/llava_trainer.py")
    print("def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True)")
    print("lengths\n", lengths)
    print("batch_size\n", batch_size)
    print("world_size\n", world_size)
    print("generator\n", generator)
    print("merge\n", merge)
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    print(f"【COND】 merge={merge}")
    if merge:
        print("【ENTER】if merge:")
        result = [i for megabatch in megabatches for batch in megabatch for i in batch]
        print("【EXIT】if merge:")
    else:
        print("【ENTER】else (not merge):")
        result = megabatches
        print("【EXIT】else (not merge):")
    print("result\n", result)
    return result


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        print("current file path", "llava/train/llava_trainer.py")
        print("def __init__(self, batch_size: int, world_size: int, lengths: Optional[List[int]] = None, generator=None, group_by_modality: bool = False)")
        print("self\n", self)
        print("batch_size\n", batch_size)
        print("world_size\n", world_size)
        print("lengths\n", lengths)
        print("generator\n", generator)
        print("group_by_modality\n", group_by_modality)
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        print("current file path", "llava/train/llava_trainer.py")
        print("def __len__(self)")
        print("self\n", self)
        return len(self.lengths)

    def __iter__(self):
        print("current file path", "llava/train/llava_trainer.py")
        print("def __iter__(self)")
        print("self\n", self)
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:

        print("current file path", "llava/train/llava_trainer.py")
        print("def _get_train_sampler(self)")
        print("self\n", self) 
        print(f"【COND】 train_dataset_is_None={self.train_dataset is None}, has_length={has_length(self.train_dataset) if self.train_dataset is not None else 'N/A'}") # train_dataset_is_None=False, has_length=True
        if self.train_dataset is None or not has_length(self.train_dataset):
            #【SKIP】
            print("【ENTER】if self.train_dataset is None or not has_length(self.train_dataset):")
            result = None
            print("【EXIT】if self.train_dataset is None or not has_length(self.train_dataset):")
            return result

        print(f"【COND】 group_by_modality_length={self.args.group_by_modality_length}") # group_by_modality_length=False
        if self.args.group_by_modality_length:
            # 【SKIP】
            print("【ENTER】if self.args.group_by_modality_length:")
            lengths = self.train_dataset.modality_lengths
            result = LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
            print("【EXIT】if self.args.group_by_modality_length:")
            return result
        else:
            # 【ENTER】
            print("【ENTER】else (not group_by_modality_length):")
            result = super()._get_train_sampler()
            print("result, super()._get_train_sampler()\n", result) # <torch.utils.data.sampler.RandomSampler object at 0x7ed63e925e70>
            print("【EXIT】else (not group_by_modality_length):")
            return result

    def create_optimizer(self):

        print("current file path", "llava/train/llava_trainer.py")
        print("def create_optimizer(self)")
        print("self\n", self)
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        print(f"【COND】 sagemaker_mp_enabled={is_sagemaker_mp_enabled()}") # False
        if is_sagemaker_mp_enabled():
            # 【SKIP】
            print("【ENTER】if is_sagemaker_mp_enabled():")
            result = super().create_optimizer()
            print("【EXIT】if is_sagemaker_mp_enabled():")
            print("result for super().create_optimizer()\n", result)
            return result
        print(f"【COND】 sharded_ddp={self.sharded_ddp}, SHARDED_DDP_SIMPLE={ShardedDDPOption.SIMPLE}") # sharded_ddp=None, SHARDED_DDP_SIMPLE=simple
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            # 【SKIP】
            print("【ENTER】if self.sharded_ddp == ShardedDDPOption.SIMPLE:")
            result = super().create_optimizer()
            print("【EXIT】if self.sharded_ddp == ShardedDDPOption.SIMPLE:")
            print("result for super().create_optimizer()\n", result)
            return result

        opt_model = self.model
        print("opt_model\n", opt_model)
        """
        LlavaLlamaForCausalLM(
        (model): LlavaLlamaModel(
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
        (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
        )
        """
        print(f"【COND】 optimizer_is_None={self.optimizer is None}") # 1回目: True, 2回目: False
        if self.optimizer is None:
            # 1回目【ENTER】, 2回目【SKIP】
            print("【ENTER】if self.optimizer is None:")

            # Risky print: self.args, opt_model, optimizer_grouped_parameters, optimizer_cls, optimizer_kwargs
            print("print(risk): print(self.args) disabled for safety")
            print("print(risk): print(opt_model) disabled for safety")
            print("print(risk): print(optimizer_grouped_parameters) disabled for safety")
            print("print(risk): print(optimizer_cls) disabled for safety")
            print("print(risk): print(optimizer_kwargs) disabled for safety")

            print("ALL_LAYERNORM_LAYERS\n", ALL_LAYERNORM_LAYERS)
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            print("decay_parameters (before removing bias)\n", decay_parameters)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            print("decay_parameters\n", decay_parameters)
            print(f"【COND】 mm_projector_lr={self.args.mm_projector_lr}") # None
            if self.args.mm_projector_lr is not None:
                #【SKIP】
                print("【ENTER】if self.args.mm_projector_lr is not None:")
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
                print("【EXIT】if self.args.mm_projector_lr is not None:")
            else:
                #【ENTER】else (mm_projector_lr is None):
                print("【ENTER】else (mm_projector_lr is None):")
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            print("optimizer_cls\n", optimizer_cls)
            print("optimizer_kwargs\n", optimizer_kwargs)

            print(f"【COND】 sharded_ddp={self.sharded_ddp}, SHARDED_DDP_SIMPLE={ShardedDDPOption.SIMPLE}") # sharded_ddp=None, SHARDED_DDP_SIMPLE=simple
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                # 【SKIP】
                print("【ENTER】if self.sharded_ddp == ShardedDDPOption.SIMPLE:")               
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
                print("【EXIT】if self.sharded_ddp == ShardedDDPOption.SIMPLE:")
            else:
                print("【ENTER】else (not sharded_ddp SIMPLE):")
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                print(f"【COND】 optimizer_cls_name={optimizer_cls.__name__}") # AdamW
                if optimizer_cls.__name__ == "Adam8bit":
                    # 【SKIP】
                    print("【ENTER】if optimizer_cls.__name__ == 'Adam8bit':")
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")
                    
                    print("【EXIT】if optimizer_cls.__name__ == 'Adam8bit':")
            print("【EXIT】if self.optimizer is None:")
        print("self.optimizer\n", self.optimizer)
        """
        AdamW (
        Parameter Group 0
            amsgrad: False
            betas: (0.9, 0.999)
            capturable: False
            differentiable: False
            eps: 1e-08
            foreach: None
            fused: None
            lr: 0.001
            maximize: False
            weight_decay: 0.0

        Parameter Group 1
            amsgrad: False
            betas: (0.9, 0.999)
            capturable: False
            differentiable: False
            eps: 1e-08
            foreach: None
            fused: None
            lr: 0.001
            maximize: False
            weight_decay: 0.0
        )
        """
        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):

        print("current file path", "llava/train/llava_trainer.py")
        print("def _save_checkpoint(self, model, trial, metrics=None)")
        print("self\n", self) # <llava.train.llava_trainer.LLaVATrainer object at 0x7ed6341f4490>
        print("model\n", model)
        """
        model
        DeepSpeedEngine(
        (module): LlavaLlamaForCausalLM(
            (model): LlavaLlamaModel(
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
            (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
        )
        )
        """
        print("trial\n", trial) # None
        print("metrics\n", metrics) # None
        print(f"【COND】 tune_mm_mlp_adapter={getattr(self.args, 'tune_mm_mlp_adapter', False)}") # True
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # 【ENTER】
            print("【ENTER】if getattr(self.args, 'tune_mm_mlp_adapter', False):")
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            print(f"【COND】 use_im_start_end={getattr(self.args, 'use_im_start_end', False)}") # False
            if getattr(self.args, "use_im_start_end", False):
                # 【SKIP】
                print("【ENTER】if getattr(self.args, 'use_im_start_end', False):")
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            print(f"【COND】 local_rank={self.args.local_rank}") # 0
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                # 【ENTER】
                print("【ENTER】if self.args.local_rank == 0 or self.args.local_rank == -1:")
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
                print("【EXIT】if self.args.local_rank == 0 or self.args.local_rank == -1:")
            print("【EXIT】if getattr(self.args, 'tune_mm_mlp_adapter', False):")
        else:
            # 【SKIP】
            print("【ENTER】else (not tune_mm_mlp_adapter):")
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)
            print("【EXIT】else (not tune_mm_mlp_adapter):")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):

        print("current file path", "llava/train/llava_trainer.py")
        print("def _save(self, output_dir: Optional[str] = None, state_dict=None)")
        print("self _save\n", self)
        print("output_dir _save\n", output_dir)
        print("state_dict _save\n", state_dict)
        print(f"【COND】 tune_mm_mlp_adapter={getattr(self.args, 'tune_mm_mlp_adapter', False)}")
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            print("【ENTER】if getattr(self.args, 'tune_mm_mlp_adapter', False):")
            pass
            print("【EXIT】if getattr(self.args, 'tune_mm_mlp_adapter', False):")
        else:
            print("【ENTER】else (not tune_mm_mlp_adapter):")
            super(LLaVATrainer, self)._save(output_dir, state_dict)
            print("【EXIT】else (not tune_mm_mlp_adapter):")
