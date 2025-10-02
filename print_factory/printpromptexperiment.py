def _save_checkpoint(self, model, trial, metrics=None):

    print("current file path", "llava/train/llava_trainer.py")
    print("def _save_checkpoint(self, model, trial, metrics=None)")
    print("self\n", self) # <llava.train.llava_trainer.LLaVATrainer object at 0x7ed6341f4490>
    print("model\n", model)

    print("trial\n", trial) # None
    print("metrics\n", metrics) # None
    print(f"【COND】 tune_mm_mlp_adapter={getattr(self.args, 'tune_mm_mlp_adapter', False)}") # True
    if getattr(self.args, 'tune_mm_mlp_adapter', False):
        # 【ENTER】
        print("【ENTER】if getattr(self.args, 'tune_mm_mlp_adapter', False):")
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        print("checkpoint_folder = f\"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}\"", checkpoint_folder)

        run_dir = self._get_output_dir(trial=trial)
        print("run_dir = self._get_output_dir(trial=trial)", run_dir)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        print("output_dir = os.path.join(run_dir, checkpoint_folder)", output_dir)

        # Only save Adapter
        keys_to_match = ['mm_projector', 'vision_resampler']
        print("keys_to_match = ['mm_projector', 'vision_resampler']", keys_to_match)
        print(f"【COND】 use_im_start_end={getattr(self.args, 'use_im_start_end', False)}") # False
        if getattr(self.args, "use_im_start_end", False):
            # 【SKIP】
            print("【ENTER】if getattr(self.args, 'use_im_start_end', False):")
            keys_to_match.extend(['embed_tokens', 'embed_in'])
            print("keys_to_match.extend(['embed_tokens', 'embed_in'])", keys_to_match)

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            print("weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)", weight_to_save)

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