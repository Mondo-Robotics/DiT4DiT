# Copyright 2025 DiT4DiT team. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].
# Modified by [Teli Ma/ HKUST GZ] in [2025]. 
# Modification: [modify more efficient distributed training and from pre-training mode to training mode].


# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import re

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from DiT4DiT.training.trainer_utils.trainer_tools import normalize_dotlist_args
from DiT4DiT.model.framework import build_framework
from DiT4DiT.training.trainer_utils.trainer_tools import TrainerUtils
from DiT4DiT.training.trainer_utils.trainer_tools import build_param_lr_groups
from DiT4DiT.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig
# 获取本地 Rank（Ray 通常会自动设置 LOCAL_RANK 环境变量）
# local_rank = int(os.environ.get("LOCAL_RANK", 0))

# 强制绑定设备，消除警告
# torch.cuda.set_device(local_rank)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config="DiT4DiT/config/deepseeds/ds_config.yaml")
# deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)


def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # # save config
        # OmegaConf.save(cfg, output_dir / "config.yaml")
        # with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
        #     yaml_cfg = yaml.safe_load(f_yaml)
        #     json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from DiT4DiT.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    # Access in main process so this key is tracked and persisted by AccessTrackedConfig.
    action_video_freq_ratio = cfg.datasets.vla_data.get("action_video_freq_ratio", 1)
    logger.info(f"Using action_video_freq_ratio={action_video_freq_ratio}")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
    
    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights
        self._init_checkpointing()

        # 根据  resume 调整 lr_scheduler
        self._adjust_lr_scheduler_for_resume()

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        # Guard: if everything is frozen, Deepspeed ZeRO optimizer init will crash with an opaque
        # `torch.cat(): expected a non-empty list of Tensors`.
        # any_trainable = any(p.requires_grad for p in self.model.parameters())
        # if not any_trainable:
        #     # Show a few top-level module names to help locate what got frozen.
        #     top_modules = list(dict(self.model.named_children()).keys())
        #     raise RuntimeError(
        #         "No trainable parameters found after freezing. "
        #         "Please check `trainer.freeze_modules` and any backbone wrappers that set requires_grad_(False).\n"
        #         f"- trainer.freeze_modules: {freeze_modules!r}\n"
        #         f"- top-level modules: {top_modules}\n"
        #         "Fix: ensure at least the action head stays trainable (e.g., do not freeze `action_model`)."
        #     )

        # # IMPORTANT: the optimizer was built before we freeze modules (see main()).
        # # Deepspeed ZeRO assumes optimizer param groups contain trainable params; if a group becomes empty after
        # # filtering `requires_grad`, it may crash during flattening.
        # if self.optimizer is not None and hasattr(self.optimizer, "param_groups"):
        #     for group in self.optimizer.param_groups:
        #         group["params"] = [p for p in group.get("params", []) if getattr(p, "requires_grad", False)]
        #     # drop empty groups
        #     self.optimizer.param_groups = [g for g in self.optimizer.param_groups if len(g.get("params", [])) > 0]
        #     total_opt_params = sum(len(g.get("params", [])) for g in self.optimizer.param_groups)
        #     if total_opt_params == 0:
        #         raise RuntimeError(
        #             "Optimizer has 0 trainable parameters after freezing/pruning. "
        #             "This will crash Deepspeed ZeRO.\n"
        #             f"- trainer.freeze_modules: {freeze_modules!r}\n"
        #             "Fix: ensure your action head parameters are included in the optimizer param_groups."
        #         )

        # Print parameter statistics after freezing
        if not dist.is_initialized() or dist.get_rank() == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            logger.info("=" * 80)
            logger.info("📊 Model Parameter Statistics (after freezing):")
            logger.info(f"  Total parameters:      {total_params:,} ({total_params / 10**6:.3f}M)")
            logger.info(f"  Trainable parameters:  {trainable_params:,} ({trainable_params / 10**6:.3f}M)")
            logger.info(f"  Frozen parameters:     {frozen_params:,} ({frozen_params / 10**6:.3f}M)")
            logger.info(f"  Trainable ratio:       {trainable_params / total_params * 100:.2f}%")
            logger.info("=" * 80)

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        self._init_wandb()


    def _adjust_lr_scheduler_for_resume(self):
        """根据已完成的步数调整学习率调度器状态"""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            
            # 方法1: 直接模拟已完成的步数（适用于大多数调度器）
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            
            # 或者方法2: 对于某些调度器，可以直接设置最后步数
            # if hasattr(self.lr_scheduler, '_step_count'):
            #     self.lr_scheduler._step_count = self.completed_steps
            
            logger.info(f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}")

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 获取预训练检查点和是否恢复训练的标志
        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint
        if is_resume:
            # 恢复训练状态
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}")
                return None
            else:
                logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
                self.completed_steps = 0

        # 加载预训练权重
        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0
    

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """save current training state"""

        if self.accelerator.is_main_process:

            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
            # ✅ Save accessed configuration only
            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                # self.config.save_accessed_config(
                #     output_dir / "config.json", 
                #     use_original_values=False
                # )
                self.config.save_accessed_config(
                    output_dir / "config.yaml", 
                    use_original_values=False 
                )
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                # add learning rate for each param group
                for group, lr in zip(self.optimizer.param_groups, self.lr_scheduler.get_last_lr()):
                    metrics[f"learning_rate/{group['name']}"] = lr

                # add epoch info
                metrics["epoch"] = round(self.completed_steps * self.config.trainer.gradient_accumulation_steps  / len(self.vla_train_dataloader), 2)

                # record to W&B
                wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"[Exp: {self.config.run_id}] Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)
        # self.vlm_iter = iter(self.vlm_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # update progress
            did_step = bool(self.accelerator.sync_gradients)
            if did_step:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data - t_start_data:.3f}",
                            "model_times": f"{t_end_model - t_start_model:.3f}",
                        }
                    )

            # Only eval/log/save on real optimizer steps (end of accumulation window)
            if did_step:
                # evaluate model (skip action eval in video-only mode)
                video_fm_only = getattr(self.model, "video_fm_only", False)
                if (not video_fm_only) and (self.completed_steps % self.config.trainer.eval_interval == 0):
                    step_metrics = self.eval_action_model(step_metrics)

                # record metrics
                step_metrics["data_time"] = t_end_data - t_start_data
                step_metrics["model_time"] = t_end_model - t_start_model
                self._log_metrics(step_metrics)

                # save checkpoint
                if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                    self._save_checkpoint()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        examples = self._get_next_batch()
        score = 0.0
        num_samples = len(examples)
        actions = [example["action"] for example in examples]  # label
        # Predict actions using the model
        action_mask = [example["action_mask"] for example in examples] # [B, len, action_dim]
        output_dict = self.model.predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]  # B, T, D
            actions = np.array(actions)  # convert actions to numpy.ndarray
            action_mask = np.array(action_mask)  # convert action_mask to numpy.ndarray [B, T, D]
            # Apply action_mask: only compute MSE on valid (True) dimensions
            masked_diff = (normalized_actions - actions) * action_mask
            mse = (masked_diff ** 2).sum() / action_mask.sum()
            step_metrics["mse_score"] = mse

        del examples
        dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            logger.info(f"  len(vla_train_dataloader) = {len(self.vla_train_dataloader)}")
            logger.info(f"  len(vla_train_dataloader.dataset) = {len(self.vla_train_dataloader.dataset)}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            # VLA task forward propagation
            # NOTE: In some DeepSpeed/Accelerate configurations, autograd can be globally disabled in engine.forward().
            # We force-enable grad here to ensure `loss.grad_fn` exists in training.

            with torch.enable_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_dict = self.model.forward(batch_vla)

                    action_loss = output_dict.get("action_loss", None)
                    future_video_loss = output_dict.get("future_video_loss", None)

                    # Validate at least one loss exists
                    if action_loss is None and future_video_loss is None:
                        raise KeyError(
                            "Model output must contain either `action_loss` or `future_video_loss` for training."
                        )
                    if action_loss is not None and not torch.is_tensor(action_loss):
                        raise TypeError(
                            f"`action_loss` must be a torch.Tensor, got {type(action_loss)}. "
                            "This usually means model.forward returned a Python float or did `.item()`."
                        )
                    if future_video_loss is not None and not torch.is_tensor(future_video_loss):
                        raise TypeError(
                            f"`future_video_loss` must be a torch.Tensor, got {type(future_video_loss)}."
                        )

                    # Compute total loss based on training mode:
                    #   video:  total = future_video_loss * scale
                    #   action: total = action_loss
                    #   joint:  total = action_loss + future_video_loss * scale
                    future_video_loss_scaled = None
                    video_loss_scale = 1.0
                    if future_video_loss is not None:
                        try:
                            video_loss_scale = float(getattr(getattr(self.config.trainer, "loss_scale", None), "future_video", 1.0))
                        except Exception:
                            video_loss_scale = 1.0
                        future_video_loss_scaled = future_video_loss * video_loss_scale

                    if action_loss is not None and future_video_loss is not None:
                        # joint: action + video auxiliary
                        total_loss = action_loss + future_video_loss_scaled
                    elif action_loss is not None:
                        # action only
                        total_loss = action_loss
                    else:
                        # video only
                        total_loss = future_video_loss_scaled

                    # VLA backward propagation (keep inside enable_grad scope)
                    self.accelerator.backward(total_loss)

            # Only step optimizer / scheduler when gradients are synchronized (i.e., end of accumulation window).
            if self.accelerator.sync_gradients:
                # gradient clipping
                if self.config.trainer.gradient_clipping is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

                # optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

        step_metrics = {}
        if action_loss is not None:
            step_metrics["action_dit_loss"] = action_loss.item()
        if future_video_loss is not None:
            # raw auxiliary loss
            step_metrics["future_video_loss"] = future_video_loss.item() if torch.is_tensor(future_video_loss) else float(future_video_loss)
        if future_video_loss_scaled is not None:
            step_metrics["future_video_loss_scaled"] = (
                future_video_loss_scaled.item()
                if torch.is_tensor(future_video_loss_scaled)
                else float(future_video_loss_scaled)
            )
        # total loss (for quick monitoring only)
        step_metrics["total_loss"] = total_loss.item() if torch.is_tensor(total_loss) else float(total_loss)
        return step_metrics

    def _finalize_training(self):
        """training end processing"""
        # save final model
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")


        # close W&B
        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    #  Wrap config to enable access tracking
    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model
    vla = build_framework(cfg)
    # prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="xxx.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    main(cfg)
