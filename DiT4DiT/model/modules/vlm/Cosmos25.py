# Copyright 2025 DiT4DiT team. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Teli Ma/ HKUST GZ] in [2025]. 

"""
Cosmos 2.5 feature extractor backend for DiT4DiT.

Goal:
  - Replace VLM backbone with a video diffusion transformer feature extractor.
  - Expose a lightweight interface compatible with DiT4DiT frameworks:
      - build_cosmos_inputs(images, instructions, ...) -> dict
      - forward(**inputs) -> object/dict containing `hidden_states` (list[Tensor])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
def _exists(v) -> bool:
    return v is not None


@dataclass
class BackboneOutput:
    """Minimal output container."""

    hidden_states: List[torch.Tensor]
    future_video_loss: Optional[torch.Tensor] = None
    pred_future_video: Optional[torch.Tensor] = None  # (B,T,3,H,W) in [0,1] if returned


class _DefaultDummySafetyChecker:
    """Used only to construct Cosmos pipeline without importing/downloading guardrail."""

    def __init__(self, *args, **kwargs):
        pass

    def to(self, device):
        return self

    def check_text_safety(self, text):
        return True

    def check_video_safety(self, video):
        return video


class Cosmos25FeatureExtractor(nn.Module):
    """
    Feature extractor based on `diffusers.Cosmos2_5_PredictBasePipeline`.
      - returns a hidden feature tensor (hooked from a transformer block)
      - optionally returns repeated per-layer hidden list for Layerwise DiT conditioning
    """

    def __init__(
        self,
        model_id_or_path: str,
        *,
        revision: str = "diffusers/base/post-trained",
        torch_dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = True,
        extract_layer: int = 0,
        max_sequence_length: int = 512,
        trainable: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        safety_checker: Any = None,
        config: Optional[Any] = None,
    ):
        super().__init__()
        self.config = config

        from diffusers import Cosmos2_5_PredictBasePipeline

        self.model_id_or_path = model_id_or_path
        self.revision = revision
        self.torch_dtype = torch_dtype
        self.local_files_only = local_files_only
        self.extract_layer = extract_layer
        self.max_sequence_length = max_sequence_length
        self.trainable = trainable

        if safety_checker is None:
            safety_checker = _DefaultDummySafetyChecker()

        pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
            model_id_or_path,
            revision=revision,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            safety_checker=safety_checker,
        )

        self.text_encoder = pipe.text_encoder
        self.tokenizer = pipe.tokenizer
        self.transformer = pipe.transformer
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler
        self.video_processor = pipe.video_processor

        self.latents_mean = getattr(pipe, "latents_mean", None)
        self.latents_std = getattr(pipe, "latents_std", None)
        self.vae_scale_factor_temporal = getattr(pipe, "vae_scale_factor_temporal", None)
        self.vae_scale_factor_spatial = getattr(pipe, "vae_scale_factor_spatial", None)

        self._hook_handle = None
        self._cached_hidden: List[torch.Tensor] = []
        # Used to freeze hidden capture at step 0 (so later denoising steps don't overwrite it)
        self._capture_hidden_enabled: bool = True
        self._register_hook()

        if _exists(device):
            self.to(device)

        # Free memory
        del pipe



    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def hidden_size(self) -> Optional[int]:
        # Best-effort inference from common config names; frameworks may override via config if None.
        cfg = getattr(self.transformer, "config", None)
        if cfg is None:
            return None
        for key in ("hidden_size", "inner_dim", "model_dim", "cross_attention_dim"):
            v = getattr(cfg, key, None)
            if isinstance(v, int) and v > 0:
                return v
        return None

    def __del__(self):
        if _exists(self._hook_handle):
            try:
                self._hook_handle.remove()
            except Exception:
                pass

    def _register_hook(self):
        if not hasattr(self.transformer, "transformer_blocks"):
            return
        blocks = getattr(self.transformer, "transformer_blocks")
        if not isinstance(blocks, (list, nn.ModuleList)) or len(blocks) == 0:
            return
        if self.extract_layer < 0 or self.extract_layer >= len(blocks):
            raise ValueError(f"extract_layer={self.extract_layer} out of bounds for {len(blocks)} blocks")

        target_layer = blocks[self.extract_layer]

        def hook_fn(module, inp, out):
            if not getattr(self, "_capture_hidden_enabled", True):
                return
            if torch.is_tensor(out):
                self._cached_hidden.append(out.detach())
            elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                self._cached_hidden.append(out[0].detach())

        self._hook_handle = target_layer.register_forward_hook(hook_fn)

    def _get_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        *,
        device: torch.device,
        dtype: torch.dtype,
        max_sequence_length: int,
    ) -> torch.Tensor:
        prompt_list = [prompt] if isinstance(prompt, str) else prompt

        input_ids_batch = []
        for sample_idx in range(len(prompt_list)):
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant who will provide prompts to an image generator.",
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt_list[sample_idx]}]},
            ]

            input_ids = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
                max_length=max_sequence_length,
                truncation=True,
                padding="max_length",
            )
            input_ids_batch.append(torch.LongTensor(input_ids))

        input_ids_batch = torch.stack(input_ids_batch, dim=0).to(device)
        outputs = self.text_encoder(input_ids_batch, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        normalized_hidden_states = []
        for layer_idx in range(1, len(hidden_states)):
            hs = hidden_states[layer_idx]
            hs = (hs - hs.mean(dim=-1, keepdim=True)) / (hs.std(dim=-1, keepdim=True) + 1e-8)
            normalized_hidden_states.append(hs)

        prompt_embeds = torch.cat(normalized_hidden_states, dim=-1).to(dtype=dtype, device=device)
        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        *,
        guidance_scale: float = 1.0,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype
        max_sequence_length = max_sequence_length or self.max_sequence_length

        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt_list) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_prompt_embeds(
                prompt_list, device=device, dtype=dtype, max_sequence_length=max_sequence_length
            )
            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        # Feature extraction path uses guidance_scale=1.0 -> no negative embeds needed.
        _ = guidance_scale
        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        *,
        video: Optional[Union[torch.Tensor, List[Any]]],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames_in: int,
        num_frames_out: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.utils.torch_utils import randn_tensor

        B = batch_size
        C = num_channels_latents
        if not _exists(self.vae_scale_factor_temporal) or not _exists(self.vae_scale_factor_spatial):
            raise ValueError("missing VAE scale factors")

        T = (num_frames_out - 1) // int(self.vae_scale_factor_temporal) + 1
        H = height // int(self.vae_scale_factor_spatial)
        W = width // int(self.vae_scale_factor_spatial)
        shape = (B, C, T, H, W)

        if num_frames_in == 0:
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            cond_mask = torch.zeros((B, 1, T, H, W), dtype=latents.dtype, device=latents.device)
            cond_indicator = torch.zeros((B, 1, T, 1, 1), dtype=latents.dtype, device=latents.device)
            cond_latents = torch.zeros_like(latents)
            return latents, cond_latents, cond_mask, cond_indicator

        if video is None:
            raise ValueError("`video` must be provided when `num_frames_in` > 0")

        needs_preprocessing = not (
            isinstance(video, torch.Tensor) and video.ndim == 5 and (video.shape[1] == 3 or video.shape[2] == 3)
        )
        if needs_preprocessing:
            raise ValueError("`video` must be a 5D torch.Tensor in (B,3,T,H,W) or (B,T,3,H,W)")

        # Normalize to VAE dtype before encode
        video_t = video.to(device=device, dtype=self.vae.dtype)
        if video_t.shape[1] == 3:
            video_bcthw = video_t
        else:
            # preprocess_video expects (B,T,C,H,W) and returns (B,C,T,H,W)
            video_bcthw = self.video_processor.preprocess_video(video_t, height=height, width=width)

        # Pad video in pixel space to num_frames_out BEFORE VAE encode
        # This matches the official Cosmos pipeline behavior
        temporal_factor = int(self.vae_scale_factor_temporal) if _exists(self.vae_scale_factor_temporal) else 1
        temporal_factor = max(1, temporal_factor)
        T_in_pix = int(video_bcthw.shape[2])

        if T_in_pix < num_frames_out:
            # Pad with zeros in pixel space (matching official implementation)
            n_pad = num_frames_out - T_in_pix
            zero_pad = torch.zeros(
                (video_bcthw.shape[0], video_bcthw.shape[1], n_pad, video_bcthw.shape[3], video_bcthw.shape[4]),
                device=video_bcthw.device,
                dtype=video_bcthw.dtype,
            )
            video_bcthw_padded = torch.cat([video_bcthw, zero_pad], dim=2)
            # video_bcthw_padded = video_bcthw.repeat(1, 1, num_frames_out, 1, 1)
        else:
            video_bcthw_padded = video_bcthw

        # Encode the padded video
        if isinstance(generator, list):
            cond_latents_list = [
                self.vae.encode(video_bcthw_padded[i].unsqueeze(0)).latent_dist.sample(generator=generator[i])
                for i in range(batch_size)
            ]
        else:
            cond_latents_list = [
                self.vae.encode(vid.unsqueeze(0)).latent_dist.sample(generator=generator) for vid in video_bcthw_padded
            ]

        cond_latents = torch.cat(cond_latents_list, dim=0).to(dtype)

        if self.latents_mean is None or self.latents_std is None:
            raise ValueError("VAE configuration must define both `latents_mean` and `latents_std`.")

        latents_mean = self.latents_mean.to(device=device, dtype=dtype)
        latents_std = self.latents_std.to(device=device, dtype=dtype)
        cond_latents = (cond_latents - latents_mean) / latents_std

        # Compute output latent shape
        T_out_lat = (num_frames_out - 1) // temporal_factor + 1

        H_lat = int(cond_latents.shape[-2])
        W_lat = int(cond_latents.shape[-1])
        shape_out = (B, C, int(T_out_lat), H_lat, W_lat)

        if latents is None:
            # Generate noise with same pattern across batch dimension
            if generator is not None:
                single_noise = randn_tensor((1, C, int(T_out_lat), H_lat, W_lat), generator=generator, device=device, dtype=dtype)
                latents = single_noise.repeat(B, 1, 1, 1, 1)
            else:
                latents = randn_tensor(shape_out, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
            if tuple(latents.shape) != shape_out:
                raise ValueError(f"Provided `latents` shape {tuple(latents.shape)} != expected latent shape {shape_out}")

        # cond_latents should now already have the correct shape from VAE encode
        # Just ensure it matches shape_out
        if cond_latents.shape[2] != T_out_lat:
            # Truncate or pad if needed (should rarely happen)
            cond_latents_adjusted = latents.new_zeros(shape_out)
            T_copy = min(int(cond_latents.shape[2]), int(T_out_lat))
            cond_latents_adjusted[:, :, :T_copy] = cond_latents[:, :, :T_copy].to(dtype=dtype, device=device)
            cond_latents = cond_latents_adjusted

        _, _, T_lat, H_lat, W_lat = shape_out
        ones_padding = latents.new_ones((B, 1, T_lat, H_lat, W_lat))
        zeros_padding = latents.new_zeros((B, 1, T_lat, H_lat, W_lat))

        num_cond_latent_frames = min(T_lat, (num_frames_in - 1) // temporal_factor + 1)

        cond_indicator = latents.new_zeros(B, 1, T_lat, 1, 1)
        cond_indicator[:, :, 0:num_cond_latent_frames] = 1.0
        cond_mask = cond_indicator * ones_padding + (1.0 - cond_indicator) * zeros_padding

        return latents, cond_latents, cond_mask, cond_indicator

    def _denorm_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Inverse of normalization done in `prepare_latents`:
            latents_norm = (latents_raw - mean) / std
        so:
            latents_raw = latents_norm * std + mean
        Supports mean/std with or without a temporal dimension.
        """
        if self.latents_mean is None or self.latents_std is None:
            raise ValueError("VAE configuration must define both `latents_mean` and `latents_std`.")

        mean = self.latents_mean.to(device=latents.device, dtype=latents.dtype)
        std = self.latents_std.to(device=latents.device, dtype=latents.dtype)

        # If mean/std have a temporal dimension, slice to match
        if mean.ndim == 5 and latents.ndim == 5 and mean.shape[2] >= latents.shape[2]:
            mean = mean[:, :, : latents.shape[2]]
        if std.ndim == 5 and latents.ndim == 5 and std.shape[2] >= latents.shape[2]:
            std = std[:, :, : latents.shape[2]]

        return latents * std + mean

    def _match_num_frames(self, video: torch.Tensor, target_num_frames: int) -> torch.Tensor:
        if target_num_frames <= 0 or video.shape[2] == target_num_frames:
            return video

        frames_per_latent = max(self.vae_scale_factor_temporal, 1)
        video = torch.repeat_interleave(video, repeats=frames_per_latent, dim=2)

        current_frames = video.shape[2]
        if current_frames < target_num_frames:
            pad = video[:, :, -1:, :, :].repeat(1, 1, target_num_frames - current_frames, 1, 1)
            video = torch.cat([video, pad], dim=2)
        elif current_frames > target_num_frames:
            video = video[:, :, :target_num_frames]

        return video

    def _decode_latents_to_video(self, latents_norm: torch.Tensor) -> torch.Tensor:
        """
        Decode normalized VAE latents to pixel video in [0,1].
        Returns shape (B,T,3,H,W).
        """


        latents_raw = self._denorm_latents(latents_norm).to(dtype=self.vae.dtype)
        out = self.vae.decode(latents_raw, return_dict=False)[0]
        x = out

        # Normalize layout to (B,T,3,H,W)
        if x.ndim == 5:
            # common: (B,3,T,H,W)
            if x.shape[1] == 3:
                x = x.permute(0, 2, 1, 3, 4).contiguous()
            # else assume already (B,T,3,H,W)
        elif x.ndim == 4:
            # (B,3,H,W) -> single frame
            if x.shape[1] == 3:
                x = x.unsqueeze(1)
            else:
                raise ValueError(f"Unexpected decoded tensor shape {tuple(x.shape)}")
        else:
            raise ValueError(f"Unexpected decoded tensor ndim={x.ndim} shape={tuple(x.shape)}")

        x = x.float()
        min_val, max_val = x.min().item(), x.max().item()

        if min_val < -0.5:  # 真正的 [-1, 1] 范围
            x = (x + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)


        return x


    def _encode_video_to_latents_norm(self, video_bcthw: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel video (B,3,T,H,W) into normalized latents (B,C,T_lat,H_lat,W_lat).
        Deterministic: uses encoder mean.
        """
        if video_bcthw.ndim != 5 or video_bcthw.shape[1] != 3:
            raise ValueError(f"video_bcthw must be (B,3,T,H,W), got {tuple(video_bcthw.shape)}")
        if self.latents_mean is None or self.latents_std is None:
            raise ValueError("VAE configuration must define both `latents_mean` and `latents_std`.")

        vb = video_bcthw.to(device=self.device, dtype=self.vae.dtype)
        enc = self.vae.encode(vb)
        if hasattr(enc, "latent_dist"):
            latents_raw = enc.latent_dist.mean
        else:
            latents_raw = enc

        latents_raw = latents_raw.to(dtype=torch.float32)
        mean = self.latents_mean.to(device=latents_raw.device, dtype=latents_raw.dtype)
        std = self.latents_std.to(device=latents_raw.device, dtype=latents_raw.dtype)

        if mean.ndim == 5 and mean.shape[2] >= latents_raw.shape[2]:
            mean = mean[:, :, : latents_raw.shape[2]]
        if std.ndim == 5 and std.shape[2] >= latents_raw.shape[2]:
            std = std[:, :, : latents_raw.shape[2]]

        return (latents_raw - mean) / std

    def _coerce_videos_to_bcthw(self, videos: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        if not isinstance(videos, torch.Tensor) or videos.ndim != 5:
            raise ValueError("`videos` must be a 5D torch.Tensor")

        if videos.shape[1] == 3:
            return videos
        if videos.shape[2] == 3:
            return self.video_processor.preprocess_video(videos, height=height, width=width)
        raise ValueError(f"Unsupported `videos` shape {tuple(videos.shape)}; expected (B,3,T,H,W) or (B,T,3,H,W).")

    def forward(
        self,
        *,
        prompts: Union[str, List[str]],
        videos: torch.Tensor,
        height: int,
        width: int,
        num_inference_steps: int = 1,
        capture_step_index: int = 0,
        conditional_frame_timestep: float = 0.001,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        detach: bool = True,
        num_frames_out: Optional[int] = None,
        gt_future_videos: Optional[torch.Tensor] = None,  # (B,Tf,3,H,W) float in [0,1]
        return_pred_future_video: bool = False,
        fixed_seed: Optional[int] = None,
    ) -> torch.Tensor:
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 16 but are {height} and {width}.")

        device = self.device
        transformer_dtype = self.transformer.dtype

        b = int(videos.shape[0])
        if isinstance(prompts, str):
            prompts_list: List[str] = [prompts] * b
        else:
            prompts_list = prompts
            if len(prompts_list) != b:
                raise ValueError(f"len(prompts)={len(prompts_list)} must match batch size B={b}")

        # Create fixed generator if seed is provided
        if fixed_seed is not None and generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(fixed_seed)

        # ctx = torch.enable_grad() if self.trainable else torch.no_grad()
        with torch.enable_grad():
            prompt_embeds, _ = self.encode_prompt(
                prompt=prompts_list,
                guidance_scale=1.0,
                num_videos_per_prompt=1,
                device=device,
                max_sequence_length=self.max_sequence_length,
            )

            videos_bcthw = self._coerce_videos_to_bcthw(videos, height=height, width=width)
            videos_bcthw = videos_bcthw.to(device=device, dtype=self.vae.dtype) ##[-1,1]
            num_frames_in = int(videos_bcthw.shape[2])
            # NOTE: Cosmos VAE is temporally downsampled (vae_scale_factor_temporal > 1).
            temporal_factor = int(getattr(self, "vae_scale_factor_temporal", 1) or 1)
            temporal_factor = max(1, temporal_factor)
            min_frames_out_for_future = 1 + temporal_factor

            if num_frames_out is None:
                if gt_future_videos is not None and torch.is_tensor(gt_future_videos):
                    # gt_future_videos: (B,T_f,3,H,W) or (B,3,T_f,H,W)
                    gt_t_dim = gt_future_videos.shape[1] if gt_future_videos.shape[2] == 3 else gt_future_videos.shape[2]
                    num_frames_out = num_frames_in + int(gt_t_dim)
                else:
                    num_frames_out = num_frames_in
                # Ensure enough output frames for at least one future latent slot
                num_frames_out = max(int(num_frames_out), int(min_frames_out_for_future))

            # If we need any future prediction/supervision, ensure enough output frames to create a future latent slot.
            # if (gt_future_videos is not None and torch.is_tensor(gt_future_videos)) or bool(return_pred_future_video):
            #     num_frames_out = max(int(num_frames_out), int(min_frames_out_for_future))

            num_channels_latents = int(self.transformer.config.in_channels) - 1
            latents, cond_latents, cond_mask, cond_indicator = self.prepare_latents(
                video=videos_bcthw,
                batch_size=b,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames_in=num_frames_in,
                num_frames_out=int(num_frames_out),
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=None,
            )

            cond_timestep = torch.ones_like(cond_indicator) * float(conditional_frame_timestep)
            cond_mask_t = cond_mask.to(transformer_dtype)
            padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)

            self._cached_hidden.clear()
            self._capture_hidden_enabled = True

            # Resolve future loss type early so we can skip expensive sampling/decoding when training with
            # latent-space flow-matching loss only.
            loss_type = None
            if gt_future_videos is not None:
                cosmos_cfg = getattr(getattr(self, "config", None), "framework", None)
                cosmos_cfg = getattr(cosmos_cfg, "cosmos25", None) if cosmos_cfg is not None else None
                if cosmos_cfg is not None:
                    if isinstance(cosmos_cfg, dict):
                        loss_type = str(cosmos_cfg.get("future_loss_type", "pixel_l1")).lower()
                    else:
                        loss_type = str(getattr(cosmos_cfg, "future_loss_type", "pixel_l1")).lower()
                else:
                    loss_type = "pixel_l1"

            flow_matching_types = ("flow_matching", "latent_flow_matching", "rectified_flow", "rf")
            train_flow_matching_only = (gt_future_videos is not None) and (loss_type in flow_matching_types) and (not return_pred_future_video)

            # If we need generated video / future supervision, run the full denoising loop.
            # Otherwise, we can early-exit after `capture_step_index` (default 0).
            #
            # Match diffusers' Cosmos2.5 Predict pipeline behavior:
            # - Use UniPCMultistepScheduler with `prediction_type="flow_prediction"` + `use_flow_sigmas=True`
            # - Treat transformer output as velocity (a.k.a. flow) and replace conditioned frames with gt_velocity
            # For pure flow-matching training, we don't need to run the sampling loop or decode videos.
            need_generate = ((gt_future_videos is not None) or bool(return_pred_future_video)) and (not train_flow_matching_only)

            # If we aren't generating, only run enough steps to reach capture_step_index (usually 0).
            steps_for_hidden = int(num_inference_steps) if need_generate else max(1, int(capture_step_index) + 1)
            self.scheduler.set_timesteps(steps_for_hidden, device=device)
            timesteps = self.scheduler.timesteps

            stop_i = (len(timesteps) - 1) if need_generate else capture_step_index
            if stop_i < 0 or stop_i >= len(timesteps):
                raise ValueError(f"stop_i={stop_i} out of range for {len(timesteps)} steps")

            # Determine whether scheduler expects flow/velocity outputs.
            scheduler_cfg = getattr(self.scheduler, "config", None)
            prediction_type = str(getattr(scheduler_cfg, "prediction_type", "")).lower() if scheduler_cfg is not None else ""
            use_flow_prediction = prediction_type == "flow_prediction"

            # In diffusers pipeline_cosmos2_5_predict.py:
            #   gt_velocity = (latents - cond_latent) * cond_mask
            # where `latents` are the initial noise latents and `cond_latent` are the (padded) conditional latents.
            gt_velocity = None
            if use_flow_prediction:
                gt_velocity = (latents - cond_latents).to(dtype=transformer_dtype) * cond_mask_t

            hidden_first: Optional[torch.Tensor] = None
            for i, t in enumerate(timesteps):
                # NOTE: for Cosmos2.5 Predict with flow sigmas, scheduler.sigmas are mapped to [0,1]
                sigma_t = (
                    torch.tensor(self.scheduler.sigmas[i].item()).unsqueeze(0).to(device=device, dtype=transformer_dtype)
                )
                in_latents = cond_mask_t * cond_latents + (1 - cond_mask_t) * latents
                in_latents = in_latents.to(transformer_dtype)
                in_timestep = cond_indicator * cond_timestep + (1 - cond_indicator) * sigma_t

                model_out = self.transformer(
                    hidden_states=in_latents,
                    condition_mask=cond_mask_t,
                    timestep=in_timestep,
                    encoder_hidden_states=prompt_embeds,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]

                # Replace conditional-frame velocity with gt_velocity (diffusers behavior)
                if gt_velocity is not None:
                    model_out = gt_velocity + model_out * (1 - cond_mask_t)


                # Capture hidden from the FIRST transformer call and freeze capture afterwards.
                if i == 0 and hidden_first is None and len(self._cached_hidden) > 0:
                    hidden_first = self._cached_hidden[-1]
                    self._capture_hidden_enabled = False

                # Only advance diffusion if we actually need generated video / future loss.
                if need_generate:
                    latents = self.scheduler.step(model_out, t, latents, return_dict=False)[0]

                if i == stop_i:
                    break

            if len(self._cached_hidden) == 0:
                raise RuntimeError(
                    "No transformer hidden captured. Your transformer may not expose `transformer_blocks`, "
                    "or `extract_layer` is not a valid block index."
                )

            hidden = hidden_first if hidden_first is not None else self._cached_hidden[-1]

            # Optional future-video prediction + auxiliary supervision loss
            future_loss = None
            pred_video = None
            # gt_future_videos = None
            if gt_future_videos is not None or return_pred_future_video:
                # Only decode when needed:
                # - return_pred_future_video=True
                # - pixel-space supervision (pixel_l1 / pixel_mse)
                need_decode = bool(return_pred_future_video) or (
                    (gt_future_videos is not None) and (loss_type is not None) and (loss_type not in ("latent_mse",) + flow_matching_types)
                )
                pred_video_full = None
                if need_decode:
                    pred_video_full = self._decode_latents_to_video(latents)  # (B,T,3,H,W)
                    if return_pred_future_video:
                        pred_video = pred_video_full

                if gt_future_videos is not None:
                    gt = gt_future_videos
                    if gt.ndim == 5 and gt.shape[2] == 3:
                        # (B,T,3,H,W) OK
                        pass
                    elif gt.ndim == 5 and gt.shape[1] == 3:
                        # (B,3,T,H,W) -> (B,T,3,H,W)
                        gt = gt.permute(0, 2, 1, 3, 4).contiguous()
                    else:
                        raise ValueError(f"gt_future_videos must be (B,T,3,H,W) or (B,3,T,H,W), got {tuple(gt.shape)}")

                    # Loss type: pixel_l1 | pixel_mse | latent_mse | flow_matching (latent-space)
                    # `loss_type` is resolved above to allow skipping sampling/decode in flow-matching-only training.
                    loss_type = loss_type or "pixel_l1"

                    if loss_type in flow_matching_types:
                        # Latent-space flow matching (rectified flow) loss, only on future latent frames.
                        #
                        # We construct x_t = (1 - t) * x0 + t * z, and train the model to predict velocity:
                        #   v*(x_t, t) = z - x0
                        # conditioned frames are masked out, and we only supervise future frames.

                        # Build full GT pixel video (B,3,T_in+T_f,H,W) and pad GT if needed so that
                        # VAE temporal downsampling produces at least one "future latent slot" (important when tf>1).
                        gt_bcthw = gt.permute(0, 2, 1, 3, 4).contiguous()
                        # gt 来自 dataloader，是 [0,1] 范围，需要转换到 [-1,1] 以匹配 videos_bcthw
                        if gt_bcthw.min() >= 0:
                            gt_bcthw = gt_bcthw * 2.0 - 1.0
                        full_bcthw = torch.cat([videos_bcthw.to(gt_bcthw.device, dtype=gt_bcthw.dtype), gt_bcthw], dim=2)

                        # Ensure we have enough *pixel* frames to create a future latent timestep.
                        temporal_factor = int(getattr(self, "vae_scale_factor_temporal", 1) or 1)
                        temporal_factor = max(1, temporal_factor)
                        min_full_frames = 1 + temporal_factor  # 2 latent timesteps in pixel domain
                        if int(full_bcthw.shape[2]) < int(min_full_frames):
                            n_pad = int(min_full_frames - int(full_bcthw.shape[2]))
                            last = full_bcthw[:, :, -1:, :, :].repeat(1, 1, n_pad, 1, 1)
                            full_bcthw = torch.cat([full_bcthw, last], dim=2)

                        with torch.no_grad():
                            gt_latents_norm = self._encode_video_to_latents_norm(full_bcthw)

                        cond_count = int(cond_indicator[0, 0, :, 0, 0].sum().item())
                        pred_latents_future = latents[:, :, cond_count:]
                        gt_latents_future = gt_latents_norm[:, :, cond_count : cond_count + pred_latents_future.shape[2]]

                        # If we still don't have a future latent to supervise, return 0.
                        if pred_latents_future.numel() == 0 or gt_latents_future.numel() == 0:
                            future_loss = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
                        else:
                            # Align lengths (just in case)
                            T_sup = min(int(pred_latents_future.shape[2]), int(gt_latents_future.shape[2]))
                            x0_future = gt_latents_future[:, :, :T_sup].to(device=latents.device, dtype=torch.float32)

                            B_fm = latents.shape[0]
                            # Read flow-matching config: time_distribution and high_sigma_strategy
                            fm_cfg = getattr(cosmos_cfg, "flow_matching", None) if cosmos_cfg is not None else None
                            time_dist = str(getattr(fm_cfg, "time_distribution", "logit_normal")) if fm_cfg else "logit_normal"
                            # high_sigma_ratio = float(getattr(fm_cfg, "high_sigma_ratio", 0.05)) if fm_cfg else 0.05
                            _hsr = getattr(fm_cfg, "high_sigma_ratio", 0.05) if fm_cfg else 0.05
                            _hsm = getattr(fm_cfg, "high_sigma_min", 0.98) if fm_cfg else 0.98
                            high_sigma_ratio = None if _hsr is None else float(_hsr)
                            high_sigma_min = None if _hsm is None else float(_hsm)

                            # Sample rectified-flow time t in [0,1]
                            if time_dist == "logit_normal":
                                # Logit-normal: t = sigmoid(N(0,1)), concentrates around t≈0.5
                                t = torch.sigmoid(torch.randn((B_fm,), device=latents.device, dtype=torch.float32))
                            else:
                                t = torch.rand((B_fm,), device=latents.device, dtype=torch.float32)

                            # High sigma strategy: force a fraction of samples to high noise (t close to 1)
                            if high_sigma_ratio is not None and high_sigma_ratio > 0:
                                high_mask = torch.rand((B_fm,), device=latents.device) < high_sigma_ratio
                                high_t = torch.rand((B_fm,), device=latents.device, dtype=torch.float32) * (1.0 - high_sigma_min) + high_sigma_min
                                t = torch.where(high_mask, high_t, t)

                            t = t.view(B_fm, 1, 1, 1, 1)  # (B,1,1,1,1) for latent broadcast

                            # Sample noise z
                            z_future = torch.randn_like(x0_future)

                            # x_t for future frames only
                            xt_future = (1.0 - t) * x0_future + t * z_future

                            # Build a full latent tensor for conditioning:
                            # - cond frames: pinned via in_latents mixing (cond_mask_t * cond_latents)
                            # - future frames: use x_t for supervised portion; fill the rest with noise
                            xt_full = torch.randn_like(latents[:, :, :, :, :].float())
                            xt_full[:, :, cond_count : cond_count + T_sup] = xt_future

                            # timestep tensor shaped like (B,1,T,1,1): cond frames use cond_timestep, future use t
                            t_B1T11 = latents.new_zeros(cond_indicator.shape, dtype=torch.float32)
                            t_B1T11[:, :, cond_count : cond_count + T_sup] = t  # broadcast over time slice
                            t_B1T11 = t_B1T11.to(dtype=transformer_dtype)

                            in_latents_fm = cond_mask_t * cond_latents + (1 - cond_mask_t) * xt_full.to(transformer_dtype)
                            in_timestep_fm = cond_indicator * cond_timestep + (1 - cond_indicator) * t_B1T11

                            v_pred = self.transformer(
                                hidden_states=in_latents_fm,
                                condition_mask=cond_mask_t,
                                timestep=in_timestep_fm,
                                encoder_hidden_states=prompt_embeds,
                                padding_mask=padding_mask,
                                return_dict=False,
                            )[0]

                            # Supervise only future frames: v* = z - x0
                            v_tgt_future = (z_future - x0_future).to(device=v_pred.device, dtype=v_pred.dtype)
                            v_pred_future = v_pred[:, :, cond_count : cond_count + T_sup]

                            future_loss = F.mse_loss(v_pred_future.float(), v_tgt_future.float())
                            
                    elif loss_type == "latent_mse":
                        # Build full GT pixel video (B,3,T_in+T_f,H,W)
                        gt_bcthw = gt.permute(0, 2, 1, 3, 4).contiguous()
                        full_bcthw = torch.cat([videos_bcthw.to(gt_bcthw.device, dtype=gt_bcthw.dtype), gt_bcthw], dim=2)

                        with torch.no_grad():
                            gt_latents_norm = self._encode_video_to_latents_norm(full_bcthw)

                        # conditional latent frames count (from cond_indicator)
                        cond_count = int(cond_indicator[0, 0, :, 0, 0].sum().item())
                        pred_latents_future = latents[:, :, cond_count:]
                        gt_latents_future = gt_latents_norm[:, :, cond_count : cond_count + pred_latents_future.shape[2]]

                        if pred_latents_future.numel() == 0 or gt_latents_future.numel() == 0:
                            future_loss = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
                        else:
                            future_loss = F.mse_loss(
                                pred_latents_future.float(),
                                gt_latents_future.to(pred_latents_future.device, dtype=torch.float32),
                            )
                    else:
                        # Pixel supervision: only on future frames (skip t0)
                        if pred_video_full is None:
                            # Should not happen because pixel loss requires decode, but keep safe.
                            future_loss = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
                        else:
                            tf = int(gt.shape[1])
                            if pred_video_full.shape[1] >= 1 + tf:
                                pred_future = pred_video_full[:, 1 : 1 + tf]
                            else:
                                pred_future = pred_video_full[:, -tf:]
                            gt_pix = gt.to(pred_future.device, dtype=pred_future.dtype)
                            if loss_type in ("pixel_mse", "mse"):
                                future_loss = F.mse_loss(pred_future, gt_pix)
                            else:
                                future_loss = torch.abs(pred_future - gt_pix).mean()

            if detach:
                hidden = hidden.detach()
                if pred_video is not None:
                    pred_video = pred_video.detach()

            if gt_future_videos is not None or return_pred_future_video:
                return hidden, pred_video, future_loss  # type: ignore[return-value]

            return hidden


class _Cosmos25_Interface(nn.Module):
    """
    Interface wrapper around Cosmos25FeatureExtractor.

    Expected usage pattern:
      inputs = build_cosmos_inputs(images=batch_images, instructions=instructions)
      out = self(**inputs, output_hidden_states=True, return_dict=True)
      out.hidden_states -> list[Tensor] each (B, S, D)
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        self.config = config

        cosmos_cfg = getattr(config.framework, "cosmos25", None) if config is not None else None
        if cosmos_cfg is None:
            cosmos_cfg = {}

        model_id = cosmos_cfg.get("model_id_or_path", None) or cosmos_cfg.get("base_model", None)
        if not model_id:
            raise ValueError("Cosmos25 backend requires `framework.cosmos25.model_id_or_path`")

        self.extract_layer = int(cosmos_cfg.get("extract_layer", 19))
        self.max_sequence_length = int(cosmos_cfg.get("max_sequence_length", 512))
        self.revision = cosmos_cfg.get("revision", "diffusers/base/post-trained")
        self.local_files_only = bool(cosmos_cfg.get("local_files_only", True))
        self.trainable = bool(cosmos_cfg.get("trainable", False))

        dtype_str = cosmos_cfg.get("torch_dtype", "bfloat16")
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
            str(dtype_str).lower(), torch.bfloat16
        )

        self.extractor = Cosmos25FeatureExtractor(
            model_id_or_path=model_id,
            revision=self.revision,
            torch_dtype=torch_dtype,
            local_files_only=self.local_files_only,
            extract_layer=self.extract_layer,
            max_sequence_length=self.max_sequence_length,
            trainable=self.trainable,
            device=cosmos_cfg.get("device", None),
            config=config,
        )

        # Try to set vl_hidden_dim in config if missing
        if config is not None and getattr(config.framework.cosmos25, "vl_hidden_dim", None) in (None, 0):
            hs = self.extractor.hidden_size
            if hs is not None:
                config.framework.cosmos25.vl_hidden_dim = int(hs)

    @property
    def device(self) -> torch.device:
        return self.extractor.device

    @staticmethod
    def _to_chw_float01(x) -> torch.Tensor:
        """
        Accept PIL / numpy(HWC) / torch(CHW or HWC) and return torch float tensor in CHW, normalized to [0,1].
        """
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu()
            if t.ndim == 3 and t.shape[0] in (1, 3, 4):  # CHW
                if t.shape[0] == 4:
                    t = t[:3]
            elif t.ndim == 3 and t.shape[-1] in (1, 3, 4):  # HWC
                if t.shape[-1] == 4:
                    t = t[..., :3]
                t = t.permute(2, 0, 1).contiguous()
            else:
                raise ValueError(f"Unsupported torch image shape {tuple(t.shape)}; expected CHW or HWC.")
            t = t.float()
            if t.max() > 1.0:
                t = t / 255.0
            return t.contiguous()

        if isinstance(x, np.ndarray):
            arr = x
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            if arr.ndim != 3:
                raise ValueError(f"Expected numpy image with shape (H,W,C), got {arr.shape}")
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.shape[2] != 3:
                raise ValueError(f"Expected RGB image, got shape={arr.shape}")
            t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
            if t.max() > 1.0:
                t = t / 255.0
            return t.contiguous()

        # PIL.Image.Image or compatible
        arr = np.array(x, dtype=np.uint8)  # (H,W,3)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape={arr.shape}")
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
        return t.contiguous()

    def build_cosmos_inputs(
        self,
        images: Sequence[Sequence[Any]],
        instructions: Sequence[str],
        **kwargs,
    ) -> dict:
        """
        Convert multi-frame images + instruction into Cosmos inputs.

        `images` is a list of lists: images[b] = [frame_0, frame_1, ..., frame_T-1]
        where frames are ordered temporally after pixel-space downsampling.

        Frame 0 is used as the conditioning frame (videos), and the remaining
        frames are future frames for video flow-matching supervision (future_videos).
        """

        if len(images) != len(instructions):
            raise ValueError("images and instructions must have the same batch size")

        all_cond = []      # frame 0 per sample
        all_future = []    # frames 1..T-1 per sample
        n_future = None

        for views in images:
            if not isinstance(views, (list, tuple)) or len(views) == 0:
                raise ValueError("Each sample must provide at least 1 view image")

            chw_frames = [self._to_chw_float01(v) for v in views]

            all_cond.append(chw_frames[0])
            if len(chw_frames) > 1:
                future = torch.stack(chw_frames[1:], dim=0)  # (T_f, C, H, W)
                all_future.append(future)
                if n_future is None:
                    n_future = future.shape[0]

        # videos (condition): (B, T=1, C, H, W)
        videos = torch.stack(all_cond, dim=0).unsqueeze(1)
        height, width = int(videos.shape[-2]), int(videos.shape[-1])

        # future_videos: (B, T_f, C, H, W) or None
        future_videos = None
        if len(all_future) > 0:
            future_videos = torch.stack(all_future, dim=0)  # (B, T_f, C, H, W)

        return {
            "prompts": list(instructions),
            "videos": videos,
            "height": height,
            "width": width,
            "future_videos": future_videos,
        }

    @staticmethod
    def _hidden_to_bsd(hidden: torch.Tensor) -> torch.Tensor:
        """
        Normalize different possible hidden shapes to (B, S, D) for DiT conditioning.
        Supported:
          - (B, S, D): returned as-is
          - (B, C, T, H, W): flattened to tokens (S=T*H*W), D=C
        """
        if hidden.ndim == 3:
            return hidden
        if hidden.ndim == 5:
            b, c, t, h, w = hidden.shape
            x = hidden.permute(0, 2, 3, 4, 1).contiguous().view(b, t * h * w, c)
            return x
        raise ValueError(f"Unsupported hidden shape {tuple(hidden.shape)}; expected 3D or 5D tensor.")

    def forward(
        self,
        *,
        prompts: Union[str, List[str]],
        videos: torch.Tensor,
        height: int,
        width: int,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        future_videos: Optional[torch.Tensor] = None,
        predict_future: bool = False,
        **kwargs,
    ):
        _ = kwargs

        future_loss = None
        pred_future_video = None
        fixed_seed = 42
        # future_videos = None

        cosmos_cfg = getattr(self.config.framework, "cosmos25", None) if self.config is not None else None
        conditional_frame_timestep = float(getattr(cosmos_cfg, "conditional_frame_timestep", 0.001)) if cosmos_cfg is not None else 0.001

        # Compute training-time num_frames_out from config so that eval latent
        # shape matches training (video_delta_indices + action_video_freq_ratio).
        train_num_frames_out = None
        if cosmos_cfg is not None and self.config is not None:
            data_cfg = getattr(getattr(self.config, "datasets", None), "vla_data", None)
            if data_cfg is not None:
                video_delta = getattr(data_cfg, "video_delta_indices", None)
                if video_delta is not None:
                    n_raw = len(video_delta)
                    ratio = int(getattr(data_cfg, "action_video_freq_ratio", 1))
                    # number of frames after pixel-space downsampling
                    train_num_frames_out = len(range(0, n_raw, ratio))

        if predict_future or (future_videos is not None):
            future_steps = int(getattr(cosmos_cfg, "future_num_inference_steps", 2)) if cosmos_cfg is not None else 2
            hidden, pred_future_video, future_loss = self.extractor(
                prompts=prompts,
                videos=videos,
                height=height,
                width=width,
                detach=True,
                gt_future_videos=future_videos,
                return_pred_future_video=False,
                num_inference_steps=future_steps,
                num_frames_out=train_num_frames_out,
                conditional_frame_timestep=conditional_frame_timestep,
            )
        else:
            hidden = self.extractor(
                prompts=prompts,
                videos=videos,
                height=height,
                width=width,
                detach=True,
                num_frames_out=train_num_frames_out,
                conditional_frame_timestep=conditional_frame_timestep,
            )

        bsd = self._hidden_to_bsd(hidden)

        # hidden_states = [bsd for _ in range(self.num_vl_layers)] if output_hidden_states else []
        out = BackboneOutput(hidden_states=[bsd], future_video_loss=future_loss, pred_future_video=pred_future_video)
        # return out if return_dict else (out.hidden_states,)
        return out


