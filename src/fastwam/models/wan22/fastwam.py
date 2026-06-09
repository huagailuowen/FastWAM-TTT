from typing import Any, Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler
from .video_ttt import VideoTTTQKVAdapter

logger = get_logger(__name__)


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_video_ttt: float = 0.0,
        video_ttt_config: Optional[dict[str, Any]] = None,
        video_ttt_observation_training: bool = False,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.loss_lambda_video_ttt = float(loss_lambda_video_ttt)
        self.video_ttt_observation_training = bool(video_ttt_observation_training)
        self._video_ttt_inference_state: Optional[dict[str, torch.Tensor]] = None

        video_ttt_config = {} if video_ttt_config is None else dict(video_ttt_config)
        self.video_ttt_train_backbone = bool(video_ttt_config.get("train_backbone", False))
        self.video_ttt_train_residual_gate = bool(video_ttt_config.get("train_residual_gate", True))
        self.video_ttt_num_tries = max(int(video_ttt_config.get("num_tries", 1)), 1)
        self.video_ttt_switch_chunks = bool(video_ttt_config.get("switch_chunks", False))
        self.video_ttt_w0_training = bool(video_ttt_config.get("w0_training", False))
        self.video_ttt_observe_prefill_group_size = max(
            int(video_ttt_config.get("observe_prefill_group_size", 1)),
            1,
        )
        self.video_ttt_execution_backward_group_size = max(
            int(video_ttt_config.get("execution_backward_group_size", 1)),
            1,
        )
        self.video_ttt_time_stride = max(int(video_ttt_config.get("time_stride", 2)), 1)
        self.video_ttt_include_context_tokens = bool(video_ttt_config.get("include_context_tokens", True))
        default_state_tokens = 1 if self.proprio_dim is not None else 0
        self.video_ttt_state_token_count = max(
            int(video_ttt_config.get("state_token_count", default_state_tokens)),
            0,
        )
        video_ttt_uses_layer_hook = bool(
            video_ttt_config.get(
                "use_layer_hook",
                "num_layers" in video_ttt_config
                or video_ttt_config.get("layer_indices", None) is not None
                or video_ttt_config.get("rank", None) is not None,
            )
        )
        self.video_ttt_uses_layer_hook = video_ttt_uses_layer_hook
        video_ttt_num_layers = max(int(video_ttt_config.get("num_layers", 1)), 1)
        configured_layer_indices = video_ttt_config.get("layer_indices", None)
        if not video_ttt_uses_layer_hook:
            layer_indices = ()
        elif configured_layer_indices is None:
            layer_indices = self._uniform_video_ttt_layer_indices(
                total_layers=len(self.video_expert.blocks),
                num_ttt_layers=video_ttt_num_layers,
            )
        else:
            layer_indices = tuple(int(idx) for idx in configured_layer_indices)
            if len(layer_indices) != video_ttt_num_layers:
                raise ValueError(
                    "`video_ttt.layer_indices` length must match `video_ttt.num_layers`, "
                    f"got {len(layer_indices)} and {video_ttt_num_layers}."
                )
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError(f"`video_ttt.layer_indices` must be unique, got {layer_indices}.")
        if any(idx < 0 or idx >= len(self.video_expert.blocks) for idx in layer_indices):
            raise ValueError(
                f"`video_ttt.layer_indices` must be in [0, {len(self.video_expert.blocks)}), got {layer_indices}."
            )
        self.video_ttt_layer_indices = tuple(layer_indices)
        self.video_ttt_layer_to_slot = {int(layer_idx): slot_idx for slot_idx, layer_idx in enumerate(layer_indices)}
        residual_gate_override = video_ttt_config.get("residual_gate_override", None)
        self.video_ttt_residual_gate_override = (
            None if residual_gate_override is None else float(residual_gate_override)
        )
        if bool(video_ttt_config.get("enabled", False)):
            video_ttt_adapter = VideoTTTQKVAdapter(
                hidden_dim=int(self.video_expert.hidden_dim),
                num_heads=int(self.video_expert.num_heads),
                mini_batch_size=int(video_ttt_config.get("mini_batch_size", 64)),
                ttt_lr=float(video_ttt_config.get("ttt_lr", 0.1)),
                init_std=float(video_ttt_config.get("init_std", 0.02)),
                residual_gate_init=float(video_ttt_config.get("residual_gate_init", 0.0)),
                num_layers=video_ttt_num_layers,
                rank=video_ttt_config.get("rank", None),
                inner_update_mode=str(video_ttt_config.get("inner_update_mode", "scan")),
                scan_kernel=str(video_ttt_config.get("scan_kernel", "torch")),
                use_global_rope=bool(video_ttt_config.get("use_global_rope", True)),
                rope_theta=float(video_ttt_config.get("rope_theta", 10000.0)),
            ).to(dtype=torch_dtype)
            self.mot.add_module("video_ttt_adapter", video_ttt_adapter)

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_video_ttt: float = 0.0,
        video_ttt_config: Optional[dict[str, Any]] = None,
        video_ttt_observation_training: bool = False,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            loss_lambda_video_ttt=loss_lambda_video_ttt,
            video_ttt_config=video_ttt_config,
            video_ttt_observation_training=video_ttt_observation_training,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _uniform_video_ttt_layer_indices(*, total_layers: int, num_ttt_layers: int) -> tuple[int, ...]:
        if total_layers <= 0:
            raise ValueError(f"`total_layers` must be positive, got {total_layers}.")
        if num_ttt_layers <= 0:
            raise ValueError(f"`num_ttt_layers` must be positive, got {num_ttt_layers}.")
        if num_ttt_layers > total_layers:
            raise ValueError(
                f"`num_ttt_layers` cannot exceed transformer depth, got {num_ttt_layers} > {total_layers}."
            )
        return tuple(
            min(total_layers - 1, max(0, int(float(i + 1) * total_layers / float(num_ttt_layers + 1))))
            for i in range(num_ttt_layers)
        )

    def _get_video_ttt_adapter(self) -> Optional[VideoTTTQKVAdapter]:
        adapter = getattr(self.mot, "video_ttt_adapter", None)
        if adapter is None:
            return None
        if not isinstance(adapter, VideoTTTQKVAdapter):
            raise TypeError(f"`mot.video_ttt_adapter` must be VideoTTTQKVAdapter, got {type(adapter)}.")
        return adapter

    @property
    def video_ttt_enabled(self) -> bool:
        return self._get_video_ttt_adapter() is not None

    def _video_ttt_gate_metrics(self) -> dict[str, float]:
        adapter = self._get_video_ttt_adapter()
        if adapter is None:
            return {}
        gates = adapter.residual_gate.detach().float()
        nonzero = (gates.abs() > 1.0e-8).to(dtype=torch.float32)
        return {
            "video_ttt_residual_gate_mean": float(gates.mean().item()),
            "video_ttt_residual_gate_abs_mean": float(gates.abs().mean().item()),
            "video_ttt_residual_gate_abs_max": float(gates.abs().max().item()),
            "video_ttt_residual_gate_nonzero_frac": float(nonzero.mean().item()),
            "video_ttt_residual_gate_count": float(gates.numel()),
        }

    def reset_video_ttt_state(self) -> None:
        """Reset inference-time fast weights at an episode/task boundary."""
        self._video_ttt_inference_state = None

    @staticmethod
    def _unpack_video_ttt_payload(payload: Any) -> tuple[Any, Any]:
        if isinstance(payload, dict) and "tokens" in payload:
            return payload["tokens"], payload.get("positions")
        return payload, None

    def _coerce_video_ttt_global_time(
        self,
        global_time: Optional[torch.Tensor | int | float],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if global_time is None:
            return torch.zeros((batch_size,), device=device, dtype=torch.float32)
        if not isinstance(global_time, torch.Tensor):
            global_time = torch.as_tensor(global_time)
        global_time = global_time.to(device=device, dtype=torch.float32)
        if global_time.ndim == 0:
            return global_time.view(1).expand(batch_size)
        if global_time.ndim == 1:
            if global_time.shape[0] == 1:
                return global_time.expand(batch_size)
            if global_time.shape[0] == batch_size:
                return global_time
        if global_time.ndim == 2 and global_time.shape == (batch_size, 1):
            return global_time[:, 0]
        raise ValueError(
            "`global_time` must be scalar or [B] matching the token batch, "
            f"got shape {tuple(global_time.shape)} for batch_size={batch_size}."
        )

    def _offsets_to_video_ttt_global_time(self, offsets: torch.Tensor | int | float) -> torch.Tensor:
        if not isinstance(offsets, torch.Tensor):
            offsets = torch.as_tensor(offsets)
        return torch.div(offsets.to(dtype=torch.long), int(self.video_ttt_time_stride), rounding_mode="floor")

    def _build_video_ttt_positions(
        self,
        *,
        meta: dict[str, Any],
        batch_size: int,
        device: torch.device,
        global_time: Optional[torch.Tensor | int | float],
        prefix_len: int = 0,
    ) -> dict[str, torch.Tensor]:
        grid_size = meta.get("grid_size")
        if grid_size is None or len(grid_size) != 3:
            raise ValueError(f"Video TTT positions require `meta['grid_size']=(f,h,w)`, got {grid_size!r}.")
        f, h, w = (int(grid_size[0]), int(grid_size[1]), int(grid_size[2]))
        if f <= 0 or h <= 0 or w <= 0:
            raise ValueError(f"Invalid video TTT grid size {(f, h, w)}.")

        base_time = self._coerce_video_ttt_global_time(
            global_time,
            batch_size=batch_size,
            device=device,
        )
        frame_time = base_time.view(batch_size, 1, 1, 1) + torch.arange(
            f,
            device=device,
            dtype=torch.float32,
        ).view(1, f, 1, 1)
        video_time = frame_time.expand(batch_size, f, h, w).reshape(batch_size, f * h * w)
        video_height = torch.arange(h, device=device, dtype=torch.float32).view(1, 1, h, 1)
        video_height = video_height.expand(batch_size, f, h, w).reshape(batch_size, f * h * w)
        video_width = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, 1, w)
        video_width = video_width.expand(batch_size, f, h, w).reshape(batch_size, f * h * w)

        if prefix_len > 0:
            prefix_time = torch.zeros((batch_size, prefix_len), device=device, dtype=torch.float32)
            state_tokens = min(int(self.video_ttt_state_token_count), int(prefix_len))
            if state_tokens > 0:
                prefix_time[:, -state_tokens:] = base_time.view(batch_size, 1)
            zeros = torch.zeros((batch_size, prefix_len), device=device, dtype=torch.float32)
            video_time = torch.cat([prefix_time, video_time], dim=1)
            video_height = torch.cat([zeros, video_height], dim=1)
            video_width = torch.cat([zeros, video_width], dim=1)

        return {
            "time": video_time,
            "height": video_height,
            "width": video_width,
        }

    def _apply_video_ttt_observation(
        self,
        video_tokens: torch.Tensor,
        *,
        state: Optional[dict[str, torch.Tensor]],
        persist_state: bool,
        update: bool = True,
        update_tokens: Optional[torch.Tensor] = None,
        positions: Optional[Any] = None,
        update_positions: Optional[Any] = None,
        layer_idx: Optional[int] = None,
        compute_loss: bool = True,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        adapter = self._get_video_ttt_adapter()
        if adapter is None:
            return video_tokens, state, None

        video_tokens, payload_positions = self._unpack_video_ttt_payload(video_tokens)
        if payload_positions is not None:
            positions = payload_positions
        update_tokens, payload_update_positions = self._unpack_video_ttt_payload(update_tokens)
        if payload_update_positions is not None:
            update_positions = payload_update_positions

        if isinstance(video_tokens, (list, tuple)):
            token_layers = list(video_tokens)
            if len(token_layers) <= 0:
                raise ValueError("`video_tokens` sequence must contain at least one tensor.")
            if isinstance(update_tokens, (list, tuple)):
                update_layers = list(update_tokens)
            else:
                update_layers = token_layers if update_tokens is None else [update_tokens] * len(token_layers)
            if isinstance(positions, (list, tuple)):
                position_layers = list(positions)
            else:
                position_layers = [positions] * len(token_layers)
            if isinstance(update_positions, (list, tuple)):
                update_position_layers = list(update_positions)
            else:
                update_position_layers = position_layers if update_positions is None else [update_positions] * len(token_layers)
            num_layers = min(len(token_layers), adapter.num_layers)
            if len(update_layers) < num_layers:
                raise ValueError(
                    f"`update_tokens` sequence must have at least {num_layers} entries, got {len(update_layers)}."
                )
            if len(position_layers) < num_layers:
                raise ValueError(
                    f"`positions` sequence must have at least {num_layers} entries, got {len(position_layers)}."
                )
            if len(update_position_layers) < num_layers:
                raise ValueError(
                    "`update_positions` sequence must have at least "
                    f"{num_layers} entries, got {len(update_position_layers)}."
                )
            adapted_tokens = token_layers[num_layers - 1]
            new_state = state
            losses = []
            for current_layer_idx in range(num_layers):
                adapted_tokens, new_state, ttt_loss = adapter(
                    token_layers[current_layer_idx],
                    state=new_state,
                    update=update,
                    update_tokens=update_layers[current_layer_idx],
                    positions=position_layers[current_layer_idx],
                    update_positions=update_position_layers[current_layer_idx],
                    layer_idx=current_layer_idx,
                    compute_loss=compute_loss,
                )
                if ttt_loss is not None:
                    losses.append(ttt_loss)
            ttt_loss = None if len(losses) == 0 else torch.stack(losses).mean()
            if persist_state:
                self._video_ttt_inference_state = adapter.detach_state(new_state)
            return adapted_tokens, new_state, ttt_loss

        layer_indices = range(adapter.num_layers) if layer_idx is None else [int(layer_idx)]
        adapted_tokens = video_tokens
        new_state = state
        losses = []
        update_token_layers = list(update_tokens) if isinstance(update_tokens, (list, tuple)) else None
        position_layers = list(positions) if isinstance(positions, (list, tuple)) else None
        update_position_layers = list(update_positions) if isinstance(update_positions, (list, tuple)) else None
        for current_layer_idx in layer_indices:
            if update_token_layers is not None:
                if current_layer_idx >= len(update_token_layers):
                    raise ValueError(
                        f"`update_tokens` sequence has {len(update_token_layers)} entries, "
                        f"but layer {current_layer_idx} was requested."
                    )
                update_tokens_for_layer = update_token_layers[current_layer_idx]
            else:
                update_tokens_for_layer = update_tokens
            if position_layers is not None:
                if current_layer_idx >= len(position_layers):
                    raise ValueError(
                        f"`positions` sequence has {len(position_layers)} entries, "
                        f"but layer {current_layer_idx} was requested."
                    )
                positions_for_layer = position_layers[current_layer_idx]
            else:
                positions_for_layer = positions
            if update_position_layers is not None:
                if current_layer_idx >= len(update_position_layers):
                    raise ValueError(
                        f"`update_positions` sequence has {len(update_position_layers)} entries, "
                        f"but layer {current_layer_idx} was requested."
                    )
                update_positions_for_layer = update_position_layers[current_layer_idx]
            else:
                update_positions_for_layer = update_positions
            adapted_tokens, new_state, ttt_loss = adapter(
                adapted_tokens,
                state=new_state,
                update=update,
                update_tokens=update_tokens_for_layer,
                positions=positions_for_layer,
                update_positions=update_positions_for_layer,
                layer_idx=current_layer_idx,
                compute_loss=compute_loss,
            )
            if ttt_loss is not None:
                losses.append(ttt_loss)
        ttt_loss = None if len(losses) == 0 else torch.stack(losses).mean()
        if persist_state:
            self._video_ttt_inference_state = adapter.detach_state(new_state)
        return adapted_tokens, new_state, ttt_loss

    def _update_video_ttt_observation_inline(
        self,
        *,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        state: Optional[dict[str, torch.Tensor]],
        persist_state: bool,
        global_time: Optional[torch.Tensor | int | float] = None,
        update: bool = True,
        compute_loss: bool = True,
    ) -> tuple[Optional[dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        adapter = self._get_video_ttt_adapter()
        if adapter is None:
            return state, None

        timestep_video_obs = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        obs_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video_obs,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        prefix_tokens = obs_pre["context"] if self.video_ttt_include_context_tokens else None
        prefix_len = 0 if prefix_tokens is None else int(prefix_tokens.shape[1])
        positions = self._build_video_ttt_positions(
            meta=obs_pre["meta"],
            batch_size=int(obs_pre["tokens"].shape[0]),
            device=obs_pre["tokens"].device,
            global_time=global_time,
            prefix_len=prefix_len,
        )

        def with_prefix(tokens: torch.Tensor) -> torch.Tensor:
            if prefix_tokens is None:
                return tokens
            return torch.cat([prefix_tokens.to(device=tokens.device, dtype=tokens.dtype), tokens], dim=1)

        selected_layers = tuple(getattr(self, "video_ttt_layer_indices", ()))
        if len(selected_layers) <= 0:
            observation_tokens = {
                "tokens": with_prefix(obs_pre["tokens"]),
                "positions": positions,
            }
            _, new_state, ttt_loss = self._apply_video_ttt_observation(
                observation_tokens,
                state=state,
                persist_state=persist_state,
                update=update,
                update_tokens=observation_tokens,
                compute_loss=compute_loss,
            )
            return new_state, ttt_loss

        selected_layers = selected_layers[: int(adapter.num_layers)]
        max_selected_layer = max(selected_layers)
        x_tokens = obs_pre["tokens"]
        self_attn_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=int(x_tokens.shape[1]),
            video_tokens_per_frame=int(obs_pre["meta"]["tokens_per_frame"]),
            device=x_tokens.device,
        )
        new_state = state
        losses = []
        for block_idx in range(max_selected_layer + 1):
            block = self.video_expert.blocks[block_idx]
            if self.video_ttt_train_backbone:
                x_tokens = block(
                    x_tokens,
                    obs_pre["context"],
                    obs_pre["t_mod"],
                    obs_pre["freqs"],
                    context_mask=obs_pre["context_mask"],
                    self_attn_mask=self_attn_mask,
                )
            else:
                with torch.no_grad():
                    x_tokens = block(
                        x_tokens,
                        obs_pre["context"],
                        obs_pre["t_mod"],
                        obs_pre["freqs"],
                        context_mask=obs_pre["context_mask"],
                        self_attn_mask=self_attn_mask,
                    )
            slot_idx = self.video_ttt_layer_to_slot.get(int(block_idx))
            if slot_idx is None:
                continue
            adapted_tokens, new_state, ttt_loss = adapter(
                with_prefix(x_tokens),
                state=new_state,
                update=update,
                update_tokens=None,
                positions=positions,
                layer_idx=int(slot_idx),
                compute_loss=compute_loss,
            )
            if ttt_loss is not None:
                losses.append(ttt_loss)
            x_tokens = adapted_tokens[:, prefix_len:].contiguous() if prefix_len > 0 else adapted_tokens

        if persist_state:
            self._video_ttt_inference_state = adapter.detach_state(new_state)
        ttt_loss = None if len(losses) == 0 else torch.stack(losses).mean()
        return new_state, ttt_loss

    def _training_loss_video_ttt_switch_chunk_group(
        self,
        *,
        state: Optional[dict[str, torch.Tensor]],
        frames: torch.Tensor,
        proprios: torch.Tensor,
        mask: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        tiled: bool = False,
        global_times: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[dict[str, torch.Tensor]], torch.Tensor, int]:
        """Batched observe prefill with serial-equivalent per-frame TTT updates."""
        adapter = self._get_video_ttt_adapter()
        if adapter is None or not self.video_ttt_uses_layer_hook:
            return state, frames.new_tensor(0.0, dtype=torch.float32), 0
        if self.video_ttt_train_backbone:
            raise ValueError("Grouped observe prefill is only enabled for frozen video backbone training.")
        if frames.ndim != 5:
            raise ValueError(f"`frames` must be [B,G,3,H,W], got {tuple(frames.shape)}.")
        if proprios.ndim != 3:
            raise ValueError(f"`proprios` must be [B,G,D], got {tuple(proprios.shape)}.")
        if mask.ndim != 2 or mask.shape[:2] != frames.shape[:2]:
            raise ValueError(f"`mask` must be [B,G] matching frames, got {tuple(mask.shape)}.")

        batch_size, group_size = int(frames.shape[0]), int(frames.shape[1])
        mask = mask.to(device=frames.device, dtype=torch.bool)
        if not bool(mask.any().item()):
            return state, frames.new_tensor(0.0, dtype=torch.float32), 0

        pair_sample_indices = []
        pair_step_indices = []
        step_pair_indices: list[torch.Tensor] = []
        for step_idx in range(group_size):
            valid_indices = torch.nonzero(mask[:, step_idx], as_tuple=False).flatten()
            step_pair_indices.append(
                torch.arange(
                    sum(int(x.numel()) for x in pair_sample_indices),
                    sum(int(x.numel()) for x in pair_sample_indices) + int(valid_indices.numel()),
                    device=frames.device,
                    dtype=torch.long,
                )
            )
            if valid_indices.numel() == 0:
                continue
            pair_sample_indices.append(valid_indices)
            pair_step_indices.append(torch.full_like(valid_indices, int(step_idx)))

        if not pair_sample_indices:
            return state, frames.new_tensor(0.0, dtype=torch.float32), 0
        pair_sample = torch.cat(pair_sample_indices, dim=0).to(device=frames.device)
        pair_step = torch.cat(pair_step_indices, dim=0).to(device=frames.device)
        flat_frames = frames[pair_sample, pair_step].unsqueeze(2)
        first_frame_latents = self._encode_video_latents(flat_frames, tiled=tiled)

        flat_context = context.index_select(0, pair_sample)
        flat_context_mask = context_mask.index_select(0, pair_sample)
        if self.proprio_encoder is not None:
            flat_context, flat_context_mask = self._append_proprio_to_context(
                context=flat_context,
                context_mask=flat_context_mask,
                proprio=proprios[pair_sample, pair_step],
            )

        flat_global_time = None
        if isinstance(global_times, torch.Tensor):
            flat_global_time = global_times.to(device=frames.device).index_select(0, pair_sample)
            flat_global_time = flat_global_time.gather(1, pair_step.view(-1, 1)).flatten()

        timestep_video_obs = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        obs_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video_obs,
            context=flat_context,
            context_mask=flat_context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        prefix_tokens = obs_pre["context"] if self.video_ttt_include_context_tokens else None
        prefix_len = 0 if prefix_tokens is None else int(prefix_tokens.shape[1])
        positions = self._build_video_ttt_positions(
            meta=obs_pre["meta"],
            batch_size=int(obs_pre["tokens"].shape[0]),
            device=obs_pre["tokens"].device,
            global_time=flat_global_time,
            prefix_len=prefix_len,
        )

        def with_prefix(tokens: torch.Tensor, pair_indices: torch.Tensor) -> torch.Tensor:
            selected = tokens.index_select(0, pair_indices)
            if prefix_tokens is None:
                return selected
            selected_prefix = prefix_tokens.index_select(0, pair_indices).to(
                device=selected.device,
                dtype=selected.dtype,
            )
            return torch.cat([selected_prefix, selected], dim=1)

        def select_positions(pair_indices: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                key: value.index_select(0, pair_indices.to(device=value.device))
                for key, value in positions.items()
            }

        selected_layers = tuple(getattr(self, "video_ttt_layer_indices", ()))
        selected_layers = selected_layers[: int(adapter.num_layers)]
        if len(selected_layers) <= 0:
            return state, frames.new_tensor(0.0, dtype=torch.float32), 0

        max_selected_layer = max(selected_layers)
        x_tokens = obs_pre["tokens"]
        self_attn_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=int(x_tokens.shape[1]),
            video_tokens_per_frame=int(obs_pre["meta"]["tokens_per_frame"]),
            device=x_tokens.device,
        )
        loss_sums: list[Optional[torch.Tensor]] = [None for _ in range(group_size)]
        loss_layer_counts = [0 for _ in range(group_size)]
        valid_counts = [int(mask[:, idx].sum().item()) for idx in range(group_size)]

        for block_idx in range(max_selected_layer + 1):
            block = self.video_expert.blocks[block_idx]
            with torch.no_grad():
                x_tokens = block(
                    x_tokens,
                    obs_pre["context"],
                    obs_pre["t_mod"],
                    obs_pre["freqs"],
                    context_mask=obs_pre["context_mask"],
                    self_attn_mask=self_attn_mask,
                )
            slot_idx = self.video_ttt_layer_to_slot.get(int(block_idx))
            if slot_idx is None:
                continue

            for step_idx, pair_indices in enumerate(step_pair_indices):
                if pair_indices.numel() == 0:
                    continue
                pair_indices = pair_indices.to(device=x_tokens.device)
                valid_indices_device = pair_sample.index_select(0, pair_indices).to(device=self.device)
                sub_state = self._select_video_ttt_state(state, valid_indices_device)
                adapted_tokens, sub_state, ttt_loss = adapter(
                    with_prefix(x_tokens, pair_indices),
                    state=sub_state,
                    update=True,
                    update_tokens=None,
                    positions=select_positions(pair_indices),
                    layer_idx=int(slot_idx),
                    compute_loss=True,
                )
                suffix = adapted_tokens[:, prefix_len:].contiguous() if prefix_len > 0 else adapted_tokens
                x_tokens = x_tokens.index_copy(0, pair_indices, suffix).detach()
                state = self._scatter_video_ttt_state(state, sub_state, valid_indices_device)
                if ttt_loss is not None:
                    loss_sums[step_idx] = ttt_loss if loss_sums[step_idx] is None else loss_sums[step_idx] + ttt_loss
                    loss_layer_counts[step_idx] += 1

        weighted_loss_sum = frames.new_tensor(0.0, dtype=torch.float32)
        loss_count = 0
        for step_idx, loss_sum in enumerate(loss_sums):
            if loss_sum is None or loss_layer_counts[step_idx] <= 0:
                continue
            weighted_loss_sum = weighted_loss_sum + (loss_sum / float(loss_layer_counts[step_idx])) * float(
                valid_counts[step_idx]
            )
            loss_count += valid_counts[step_idx]
        return state, weighted_loss_sum, int(loss_count)

    def _build_video_ttt_layer_hook(
        self,
        *,
        state: Optional[dict[str, torch.Tensor]] = None,
        positions: Optional[dict[str, torch.Tensor]] = None,
    ) -> Optional[Callable[[int, torch.Tensor], torch.Tensor]]:
        adapter = self._get_video_ttt_adapter()
        if adapter is None:
            return None
        hook_state = self._video_ttt_inference_state if state is None else state

        def hook(layer_idx: int, tokens: torch.Tensor) -> torch.Tensor:
            if not self.video_ttt_uses_layer_hook:
                return tokens
            slot_idx = self.video_ttt_layer_to_slot.get(int(layer_idx))
            if slot_idx is None:
                return tokens
            adapted_tokens, _, _ = self._apply_video_ttt_observation(
                tokens,
                state=hook_state,
                persist_state=False,
                update=False,
                update_tokens=None,
                positions=positions,
                layer_idx=int(slot_idx),
                compute_loss=False,
            )
            return adapted_tokens

        return hook

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        observation_context = sample.get("observation_context", None)
        observation_context_mask = sample.get("observation_context_mask", None)
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if observation_context is not None or observation_context_mask is not None:
            if observation_context is None or observation_context_mask is None:
                raise ValueError("`observation_context` and `observation_context_mask` must be provided together.")
            if observation_context.ndim != 3 or observation_context_mask.ndim != 2:
                raise ValueError(
                    "`observation_context/observation_context_mask` must be [B,L,D]/[B,L], "
                    f"got {tuple(observation_context.shape)} and {tuple(observation_context_mask.shape)}"
                )
            if observation_context.shape[0] != batch_size or observation_context_mask.shape[0] != batch_size:
                raise ValueError(
                    "`observation_context` batch size must match video batch size, "
                    f"got {observation_context.shape[0]} and {batch_size}."
                )
            observation_context = observation_context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            observation_context_mask = observation_context_mask.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
            if observation_context is not None and observation_context_mask is not None:
                observation_context, observation_context_mask = self._append_proprio_to_context(
                    context=observation_context,
                    context_mask=observation_context_mask,
                    proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
                )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        restart_context = sample.get("restart_context", None)
        restart_context_mask = sample.get("restart_context_mask", None)
        if restart_context is not None or restart_context_mask is not None:
            if restart_context is None or restart_context_mask is None:
                raise ValueError("`restart_context` and `restart_context_mask` must be provided together.")
            if restart_context.ndim != 3 or restart_context_mask.ndim != 2:
                raise ValueError(
                    "`restart_context/restart_context_mask` must be [B,L,D]/[B,L], "
                    f"got {tuple(restart_context.shape)} and {tuple(restart_context_mask.shape)}"
                )
            if restart_context.shape[0] != batch_size or restart_context_mask.shape[0] != batch_size:
                raise ValueError(
                    "`restart_context` batch size must match video batch size, "
                    f"got {restart_context.shape[0]} and {batch_size}."
                )
            restart_context = restart_context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            restart_context_mask = restart_context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            if self.proprio_encoder is not None:
                zero_proprio = torch.zeros(
                    (batch_size, self.proprio_dim),
                    device=self.device,
                    dtype=self.torch_dtype,
                )
                restart_context, restart_context_mask = self._append_proprio_to_context(
                    context=restart_context,
                    context_mask=restart_context_mask,
                    proprio=zero_proprio,
                )

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        ttt_global_time = sample.get("ttt_global_time", None)
        if ttt_global_time is None and "ttt_offset" in sample:
            ttt_global_time = self._offsets_to_video_ttt_global_time(sample["ttt_offset"])
        if isinstance(ttt_global_time, torch.Tensor):
            ttt_global_time = ttt_global_time.to(device=self.device, dtype=torch.long, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "observation_context": observation_context,
            "observation_context_mask": observation_context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "restart_context": restart_context,
            "restart_context_mask": restart_context_mask,
            "video_height": height,
            "video_width": width,
            "ttt_global_time": ttt_global_time,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _build_video_ttt_observation_tokens(
        self,
        *,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        global_time: Optional[torch.Tensor | int | float] = None,
    ) -> dict[str, Any]:
        timestep_video_obs = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        obs_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video_obs,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        prefix_tokens = obs_pre["context"] if self.video_ttt_include_context_tokens else None
        prefix_len = 0 if prefix_tokens is None else int(prefix_tokens.shape[1])
        positions = self._build_video_ttt_positions(
            meta=obs_pre["meta"],
            batch_size=int(obs_pre["tokens"].shape[0]),
            device=obs_pre["tokens"].device,
            global_time=global_time,
            prefix_len=prefix_len,
        )

        def add_prefix(tokens: torch.Tensor) -> torch.Tensor:
            if prefix_tokens is None:
                return tokens
            return torch.cat([prefix_tokens.to(device=tokens.device, dtype=tokens.dtype), tokens], dim=1)

        selected_layers = tuple(getattr(self, "video_ttt_layer_indices", ()))
        if len(selected_layers) <= 0:
            return {
                "tokens": add_prefix(obs_pre["tokens"]),
                "positions": positions,
            }
        adapter = self._get_video_ttt_adapter()
        if adapter is None:
            return {
                "tokens": add_prefix(obs_pre["tokens"]),
                "positions": positions,
            }
        selected_layers = selected_layers[: int(adapter.num_layers)]
        max_selected_layer = max(selected_layers)

        x_tokens = obs_pre["tokens"]
        self_attn_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=int(x_tokens.shape[1]),
            video_tokens_per_frame=int(obs_pre["meta"]["tokens_per_frame"]),
            device=x_tokens.device,
        )
        layer_tokens: list[Optional[torch.Tensor]] = [None] * int(adapter.num_layers)
        for block_idx in range(max_selected_layer + 1):
            block = self.video_expert.blocks[block_idx]
            if self.video_ttt_train_backbone:
                x_tokens = block(
                    x_tokens,
                    obs_pre["context"],
                    obs_pre["t_mod"],
                    obs_pre["freqs"],
                    context_mask=obs_pre["context_mask"],
                    self_attn_mask=self_attn_mask,
                )
            else:
                with torch.no_grad():
                    x_tokens = block(
                        x_tokens,
                        obs_pre["context"],
                        obs_pre["t_mod"],
                        obs_pre["freqs"],
                        context_mask=obs_pre["context_mask"],
                        self_attn_mask=self_attn_mask,
                    )
            slot_idx = self.video_ttt_layer_to_slot.get(int(block_idx))
            if slot_idx is not None and slot_idx < len(layer_tokens):
                layer_tokens[slot_idx] = add_prefix(x_tokens)
        if any(token is None for token in layer_tokens):
            raise RuntimeError(f"Failed to collect all video TTT layer tokens for layers {selected_layers}.")
        token_layers = [token for token in layer_tokens if token is not None]
        return {
            "tokens": token_layers,
            "positions": [positions] * len(token_layers),
        }

    def _training_loss_video_ttt_one_chunk(
        self,
        inputs: dict[str, Any],
        *,
        state: Optional[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[dict[str, torch.Tensor]]]:
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        observation_context = inputs.get("observation_context")
        observation_context_mask = inputs.get("observation_context_mask")
        if observation_context is None:
            observation_context = context
        if observation_context_mask is None:
            observation_context_mask = context_mask
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        first_frame_latents = inputs["first_frame_latents"]
        if first_frame_latents is None:
            first_frame_latents = input_latents[:, :, 0:1]
        ttt_global_time = inputs.get("ttt_global_time", None)

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        video_ttt_positions = self._build_video_ttt_positions(
            meta=video_pre["meta"],
            batch_size=int(video_pre["tokens"].shape[0]),
            device=video_pre["tokens"].device,
            global_time=ttt_global_time,
            prefix_len=0,
        )
        if self.video_ttt_uses_layer_hook:
            state, ttt_loss = self._update_video_ttt_observation_inline(
                first_frame_latents=first_frame_latents,
                context=observation_context,
                context_mask=observation_context_mask,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                state=state,
                persist_state=False,
                update=True,
                global_time=ttt_global_time,
            )
            video_tokens = video_pre["tokens"]
            video_ttt_hook = self._build_video_ttt_layer_hook(state=state, positions=video_ttt_positions)
        else:
            observation_tokens = self._build_video_ttt_observation_tokens(
                first_frame_latents=first_frame_latents,
                context=observation_context,
                context_mask=observation_context_mask,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                global_time=ttt_global_time,
            )
            video_tokens, state, ttt_loss = self._apply_video_ttt_observation(
                video_pre["tokens"],
                state=state,
                persist_state=False,
                update=True,
                update_tokens=observation_tokens,
                positions=video_ttt_positions,
            )
            video_ttt_hook = None

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
            video_ttt_hook=video_ttt_hook,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        target_video_for_loss = target_video
        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video_for_loss = target_video_for_loss[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video_for_loss,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        loss_action = (action_loss_per_sample * action_weight).mean()
        loss_outer = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_outer, loss_video, loss_action, ttt_loss, state

    def _training_loss_video_ttt_switch_chunk(
        self,
        *,
        state: Optional[dict[str, torch.Tensor]],
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        global_time: Optional[torch.Tensor | int | float] = None,
    ) -> tuple[Optional[dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        if self.video_ttt_uses_layer_hook:
            return self._update_video_ttt_observation_inline(
                first_frame_latents=first_frame_latents,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
                state=state,
                persist_state=False,
                update=True,
                global_time=global_time,
            )
        switch_tokens = self._build_video_ttt_observation_tokens(
            first_frame_latents=first_frame_latents,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            global_time=global_time,
        )
        _, state, ttt_switch_loss = self._apply_video_ttt_observation(
            switch_tokens,
            state=state,
            persist_state=False,
            update=True,
            update_tokens=switch_tokens,
        )
        return state, ttt_switch_loss

    @staticmethod
    def _slice_ttt_try_sample(sample: dict[str, Any], try_idx: int) -> dict[str, Any]:
        sliced: dict[str, Any] = {}
        try_keys = {
            "video",
            "action",
            "proprio",
            "context",
            "context_mask",
            "observation_context",
            "observation_context_mask",
            "image_is_pad",
            "action_is_pad",
            "proprio_is_pad",
        }
        for key, value in sample.items():
            if key in try_keys and isinstance(value, torch.Tensor) and value.ndim >= 2:
                sliced[key] = value[:, try_idx]
            elif key in {"restart_context", "restart_context_mask"}:
                sliced[key] = value
        return sliced

    @staticmethod
    def _select_video_ttt_state(
        state: Optional[dict[str, torch.Tensor]],
        indices: torch.Tensor,
    ) -> Optional[dict[str, torch.Tensor]]:
        if state is None:
            return None
        return {key: value.index_select(0, indices.to(device=value.device)) for key, value in state.items()}

    @staticmethod
    def _scatter_video_ttt_state(
        state: Optional[dict[str, torch.Tensor]],
        updated_state: Optional[dict[str, torch.Tensor]],
        indices: torch.Tensor,
    ) -> Optional[dict[str, torch.Tensor]]:
        if updated_state is None:
            return state
        if state is None:
            return updated_state
        scattered = {}
        for key, value in state.items():
            index = indices.to(device=value.device)
            if index.numel() == value.shape[0]:
                scattered[key] = updated_state[key]
            else:
                scattered[key] = value.index_copy(0, index, updated_state[key])
        return scattered

    @staticmethod
    def _slice_ttt_execution_chunk_sample(
        sample: dict[str, Any],
        chunk_idx: int,
        indices: torch.Tensor,
    ) -> dict[str, Any]:
        sliced: dict[str, Any] = {}
        base_keys = {
            "context",
            "context_mask",
            "observation_context",
            "observation_context_mask",
            "restart_context",
            "restart_context_mask",
        }
        for key, value in sample.items():
            if key in base_keys and isinstance(value, torch.Tensor):
                sliced[key] = value.index_select(0, indices)

        sequence_keys = {
            "video": "ttt_execution_video",
            "action": "ttt_execution_action",
            "proprio": "ttt_execution_proprio",
            "image_is_pad": "ttt_execution_image_is_pad",
            "action_is_pad": "ttt_execution_action_is_pad",
            "proprio_is_pad": "ttt_execution_proprio_is_pad",
        }
        for out_key, seq_key in sequence_keys.items():
            value = sample.get(seq_key)
            if isinstance(value, torch.Tensor):
                sliced[out_key] = value.index_select(0, indices)[:, chunk_idx]
        meta_sequence_keys = {
            "ttt_offset": "ttt_execution_offsets",
            "ttt_global_time": "ttt_execution_times",
        }
        for out_key, seq_key in meta_sequence_keys.items():
            value = sample.get(seq_key)
            if isinstance(value, torch.Tensor):
                sliced[out_key] = value.index_select(0, indices)[:, chunk_idx]
        return sliced

    @staticmethod
    def _slice_ttt_try_sequence_chunk_sample(
        sample: dict[str, Any],
        try_idx: int,
        chunk_idx: int,
        indices: torch.Tensor,
    ) -> dict[str, Any]:
        sliced: dict[str, Any] = {}
        for key in ("context", "context_mask", "observation_context", "observation_context_mask"):
            value = sample.get(key)
            if isinstance(value, torch.Tensor):
                selected = value.index_select(0, indices)
                sliced[key] = selected[:, try_idx] if selected.ndim >= 3 else selected
        for key in ("restart_context", "restart_context_mask"):
            value = sample.get(key)
            if isinstance(value, torch.Tensor):
                sliced[key] = value.index_select(0, indices)

        sequence_keys = {
            "video",
            "action",
            "proprio",
            "image_is_pad",
            "action_is_pad",
            "proprio_is_pad",
        }
        for key in sequence_keys:
            value = sample.get(key)
            if isinstance(value, torch.Tensor):
                sliced[key] = value.index_select(0, indices)[:, try_idx, chunk_idx]
        return sliced

    def _training_loss_video_ttt_repeat_attempt_sequence(
        self,
        sample: dict[str, Any],
        tiled: bool = False,
        backward_fn: Optional[Callable[[torch.Tensor], None]] = None,
    ):
        video = sample["video"]
        if video.ndim != 7:
            raise ValueError(f"sequential repeat-attempt TTT expects video [B,R,N,3,T,H,W], got {tuple(video.shape)}.")
        batch_size = int(video.shape[0])
        num_tries = int(video.shape[1])
        num_chunks = int(video.shape[2])
        use_switch_chunks = bool(self.video_ttt_switch_chunks and num_tries > 1)
        chunk_mask = sample.get("ttt_try_chunk_mask")
        if chunk_mask is None:
            chunk_mask = torch.ones((batch_size, num_tries, num_chunks), dtype=torch.bool)
        if chunk_mask.ndim != 3 or tuple(chunk_mask.shape[:3]) != (batch_size, num_tries, num_chunks):
            raise ValueError(
                "`ttt_try_chunk_mask` must be [B,R,N] matching sequential repeat video, "
                f"got {tuple(chunk_mask.shape)} vs {(batch_size, num_tries, num_chunks)}."
            )
        chunk_mask = chunk_mask.bool()

        streaming_backward = backward_fn is not None
        valid_count_total_expected = int(chunk_mask.sum().item())
        if valid_count_total_expected <= 0:
            raise ValueError("Sequential repeat-attempt TTT found no valid chunks.")
        switch_count_expected = batch_size * (num_tries - 1) if use_switch_chunks else 0
        ttt_loss_count_expected = max(valid_count_total_expected + switch_count_expected, 1)

        state = None
        loss_total = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_video_total = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_action_total = torch.zeros((), device=self.device, dtype=torch.float32)
        ttt_loss_total = torch.zeros((), device=self.device, dtype=torch.float32)
        ttt_switch_loss_total = torch.zeros((), device=self.device, dtype=torch.float32)
        ttt_loss_count = 0
        ttt_switch_count = 0
        valid_count_total = 0
        black_first_frame_latents = None
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        for try_idx in range(num_tries):
            if try_idx > 0 and use_switch_chunks:
                restart_context = sample.get("restart_context")
                restart_context_mask = sample.get("restart_context_mask")
                if restart_context is None or restart_context_mask is None:
                    raise ValueError("repeat-attempt TTT switch chunks require `restart_context`.")
                restart_context = restart_context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
                restart_context_mask = restart_context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
                if black_first_frame_latents is None:
                    black_video = torch.full(
                        (
                            batch_size,
                            3,
                            1,
                            int(video.shape[-2]),
                            int(video.shape[-1]),
                        ),
                        -1.0,
                        device=self.device,
                        dtype=self.torch_dtype,
                    )
                    black_first_frame_latents = self._encode_video_latents(black_video, tiled=tiled)
                state, ttt_switch_loss = self._training_loss_video_ttt_switch_chunk(
                    state=state,
                    first_frame_latents=black_first_frame_latents,
                    context=restart_context,
                    context_mask=restart_context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
                if ttt_switch_loss is not None:
                    if streaming_backward and self.loss_lambda_video_ttt != 0.0:
                        backward_fn(
                            self.loss_lambda_video_ttt
                            * ttt_switch_loss
                            * (float(batch_size) / float(ttt_loss_count_expected)),
                            retain_graph=True,
                        )
                    ttt_switch_loss_for_total = ttt_switch_loss.detach() if streaming_backward else ttt_switch_loss
                    ttt_switch_loss_total = ttt_switch_loss_total + ttt_switch_loss_for_total * float(batch_size)
                    ttt_loss_total = ttt_loss_total + ttt_switch_loss_for_total * float(batch_size)
                    ttt_switch_count += batch_size
                    ttt_loss_count += batch_size

            for chunk_idx in range(num_chunks):
                valid_indices = torch.nonzero(chunk_mask[:, try_idx, chunk_idx], as_tuple=False).flatten()
                if valid_indices.numel() == 0:
                    continue
                sub_sample = self._slice_ttt_try_sequence_chunk_sample(sample, try_idx, chunk_idx, valid_indices)
                sub_state = self._select_video_ttt_state(state, valid_indices.to(device=self.device))
                inputs = self.build_inputs(sub_sample, tiled=tiled)
                loss_outer, loss_video, loss_action, ttt_loss, sub_state = self._training_loss_video_ttt_one_chunk(
                    inputs,
                    state=sub_state,
                )
                valid_count = int(valid_indices.numel())
                if streaming_backward:
                    loss_for_backward = loss_outer * (
                        float(valid_count) / float(valid_count_total_expected)
                    )
                    if ttt_loss is not None and self.loss_lambda_video_ttt != 0.0:
                        loss_for_backward = loss_for_backward + self.loss_lambda_video_ttt * ttt_loss * (
                            float(valid_count) / float(ttt_loss_count_expected)
                        )
                    backward_fn(loss_for_backward, retain_graph=True)
                loss_outer_for_total = loss_outer.detach() if streaming_backward else loss_outer
                loss_video_for_total = loss_video.detach() if streaming_backward else loss_video
                loss_action_for_total = loss_action.detach() if streaming_backward else loss_action
                loss_total = loss_total + loss_outer_for_total * float(valid_count)
                loss_video_total = loss_video_total + loss_video_for_total * float(valid_count)
                loss_action_total = loss_action_total + loss_action_for_total * float(valid_count)
                valid_count_total += valid_count
                if ttt_loss is not None:
                    ttt_loss_for_total = ttt_loss.detach() if streaming_backward else ttt_loss
                    ttt_loss_total = ttt_loss_total + ttt_loss_for_total * float(valid_count)
                    ttt_loss_count += valid_count
                state = self._scatter_video_ttt_state(
                    state,
                    sub_state,
                    valid_indices.to(device=self.device),
                )

        if valid_count_total <= 0:
            raise ValueError("Sequential repeat-attempt TTT found no valid chunks.")
        loss_total = loss_total / float(valid_count_total)
        loss_video = loss_video_total / float(valid_count_total)
        loss_action = loss_action_total / float(valid_count_total)
        loss_video_ttt = ttt_loss_total / float(max(ttt_loss_count, 1))
        if self.loss_lambda_video_ttt != 0.0:
            loss_total = loss_total + self.loss_lambda_video_ttt * loss_video_ttt

        valid_chunks_per_try = chunk_mask.to(dtype=torch.float32).sum(dim=2)
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_video_ttt": float(loss_video_ttt.detach().item()),
            "video_ttt_num_tries": float(num_tries),
            "video_ttt_act_chunks": float(valid_chunks_per_try.detach().mean().item()),
            "video_ttt_total_act_chunks": float(valid_chunks_per_try.detach().sum(dim=1).mean().item()),
            "video_ttt_switch_chunks": float(1 if use_switch_chunks else 0),
        }
        if ttt_switch_count > 0:
            loss_dict["loss_video_ttt_switch"] = float(
                (ttt_switch_loss_total / float(ttt_switch_count)).detach().item()
            )
        if self.loss_lambda_video_ttt != 0.0:
            loss_dict["loss_video_ttt_weighted"] = self.loss_lambda_video_ttt * loss_dict["loss_video_ttt"]
        if streaming_backward:
            return loss_total.detach(), loss_dict, True
        return loss_total, loss_dict

    def training_loss_video_ttt_repeat_attempt(
        self,
        sample: dict[str, Any],
        tiled: bool = False,
        backward_fn: Optional[Callable[[torch.Tensor], None]] = None,
    ):
        video = sample["video"]
        if video.ndim == 7:
            return self._training_loss_video_ttt_repeat_attempt_sequence(sample, tiled=tiled, backward_fn=backward_fn)
        if backward_fn is not None:
            raise ValueError("Streaming backward is only supported for sequential repeat-attempt TTT video [B,R,N,3,T,H,W].")
        if video.ndim != 6:
            raise ValueError(f"repeat-attempt TTT expects video [B,R,3,T,H,W], got {tuple(video.shape)}.")
        num_tries = int(video.shape[1])
        use_switch_chunks = bool(self.video_ttt_switch_chunks and num_tries > 1)

        state = None
        loss_total = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_video_total = torch.zeros((), device=self.device, dtype=torch.float32)
        loss_action_total = torch.zeros((), device=self.device, dtype=torch.float32)
        ttt_loss_total = torch.zeros((), device=self.device, dtype=torch.float32)
        ttt_switch_loss_total = torch.zeros((), device=self.device, dtype=torch.float32)
        ttt_loss_count = 0
        ttt_switch_count = 0
        black_first_frame_latents = None

        for try_idx in range(num_tries):
            sub_sample = self._slice_ttt_try_sample(sample, try_idx)
            inputs = self.build_inputs(sub_sample, tiled=tiled)
            if try_idx > 0 and use_switch_chunks:
                restart_context = inputs["restart_context"]
                restart_context_mask = inputs["restart_context_mask"]
                if restart_context is None or restart_context_mask is None:
                    raise ValueError("repeat-attempt TTT switch chunks require `restart_context`.")
                if black_first_frame_latents is None:
                    black_video = torch.full(
                        (
                            inputs["input_latents"].shape[0],
                            3,
                            1,
                            int(inputs["video_height"]),
                            int(inputs["video_width"]),
                        ),
                        -1.0,
                        device=self.device,
                        dtype=self.torch_dtype,
                    )
                    black_first_frame_latents = self._encode_video_latents(black_video, tiled=tiled)
                state, ttt_switch_loss = self._training_loss_video_ttt_switch_chunk(
                    state=state,
                    first_frame_latents=black_first_frame_latents,
                    context=restart_context,
                    context_mask=restart_context_mask,
                    fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                )
                if ttt_switch_loss is not None:
                    ttt_switch_loss_total = ttt_switch_loss_total + ttt_switch_loss
                    ttt_loss_total = ttt_loss_total + ttt_switch_loss
                    ttt_switch_count += 1
                    ttt_loss_count += 1

            loss_outer, loss_video, loss_action, ttt_loss, state = self._training_loss_video_ttt_one_chunk(
                inputs,
                state=state,
            )
            loss_total = loss_total + loss_outer
            loss_video_total = loss_video_total + loss_video
            loss_action_total = loss_action_total + loss_action
            if ttt_loss is not None:
                ttt_loss_total = ttt_loss_total + ttt_loss
                ttt_loss_count += 1

        loss_video_avg = loss_video_total / float(num_tries)
        loss_action_avg = loss_action_total / float(num_tries)
        loss_total = loss_total / float(num_tries)
        loss_video_ttt = ttt_loss_total / float(max(ttt_loss_count, 1))
        if self.loss_lambda_video_ttt != 0.0:
            loss_total = loss_total + self.loss_lambda_video_ttt * loss_video_ttt

        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video_avg.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action_avg.detach().item()),
            "loss_video_ttt": float(loss_video_ttt.detach().item()),
            "video_ttt_num_tries": float(num_tries),
            "video_ttt_switch_chunks": float(1 if use_switch_chunks else 0),
        }
        if ttt_switch_count > 0:
            loss_dict["loss_video_ttt_switch"] = float(
                (ttt_switch_loss_total / float(ttt_switch_count)).detach().item()
            )
        if self.loss_lambda_video_ttt != 0.0:
            loss_dict["loss_video_ttt_weighted"] = self.loss_lambda_video_ttt * loss_dict["loss_video_ttt"]
        return loss_total, loss_dict

    def training_loss_video_ttt_observe_then_act(
        self,
        sample: dict[str, Any],
        tiled: bool = False,
        backward_fn: Optional[Callable[[torch.Tensor], None]] = None,
    ):
        if "ttt_warmup_video" not in sample or "ttt_warmup_proprio" not in sample:
            raise ValueError("observe-then-act TTT requires `ttt_warmup_video` and `ttt_warmup_proprio`.")

        warmup_video = sample["ttt_warmup_video"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        warmup_proprio = sample["ttt_warmup_proprio"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        if warmup_video.ndim != 5:
            raise ValueError(f"`ttt_warmup_video` must be [B,N,3,H,W], got {tuple(warmup_video.shape)}.")
        if warmup_video.shape[2] != 3:
            raise ValueError(f"`ttt_warmup_video` channel dim must be 3, got {warmup_video.shape[2]}.")
        warmup_mask = sample.get("ttt_warmup_mask")
        if isinstance(warmup_mask, torch.Tensor):
            warmup_mask = warmup_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            if warmup_mask.ndim != 2 or warmup_mask.shape[:2] != warmup_video.shape[:2]:
                raise ValueError(
                    "`ttt_warmup_mask` must be [B,N] matching `ttt_warmup_video`, "
                    f"got {tuple(warmup_mask.shape)} vs {tuple(warmup_video.shape[:2])}."
                )
        else:
            warmup_mask = torch.ones(
                warmup_video.shape[:2],
                device=self.device,
                dtype=torch.bool,
            )
        warmup_times = sample.get("ttt_warmup_times", None)
        if warmup_times is None and isinstance(sample.get("ttt_warmup_offsets"), torch.Tensor):
            warmup_times = self._offsets_to_video_ttt_global_time(sample["ttt_warmup_offsets"])
        if isinstance(warmup_times, torch.Tensor):
            warmup_times = warmup_times.to(device=self.device, dtype=torch.long, non_blocking=True)

        context = sample["context"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = sample["context_mask"].to(device=self.device, dtype=torch.bool, non_blocking=True)
        observation_context = sample.get("observation_context")
        observation_context_mask = sample.get("observation_context_mask")
        if isinstance(observation_context, torch.Tensor) and isinstance(observation_context_mask, torch.Tensor):
            observation_context = observation_context.to(
                device=self.device,
                dtype=self.torch_dtype,
                non_blocking=True,
            )
            observation_context_mask = observation_context_mask.to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )
        else:
            observation_context = context
            observation_context_mask = context_mask
        state = None
        ttt_loss_total = warmup_video.new_tensor(0.0, dtype=torch.float32)
        ttt_loss_count = 0
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        streaming_backward = backward_fn is not None
        observe_prefill_group_size = int(getattr(self, "video_ttt_observe_prefill_group_size", 1))
        execution_backward_group_size = int(getattr(self, "video_ttt_execution_backward_group_size", 1))
        use_grouped_observe_prefill = (
            observe_prefill_group_size > 1
            and self.video_ttt_uses_layer_hook
            and not self.video_ttt_train_backbone
        )
        outer_backward_group_count = 0
        outer_backward_chunk_count = 0

        execution_count_expected = int(warmup_video.shape[0])
        execution_mask_for_counts = sample.get("ttt_execution_mask")
        if isinstance(sample.get("ttt_execution_video"), torch.Tensor) and isinstance(execution_mask_for_counts, torch.Tensor):
            execution_count_expected = int(execution_mask_for_counts.bool().sum().item())
        if execution_count_expected <= 0:
            raise ValueError("Observe-then-act TTT found no valid execution chunks.")
        warmup_count_expected = int(warmup_mask.sum().item())
        if warmup_count_expected <= 0:
            raise ValueError("Observe-then-act TTT found no valid warmup chunks.")
        policy_update_count_expected = 0
        policy_update_mask_for_counts = sample.get("ttt_policy_update_mask")
        if isinstance(sample.get("ttt_policy_update_video"), torch.Tensor) and isinstance(
            policy_update_mask_for_counts,
            torch.Tensor,
        ):
            policy_update_count_expected = int(policy_update_mask_for_counts.bool().sum().item())
        ttt_loss_count_expected = max(
            warmup_count_expected + execution_count_expected + policy_update_count_expected,
            1,
        )

        if use_grouped_observe_prefill:
            chunk_idx = 0
            while chunk_idx < int(warmup_video.shape[1]):
                chunk_end = min(chunk_idx + observe_prefill_group_size, int(warmup_video.shape[1]))
                state, ttt_loss_sum, group_ttt_count = self._training_loss_video_ttt_switch_chunk_group(
                    state=state,
                    frames=warmup_video[:, chunk_idx:chunk_end],
                    proprios=warmup_proprio[:, chunk_idx:chunk_end],
                    mask=warmup_mask[:, chunk_idx:chunk_end],
                    context=observation_context,
                    context_mask=observation_context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    tiled=tiled,
                    global_times=warmup_times[:, chunk_idx:chunk_end] if isinstance(warmup_times, torch.Tensor) else None,
                )
                if group_ttt_count > 0:
                    if streaming_backward and self.loss_lambda_video_ttt != 0.0:
                        backward_fn(
                            self.loss_lambda_video_ttt * ttt_loss_sum / float(ttt_loss_count_expected),
                            retain_graph=True,
                        )
                    ttt_loss_for_total = ttt_loss_sum.detach() if streaming_backward else ttt_loss_sum
                    ttt_loss_total = ttt_loss_total + ttt_loss_for_total
                    ttt_loss_count += int(group_ttt_count)
                chunk_idx = chunk_end
        else:
            for chunk_idx in range(int(warmup_video.shape[1])):
                valid_indices = torch.nonzero(warmup_mask[:, chunk_idx], as_tuple=False).flatten()
                if valid_indices.numel() == 0:
                    continue
                valid_indices_device = valid_indices.to(device=self.device)
                frame = warmup_video.index_select(0, valid_indices_device)[:, chunk_idx].unsqueeze(2)
                first_frame_latents = self._encode_video_latents(frame, tiled=tiled)
                obs_context = observation_context.index_select(0, valid_indices_device)
                obs_context_mask = observation_context_mask.index_select(0, valid_indices_device)
                if self.proprio_encoder is not None:
                    obs_context, obs_context_mask = self._append_proprio_to_context(
                        context=obs_context,
                        context_mask=obs_context_mask,
                        proprio=warmup_proprio.index_select(0, valid_indices_device)[:, chunk_idx],
                    )
                sub_state = self._select_video_ttt_state(state, valid_indices_device)
                global_time = None
                if isinstance(warmup_times, torch.Tensor):
                    global_time = warmup_times.index_select(0, valid_indices_device)[:, chunk_idx]
                sub_state, ttt_loss = self._training_loss_video_ttt_switch_chunk(
                    state=sub_state,
                    first_frame_latents=first_frame_latents,
                    context=obs_context,
                    context_mask=obs_context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    global_time=global_time,
                )
                valid_count = int(valid_indices.numel())
                if ttt_loss is not None:
                    if streaming_backward and self.loss_lambda_video_ttt != 0.0:
                        backward_fn(
                            self.loss_lambda_video_ttt
                            * ttt_loss
                            * (float(valid_count) / float(ttt_loss_count_expected)),
                            retain_graph=True,
                        )
                    ttt_loss_for_total = ttt_loss.detach() if streaming_backward else ttt_loss
                    ttt_loss_total = ttt_loss_total + ttt_loss_for_total * float(valid_count)
                    ttt_loss_count += valid_count
                state = self._scatter_video_ttt_state(state, sub_state, valid_indices_device)

        if "ttt_execution_video" in sample:
            execution_video = sample["ttt_execution_video"]
            execution_mask = sample.get("ttt_execution_mask")
            if execution_video.ndim != 6:
                raise ValueError(
                    "`ttt_execution_video` must be [B,N,3,T,H,W], "
                    f"got {tuple(execution_video.shape)}."
                )
            if execution_mask is None:
                raise ValueError("`ttt_execution_mask` is required for sequential observe-then-act TTT.")
            if execution_mask.ndim != 2 or execution_mask.shape[:2] != execution_video.shape[:2]:
                raise ValueError(
                    "`ttt_execution_mask` must be [B,N] matching `ttt_execution_video`, "
                    f"got {tuple(execution_mask.shape)} vs {tuple(execution_video.shape[:2])}."
                )

            loss_total = warmup_video.new_tensor(0.0, dtype=torch.float32)
            loss_video_total = warmup_video.new_tensor(0.0, dtype=torch.float32)
            loss_action_total = warmup_video.new_tensor(0.0, dtype=torch.float32)
            valid_count_total = 0
            execution_mask = execution_mask.bool()
            policy_update_video = sample.get("ttt_policy_update_video")
            policy_update_proprio = sample.get("ttt_policy_update_proprio")
            policy_update_mask = sample.get("ttt_policy_update_mask")
            policy_update_times = sample.get("ttt_policy_update_times")
            if policy_update_times is None and isinstance(sample.get("ttt_policy_update_offsets"), torch.Tensor):
                policy_update_times = self._offsets_to_video_ttt_global_time(sample["ttt_policy_update_offsets"])
            updates_per_execution_chunk = 0
            raw_updates_per_chunk = sample.get("ttt_policy_updates_per_execution_chunk")
            if isinstance(raw_updates_per_chunk, torch.Tensor):
                updates_per_execution_chunk = int(raw_updates_per_chunk.flatten()[0].item())
            if isinstance(policy_update_video, torch.Tensor):
                policy_update_video = policy_update_video.to(
                    device=self.device,
                    dtype=self.torch_dtype,
                    non_blocking=True,
                )
            if isinstance(policy_update_proprio, torch.Tensor):
                policy_update_proprio = policy_update_proprio.to(
                    device=self.device,
                    dtype=self.torch_dtype,
                    non_blocking=True,
                )
            if isinstance(policy_update_mask, torch.Tensor):
                policy_update_mask = policy_update_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
            if isinstance(policy_update_times, torch.Tensor):
                policy_update_times = policy_update_times.to(device=self.device, dtype=torch.long, non_blocking=True)
            valid_count_total_expected = int(execution_mask.sum().item())
            if valid_count_total_expected <= 0:
                raise ValueError("Sequential observe-then-act TTT found no valid execution chunks.")
            pending_outer_backward_loss: Optional[torch.Tensor] = None
            pending_outer_backward_chunks = 0

            def flush_pending_outer_backward() -> None:
                nonlocal pending_outer_backward_loss
                nonlocal pending_outer_backward_chunks
                nonlocal outer_backward_group_count
                nonlocal outer_backward_chunk_count
                if pending_outer_backward_loss is None:
                    return
                if backward_fn is None:
                    raise RuntimeError("Cannot flush pending outer backward loss without `backward_fn`.")
                backward_fn(pending_outer_backward_loss, retain_graph=True)
                outer_backward_group_count += 1
                outer_backward_chunk_count += int(pending_outer_backward_chunks)
                pending_outer_backward_loss = None
                pending_outer_backward_chunks = 0

            for chunk_idx in range(int(execution_video.shape[1])):
                if (
                    chunk_idx > 0
                    and updates_per_execution_chunk > 0
                    and isinstance(policy_update_video, torch.Tensor)
                    and isinstance(policy_update_proprio, torch.Tensor)
                    and isinstance(policy_update_mask, torch.Tensor)
                ):
                    update_start = int((chunk_idx - 1) * updates_per_execution_chunk)
                    update_end = min(update_start + int(updates_per_execution_chunk), int(policy_update_video.shape[1]))
                    if use_grouped_observe_prefill:
                        group_start = update_start
                        while group_start < update_end:
                            group_end = min(group_start + observe_prefill_group_size, update_end)
                            state, ttt_loss_sum, group_ttt_count = self._training_loss_video_ttt_switch_chunk_group(
                                state=state,
                                frames=policy_update_video[:, group_start:group_end],
                                proprios=policy_update_proprio[:, group_start:group_end],
                                mask=policy_update_mask[:, group_start:group_end],
                                context=observation_context,
                                context_mask=observation_context_mask,
                                fuse_vae_embedding_in_latents=fuse_flag,
                                tiled=tiled,
                                global_times=(
                                    policy_update_times[:, group_start:group_end]
                                    if isinstance(policy_update_times, torch.Tensor)
                                    else None
                                ),
                            )
                            if group_ttt_count > 0:
                                if streaming_backward and self.loss_lambda_video_ttt != 0.0:
                                    backward_fn(
                                        self.loss_lambda_video_ttt * ttt_loss_sum / float(ttt_loss_count_expected),
                                        retain_graph=True,
                                    )
                                ttt_loss_for_total = ttt_loss_sum.detach() if streaming_backward else ttt_loss_sum
                                ttt_loss_total = ttt_loss_total + ttt_loss_for_total
                                ttt_loss_count += int(group_ttt_count)
                            group_start = group_end
                    else:
                        for update_idx in range(update_start, update_end):
                            update_indices = torch.nonzero(
                                policy_update_mask[:, update_idx],
                                as_tuple=False,
                            ).flatten()
                            if update_indices.numel() == 0:
                                continue
                            update_indices_device = update_indices.to(device=self.device)
                            frame = policy_update_video.index_select(0, update_indices_device)[:, update_idx].unsqueeze(2)
                            first_frame_latents = self._encode_video_latents(frame, tiled=tiled)
                            obs_context = observation_context.index_select(0, update_indices_device)
                            obs_context_mask = observation_context_mask.index_select(0, update_indices_device)
                            if self.proprio_encoder is not None:
                                obs_context, obs_context_mask = self._append_proprio_to_context(
                                    context=obs_context,
                                    context_mask=obs_context_mask,
                                    proprio=policy_update_proprio.index_select(0, update_indices_device)[:, update_idx],
                                )
                            sub_state = self._select_video_ttt_state(state, update_indices_device)
                            global_time = None
                            if isinstance(policy_update_times, torch.Tensor):
                                global_time = policy_update_times.index_select(0, update_indices_device)[:, update_idx]
                            sub_state, ttt_update_loss = self._training_loss_video_ttt_switch_chunk(
                                state=sub_state,
                                first_frame_latents=first_frame_latents,
                                context=obs_context,
                                context_mask=obs_context_mask,
                                fuse_vae_embedding_in_latents=fuse_flag,
                                global_time=global_time,
                            )
                            valid_update_count = int(update_indices.numel())
                            if ttt_update_loss is not None:
                                if streaming_backward and self.loss_lambda_video_ttt != 0.0:
                                    backward_fn(
                                        self.loss_lambda_video_ttt
                                        * ttt_update_loss
                                        * (float(valid_update_count) / float(ttt_loss_count_expected)),
                                        retain_graph=True,
                                    )
                                ttt_loss_for_total = ttt_update_loss.detach() if streaming_backward else ttt_update_loss
                                ttt_loss_total = ttt_loss_total + ttt_loss_for_total * float(valid_update_count)
                                ttt_loss_count += valid_update_count
                            state = self._scatter_video_ttt_state(state, sub_state, update_indices_device)

                valid_indices = torch.nonzero(execution_mask[:, chunk_idx], as_tuple=False).flatten()
                if valid_indices.numel() == 0:
                    continue
                sub_sample = self._slice_ttt_execution_chunk_sample(sample, chunk_idx, valid_indices)
                sub_state = self._select_video_ttt_state(state, valid_indices.to(device=self.device))
                inputs = self.build_inputs(sub_sample, tiled=tiled)
                loss_outer, loss_video, loss_action, ttt_loss, sub_state = self._training_loss_video_ttt_one_chunk(
                    inputs,
                    state=sub_state,
                )
                valid_count = int(valid_indices.numel())
                if streaming_backward:
                    loss_for_backward = loss_outer * (
                        float(valid_count) / float(valid_count_total_expected)
                    )
                    if ttt_loss is not None and self.loss_lambda_video_ttt != 0.0:
                        loss_for_backward = loss_for_backward + self.loss_lambda_video_ttt * ttt_loss * (
                            float(valid_count) / float(ttt_loss_count_expected)
                        )
                    pending_outer_backward_loss = (
                        loss_for_backward
                        if pending_outer_backward_loss is None
                        else pending_outer_backward_loss + loss_for_backward
                    )
                    pending_outer_backward_chunks += 1
                    if pending_outer_backward_chunks >= execution_backward_group_size:
                        flush_pending_outer_backward()
                loss_outer_for_total = loss_outer.detach() if streaming_backward else loss_outer
                loss_video_for_total = loss_video.detach() if streaming_backward else loss_video
                loss_action_for_total = loss_action.detach() if streaming_backward else loss_action
                loss_total = loss_total + loss_outer_for_total * float(valid_count)
                loss_video_total = loss_video_total + loss_video_for_total * float(valid_count)
                loss_action_total = loss_action_total + loss_action_for_total * float(valid_count)
                valid_count_total += valid_count
                if ttt_loss is not None:
                    ttt_loss_for_total = ttt_loss.detach() if streaming_backward else ttt_loss
                    ttt_loss_total = ttt_loss_total + ttt_loss_for_total * float(valid_count)
                    ttt_loss_count += valid_count
                state = self._scatter_video_ttt_state(
                    state,
                    sub_state,
                    valid_indices.to(device=self.device),
                )
            if streaming_backward:
                flush_pending_outer_backward()
            if valid_count_total <= 0:
                raise ValueError("Sequential observe-then-act TTT found no valid execution chunks.")
            loss_total = loss_total / float(valid_count_total)
            loss_video = loss_video_total / float(valid_count_total)
            loss_action = loss_action_total / float(valid_count_total)
            valid_chunks_per_sample = execution_mask.to(dtype=torch.float32).sum(dim=1)
        else:
            inputs = self.build_inputs(sample, tiled=tiled)
            loss_total, loss_video, loss_action, ttt_loss, state = self._training_loss_video_ttt_one_chunk(
                inputs,
                state=state,
            )
            if streaming_backward:
                loss_for_backward = loss_total
                if ttt_loss is not None and self.loss_lambda_video_ttt != 0.0:
                    loss_for_backward = loss_for_backward + self.loss_lambda_video_ttt * ttt_loss * (
                        float(warmup_video.shape[0]) / float(ttt_loss_count_expected)
                    )
                backward_fn(loss_for_backward, retain_graph=True)
                loss_total = loss_total.detach()
                loss_video = loss_video.detach()
                loss_action = loss_action.detach()
            if ttt_loss is not None:
                ttt_loss_for_total = ttt_loss.detach() if streaming_backward else ttt_loss
                ttt_loss_total = ttt_loss_total + ttt_loss_for_total * float(warmup_video.shape[0])
                ttt_loss_count += int(warmup_video.shape[0])
            valid_chunks_per_sample = warmup_video.new_ones((warmup_video.shape[0],), dtype=torch.float32)
        loss_video_ttt = ttt_loss_total / float(max(ttt_loss_count, 1))
        if self.loss_lambda_video_ttt != 0.0:
            loss_total = loss_total + self.loss_lambda_video_ttt * loss_video_ttt
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "loss_video_ttt": float(loss_video_ttt.detach().item()),
            "video_ttt_observe_chunks": float(warmup_mask.to(dtype=torch.float32).sum(dim=1).mean().item()),
            "video_ttt_act_chunks": float(valid_chunks_per_sample.detach().mean().item()),
            "video_ttt_policy_update_chunks": float(
                policy_update_mask_for_counts.bool().to(dtype=torch.float32).sum(dim=1).mean().item()
                if isinstance(policy_update_mask_for_counts, torch.Tensor)
                else 0.0
            ),
            "video_ttt_switch_chunks": 0.0,
            "video_ttt_execution_backward_group_size": float(execution_backward_group_size),
            "video_ttt_outer_backward_groups": float(outer_backward_group_count),
            "video_ttt_outer_backward_chunks": float(outer_backward_chunk_count),
        }
        loss_dict.update(self._video_ttt_gate_metrics())
        if self.loss_lambda_video_ttt != 0.0:
            loss_dict["loss_video_ttt_weighted"] = self.loss_lambda_video_ttt * loss_dict["loss_video_ttt"]
        if streaming_backward:
            return loss_total.detach(), loss_dict, True
        return loss_total, loss_dict

    def training_loss_video_ttt_observation(
        self,
        sample,
        tiled: bool = False,
        backward_fn: Optional[Callable[[torch.Tensor], None]] = None,
    ):
        """TTT training with repeated same-episode inner loops.

        Each outer batch item is treated as one environment instance. The same
        episode/window is replayed `video_ttt_num_tries` times with a persistent
        fast state. For every try, the current fast state is used for the action
        and future-video prediction, then the observation tokens produce the
        inner update used by the next chunk/try.
        """
        if not self.video_ttt_enabled:
            raise ValueError("`training_loss_video_ttt_observation` requires `model.video_ttt.enabled=true`.")
        if isinstance(sample.get("video"), torch.Tensor) and sample["video"].ndim in {6, 7}:
            return self.training_loss_video_ttt_repeat_attempt(sample, tiled=tiled, backward_fn=backward_fn)
        if "ttt_warmup_video" in sample:
            return self.training_loss_video_ttt_observe_then_act(sample, tiled=tiled, backward_fn=backward_fn)
        if backward_fn is not None:
            raise ValueError("Streaming backward is only supported for sequential TTT training samples.")

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        observation_context = inputs.get("observation_context")
        observation_context_mask = inputs.get("observation_context_mask")
        if observation_context is None:
            observation_context = context
        if observation_context_mask is None:
            observation_context_mask = context_mask
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        first_frame_latents = inputs["first_frame_latents"]
        if first_frame_latents is None:
            first_frame_latents = input_latents[:, :, 0:1]
        batch_size = first_frame_latents.shape[0]

        num_tries = max(int(self.video_ttt_num_tries), 1)
        use_switch_chunks = bool(self.video_ttt_switch_chunks and num_tries > 1)
        restart_context = inputs["restart_context"]
        restart_context_mask = inputs["restart_context_mask"]
        if use_switch_chunks and (restart_context is None or restart_context_mask is None):
            raise ValueError(
                "`model.video_ttt.switch_chunks=true` requires dataset `restart_instruction` "
                "so restart_context/restart_context_mask are available."
            )

        black_first_frame_latents = None
        if use_switch_chunks:
            black_video = torch.full(
                (
                    batch_size,
                    3,
                    1,
                    int(inputs["video_height"]),
                    int(inputs["video_width"]),
                ),
                -1.0,
                device=self.device,
                dtype=self.torch_dtype,
            )
            black_first_frame_latents = self._encode_video_latents(black_video, tiled=tiled)

        def build_observation_tokens(
            obs_latents: torch.Tensor,
            obs_context: torch.Tensor,
            obs_context_mask: torch.Tensor,
        ) -> torch.Tensor:
            timestep_video_obs = torch.zeros(
                (batch_size,),
                dtype=obs_latents.dtype,
                device=self.device,
            )
            obs_pre = self.video_expert.pre_dit(
                x=obs_latents,
                timestep=timestep_video_obs,
                context=obs_context,
                context_mask=obs_context_mask,
                action=None,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            )
            return obs_pre["tokens"]

        state = None
        loss_total = input_latents.new_tensor(0.0, dtype=torch.float32)
        loss_video_total = input_latents.new_tensor(0.0, dtype=torch.float32)
        loss_action_total = input_latents.new_tensor(0.0, dtype=torch.float32)
        ttt_loss_total = input_latents.new_tensor(0.0, dtype=torch.float32)
        ttt_switch_loss_total = input_latents.new_tensor(0.0, dtype=torch.float32)
        ttt_loss_count = 0
        ttt_switch_count = 0

        for try_idx in range(num_tries):
            if try_idx > 0 and use_switch_chunks:
                if self.video_ttt_uses_layer_hook:
                    state, ttt_switch_loss = self._update_video_ttt_observation_inline(
                        first_frame_latents=black_first_frame_latents,
                        context=restart_context,
                        context_mask=restart_context_mask,
                        fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                        state=state,
                        persist_state=False,
                        update=True,
                    )
                else:
                    switch_tokens = build_observation_tokens(
                        black_first_frame_latents,
                        restart_context,
                        restart_context_mask,
                    )
                    _, state, ttt_switch_loss = self._apply_video_ttt_observation(
                        switch_tokens,
                        state=state,
                        persist_state=False,
                        update=True,
                        update_tokens=switch_tokens,
                    )
                if ttt_switch_loss is not None:
                    ttt_switch_loss_total = ttt_switch_loss_total + ttt_switch_loss
                    ttt_loss_total = ttt_loss_total + ttt_switch_loss
                    ttt_switch_count += 1
                    ttt_loss_count += 1

            noise_video = torch.randn_like(input_latents)
            timestep_video = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=input_latents.dtype,
            )
            latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
            target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
            if inputs["first_frame_latents"] is not None:
                latents[:, :, 0:1] = inputs["first_frame_latents"]

            noise_action = torch.randn_like(action)
            timestep_action = self.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=action.dtype,
            )
            noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
            target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

            video_pre = self.video_expert.pre_dit(
                x=latents,
                timestep=timestep_video,
                context=context,
                context_mask=context_mask,
                action=action,
                fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            )
            video_ttt_hook = None
            if self.video_ttt_uses_layer_hook:
                state, ttt_loss = self._update_video_ttt_observation_inline(
                    first_frame_latents=first_frame_latents,
                    context=observation_context,
                    context_mask=observation_context_mask,
                    fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
                    state=state,
                    persist_state=False,
                    update=True,
                    global_time=inputs.get("ttt_global_time", None),
                )
                video_ttt_positions = self._build_video_ttt_positions(
                    meta=video_pre["meta"],
                    batch_size=int(video_pre["tokens"].shape[0]),
                    device=video_pre["tokens"].device,
                    global_time=inputs.get("ttt_global_time", None),
                    prefix_len=0,
                )
                video_tokens = video_pre["tokens"]
                video_ttt_hook = self._build_video_ttt_layer_hook(state=state, positions=video_ttt_positions)
            else:
                observation_tokens = build_observation_tokens(
                    first_frame_latents,
                    observation_context,
                    observation_context_mask,
                )
                video_tokens, state, ttt_loss = self._apply_video_ttt_observation(
                    video_pre["tokens"],
                    state=state,
                    persist_state=False,
                    update=True,
                    update_tokens=observation_tokens,
                )
            if ttt_loss is not None:
                ttt_loss_total = ttt_loss_total + ttt_loss
                ttt_loss_count += 1
            video_pre = dict(video_pre)
            video_pre["tokens"] = video_tokens

            action_pre = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )

            video_tokens = video_pre["tokens"]
            action_tokens = action_pre["tokens"]
            attention_mask = self._build_mot_attention_mask(
                video_seq_len=video_tokens.shape[1],
                action_seq_len=action_tokens.shape[1],
                video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                device=video_tokens.device,
            )
            tokens_out = self.mot(
                embeds_all={
                    "video": video_tokens,
                    "action": action_tokens,
                },
                attention_mask=attention_mask,
                freqs_all={
                    "video": video_pre["freqs"],
                    "action": action_pre["freqs"],
                },
                context_all={
                    "video": {
                        "context": video_pre["context"],
                        "mask": video_pre["context_mask"],
                    },
                    "action": {
                        "context": action_pre["context"],
                        "mask": action_pre["context_mask"],
                    },
                },
                t_mod_all={
                    "video": video_pre["t_mod"],
                    "action": action_pre["t_mod"],
                },
                video_ttt_hook=video_ttt_hook,
            )

            pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
            pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

            target_video_for_loss = target_video
            include_initial_video_step = inputs["first_frame_latents"] is None
            if inputs["first_frame_latents"] is not None:
                pred_video = pred_video[:, :, 1:]
                target_video_for_loss = target_video_for_loss[:, :, 1:]

            loss_video_per_sample = self._compute_video_loss_per_sample(
                pred_video=pred_video,
                target_video=target_video_for_loss,
                image_is_pad=image_is_pad,
                include_initial_video_step=include_initial_video_step,
            )
            video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
                loss_video_per_sample.device,
                dtype=loss_video_per_sample.dtype,
            )
            loss_video = (loss_video_per_sample * video_weight).mean()

            action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)
            if action_is_pad is not None:
                valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
                valid_sum = valid.sum(dim=1).clamp(min=1.0)
                action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
            else:
                action_loss_per_sample = action_loss_token.mean(dim=1)

            action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
                action_loss_per_sample.device,
                dtype=action_loss_per_sample.dtype,
            )
            loss_action = (action_loss_per_sample * action_weight).mean()

            loss_video_total = loss_video_total + loss_video
            loss_action_total = loss_action_total + loss_action
            loss_total = loss_total + self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action

        loss_video_avg = loss_video_total / float(num_tries)
        loss_action_avg = loss_action_total / float(num_tries)
        loss_total = loss_total / float(num_tries)
        loss_video_ttt = ttt_loss_total / float(max(ttt_loss_count, 1))
        if self.loss_lambda_video_ttt != 0.0:
            loss_total = loss_total + self.loss_lambda_video_ttt * loss_video_ttt

        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video_avg.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action_avg.detach().item()),
            "loss_video_ttt": float(loss_video_ttt.detach().item()),
            "video_ttt_num_tries": float(num_tries),
            "video_ttt_switch_chunks": float(1 if use_switch_chunks else 0),
        }
        if ttt_switch_count > 0:
            loss_video_ttt_switch = ttt_switch_loss_total / float(ttt_switch_count)
            loss_dict["loss_video_ttt_switch"] = float(loss_video_ttt_switch.detach().item())
        if self.loss_lambda_video_ttt != 0.0:
            loss_dict["loss_video_ttt_weighted"] = self.loss_lambda_video_ttt * loss_dict["loss_video_ttt"]
        return loss_total, loss_dict

    def training_loss(
        self,
        sample,
        tiled: bool = False,
        backward_fn: Optional[Callable[[torch.Tensor], None]] = None,
    ):
        if self.video_ttt_observation_training:
            return self.training_loss_video_ttt_observation(sample, tiled=tiled, backward_fn=backward_fn)
        if backward_fn is not None:
            raise ValueError("Streaming backward is only supported for video TTT observation training.")

        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        video_ttt_hook = None
        if self.video_ttt_enabled and self.video_ttt_w0_training:
            ttt_global_time = inputs.get("ttt_global_time", None)
            video_ttt_positions = self._build_video_ttt_positions(
                meta=video_pre["meta"],
                batch_size=int(video_pre["tokens"].shape[0]),
                device=video_pre["tokens"].device,
                global_time=ttt_global_time,
                prefix_len=0,
            )
            if self.video_ttt_uses_layer_hook:
                video_ttt_hook = self._build_video_ttt_layer_hook(
                    state=None,
                    positions=video_ttt_positions,
                )
            else:
                video_tokens, _, _ = self._apply_video_ttt_observation(
                    video_pre["tokens"],
                    state=None,
                    persist_state=False,
                    update=False,
                    update_tokens=None,
                    positions=video_ttt_positions,
                    compute_loss=False,
                )
                video_pre = dict(video_pre)
                video_pre["tokens"] = video_tokens

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
            video_ttt_hook=video_ttt_hook,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        if self.video_ttt_enabled and self.video_ttt_w0_training:
            loss_dict.update(self._video_ttt_gate_metrics())
            loss_dict["video_ttt_w0_training"] = 1.0
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        use_video_ttt: bool = True,
        video_ttt_global_time: Optional[torch.Tensor | int | float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        video_tokens = video_pre["tokens"]
        video_ttt_hook = None
        if use_video_ttt:
            video_ttt_positions = self._build_video_ttt_positions(
                meta=video_pre["meta"],
                batch_size=int(video_pre["tokens"].shape[0]),
                device=video_pre["tokens"].device,
                global_time=video_ttt_global_time,
                prefix_len=0,
            )
            if self.video_ttt_uses_layer_hook:
                video_ttt_hook = self._build_video_ttt_layer_hook(positions=video_ttt_positions)
            else:
                video_tokens, _, _ = self._apply_video_ttt_observation(
                    video_tokens,
                    state=self._video_ttt_inference_state,
                    persist_state=False,
                    update=False,
                    update_tokens=None,
                    positions=video_ttt_positions,
                    compute_loss=False,
                )

        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
            video_ttt_hook=video_ttt_hook,
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        update_video_ttt: bool = True,
        ttt_global_time: Optional[int | float | torch.Tensor] = None,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
                update_video_ttt=False,
                ttt_global_time=ttt_global_time,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        if self.video_ttt_enabled and update_video_ttt:
            if self.video_ttt_uses_layer_hook:
                _, _ = self._update_video_ttt_observation_inline(
                    first_frame_latents=first_frame_latents,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    state=self._video_ttt_inference_state,
                    persist_state=True,
                    update=True,
                    global_time=ttt_global_time,
                    compute_loss=False,
                )
            else:
                observation_tokens = self._build_video_ttt_observation_tokens(
                    first_frame_latents=first_frame_latents,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    global_time=ttt_global_time,
                )
                _, _, _ = self._apply_video_ttt_observation(
                    observation_tokens,
                    state=self._video_ttt_inference_state,
                    persist_state=True,
                    update=True,
                    update_tokens=observation_tokens,
                    compute_loss=False,
                )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
                use_video_ttt=True,
                video_ttt_global_time=ttt_global_time,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        update_video_ttt: bool = True,
        ttt_global_time: Optional[int | float | torch.Tensor] = None,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        if self.video_ttt_enabled and update_video_ttt:
            if self.video_ttt_uses_layer_hook:
                _, _ = self._update_video_ttt_observation_inline(
                    first_frame_latents=first_frame_latents,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    state=self._video_ttt_inference_state,
                    persist_state=True,
                    update=True,
                    global_time=ttt_global_time,
                    compute_loss=False,
                )
            else:
                video_ttt_update_tokens = self._build_video_ttt_observation_tokens(
                    first_frame_latents=first_frame_latents,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    global_time=ttt_global_time,
                )
                _, _, _ = self._apply_video_ttt_observation(
                    video_ttt_update_tokens,
                    state=self._video_ttt_inference_state,
                    persist_state=True,
                    update=True,
                    update_tokens=video_ttt_update_tokens,
                    compute_loss=False,
                )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            video_ttt_hook=self._build_video_ttt_layer_hook(
                positions=self._build_video_ttt_positions(
                    meta=video_pre["meta"],
                    batch_size=int(video_pre["tokens"].shape[0]),
                    device=video_pre["tokens"].device,
                    global_time=ttt_global_time,
                    prefix_len=0,
                ),
            ),
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        update_video_ttt: bool = True,
        ttt_global_time: Optional[int | float | torch.Tensor] = None,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            update_video_ttt=update_video_ttt,
            ttt_global_time=ttt_global_time,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
