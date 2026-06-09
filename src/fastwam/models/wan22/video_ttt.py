from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .video_ttt_triton import video_ttt_scan_triton


class VideoTTTQKVAdapter(nn.Module):
    """Video-only TTT-QKV adapter with explicit per-sequence fast weights.

    The inner update is intentionally local to observed video tokens. It mirrors
    the TTT-QKV pattern:

        K = theta_K(x), V = theta_V(x), Q = theta_Q(x)
        y = x + gate * f(Q; W)
        W_next = W - lr * grad || f(K; W) - target(V, K) ||^2

    `state` carries the fast learner across observation chunks at inference.
    When the same sequence is used for output and update, the inner learner scans
    mini-batches in order so later tokens see fast weights updated by earlier
    tokens. If a separate shorter observation sequence is supplied as
    `update_tokens`, the current tokens are adapted with the incoming state and
    the observation sequence updates the state for the next chunk.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        mini_batch_size: int = 64,
        ttt_lr: float = 0.1,
        init_std: float = 0.02,
        residual_gate_init: float = 0.0,
        num_layers: int = 1,
        rank: Optional[int] = None,
        inner_update_mode: str = "scan",
        scan_kernel: str = "torch",
        use_global_rope: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be positive, got {hidden_dim}.")
        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be positive, got {num_heads}.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"`hidden_dim` must be divisible by `num_heads`, got {hidden_dim} and {num_heads}."
            )
        if mini_batch_size <= 0:
            raise ValueError(f"`mini_batch_size` must be positive, got {mini_batch_size}.")
        if ttt_lr < 0:
            raise ValueError(f"`ttt_lr` must be non-negative, got {ttt_lr}.")
        if num_layers <= 0:
            raise ValueError(f"`num_layers` must be positive, got {num_layers}.")
        ttt_dim = hidden_dim if rank is None else int(rank)
        if ttt_dim <= 0:
            raise ValueError(f"`rank` must be positive when provided, got {rank}.")
        if ttt_dim % num_heads != 0:
            raise ValueError(f"`rank` must be divisible by `num_heads`, got {ttt_dim} and {num_heads}.")
        inner_update_mode = str(inner_update_mode).strip().lower()
        if inner_update_mode not in {"scan", "mean"}:
            raise ValueError(
                "`inner_update_mode` must be 'scan' or 'mean', "
                f"got {inner_update_mode!r}."
            )
        scan_kernel = str(scan_kernel).strip().lower()
        if scan_kernel not in {"torch", "triton", "auto"}:
            raise ValueError(
                "`scan_kernel` must be 'torch', 'triton', or 'auto', "
                f"got {scan_kernel!r}."
            )

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.ttt_dim = int(ttt_dim)
        self.head_dim = self.ttt_dim // self.num_heads
        self.mini_batch_size = int(mini_batch_size)
        self.ttt_lr = float(ttt_lr)
        self.residual_gate_init = float(residual_gate_init)
        self.num_layers = int(num_layers)
        self.legacy_single_layer = self.num_layers == 1 and rank is None
        self.inner_update_mode = inner_update_mode
        self.scan_kernel = scan_kernel
        self.use_global_rope = bool(use_global_rope)
        self.rope_theta = float(rope_theta)
        if self.use_global_rope and self.head_dim % 2 != 0:
            raise ValueError(f"`head_dim` must be even for global RoPE, got {self.head_dim}.")
        rope_pairs = self.head_dim // 2
        self.rope_height_dim = 2 * (rope_pairs // 3)
        self.rope_width_dim = 2 * (rope_pairs // 3)
        self.rope_time_dim = self.head_dim - self.rope_height_dim - self.rope_width_dim

        self.input_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)
        self.to_q = nn.Linear(self.hidden_dim, self.ttt_dim)
        self.to_k = nn.Linear(self.hidden_dim, self.ttt_dim)
        self.to_v = nn.Linear(self.hidden_dim, self.ttt_dim)
        self.out_proj = nn.Linear(self.ttt_dim, self.hidden_dim)

        if self.legacy_single_layer:
            self.W_init = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.head_dim))
            self.b_init = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
        else:
            self.W_init = nn.Parameter(torch.empty(self.num_layers, self.num_heads, self.head_dim, self.head_dim))
            self.b_init = nn.Parameter(torch.zeros(self.num_layers, self.num_heads, 1, self.head_dim))
        self.ttt_norm_weight = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.ttt_norm_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        # Exact zero keeps an enabled adapter behaviorally identical at step 0.
        if self.legacy_single_layer:
            self.residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))
        else:
            self.residual_gate = nn.Parameter(torch.zeros(self.num_layers, self.hidden_dim))

        self.reset_parameters(init_std=init_std)

    def reset_parameters(self, *, init_std: float = 0.02) -> None:
        nn.init.normal_(self.W_init, mean=0.0, std=init_std)
        nn.init.zeros_(self.b_init)
        nn.init.ones_(self.ttt_norm_weight)
        nn.init.zeros_(self.ttt_norm_bias)
        self.residual_gate.data.fill_(self.residual_gate_init)

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        del dtype
        if batch_size <= 0:
            raise ValueError(f"`batch_size` must be positive, got {batch_size}.")
        if self.legacy_single_layer:
            return {
                "W": self.W_init.to(device=device, dtype=torch.float32)
                .unsqueeze(0)
                .expand(batch_size, -1, -1, -1)
                .clone(),
                "b": self.b_init.to(device=device, dtype=torch.float32)
                .unsqueeze(0)
                .expand(batch_size, -1, -1, -1)
                .clone(),
            }
        return {
            "W": self.W_init.to(device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1, -1)
            .clone(),
            "b": self.b_init.to(device=device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1, -1)
            .clone(),
        }

    @staticmethod
    def detach_state(state: Optional[dict[str, torch.Tensor]]) -> Optional[dict[str, torch.Tensor]]:
        if state is None:
            return None
        return {key: value.detach() for key, value in state.items()}

    def _state_matches(self, state: dict[str, torch.Tensor], tokens: torch.Tensor) -> bool:
        if self.legacy_single_layer:
            return (
                "W" in state
                and "b" in state
                and state["W"].shape == (tokens.shape[0], self.num_heads, self.head_dim, self.head_dim)
                and state["b"].shape == (tokens.shape[0], self.num_heads, 1, self.head_dim)
                and state["W"].device == tokens.device
                and state["b"].device == tokens.device
                and state["W"].dtype == torch.float32
                and state["b"].dtype == torch.float32
            )
        return (
            "W" in state
            and "b" in state
            and state["W"].shape == (tokens.shape[0], self.num_layers, self.num_heads, self.head_dim, self.head_dim)
            and state["b"].shape == (tokens.shape[0], self.num_layers, self.num_heads, 1, self.head_dim)
            and state["W"].device == tokens.device
            and state["b"].device == tokens.device
            and state["W"].dtype == torch.float32
            and state["b"].dtype == torch.float32
        )

    @staticmethod
    def _position_tensor(
        positions: Optional[dict[str, torch.Tensor]],
        key: str,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if positions is None or key not in positions or positions[key] is None:
            return None
        value = positions[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        value = value.to(device=device, dtype=torch.float32)
        if value.ndim == 0:
            value = value.view(1, 1).expand(batch_size, seq_len)
        elif value.ndim == 1:
            if value.shape[0] == seq_len:
                value = value.view(1, seq_len).expand(batch_size, -1)
            elif value.shape[0] == batch_size:
                value = value.view(batch_size, 1).expand(-1, seq_len)
            else:
                raise ValueError(
                    f"`positions[{key}]` must have length batch_size={batch_size} or seq_len={seq_len}, "
                    f"got {value.shape[0]}."
                )
        elif value.ndim == 2:
            if value.shape == (batch_size, seq_len):
                pass
            elif value.shape == (1, seq_len):
                value = value.expand(batch_size, -1)
            elif value.shape == (batch_size, 1):
                value = value.expand(-1, seq_len)
            else:
                raise ValueError(
                    f"`positions[{key}]` must be [B,S], [1,S], or [B,1], got {tuple(value.shape)}."
                )
        else:
            raise ValueError(f"`positions[{key}]` must be scalar, 1D, or 2D, got {tuple(value.shape)}.")
        return value

    def _apply_rope_axis(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        *,
        start: int,
        dim: int,
    ) -> torch.Tensor:
        if dim <= 0:
            return x
        if dim % 2 != 0:
            raise ValueError(f"RoPE axis dim must be even, got {dim}.")
        stop = int(start) + int(dim)
        part = x[..., start:stop]
        pair = part.float().reshape(part.shape[0], part.shape[1], part.shape[2], dim // 2, 2)
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / float(dim))
        )
        angle = positions.to(device=x.device, dtype=torch.float32)[:, None, :, None] * inv_freq.view(1, 1, 1, -1)
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        x0 = pair[..., 0]
        x1 = pair[..., 1]
        rotated = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)
        x = x.clone()
        x[..., start:stop] = rotated.to(dtype=x.dtype)
        return x

    def _apply_global_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: Optional[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_global_rope or positions is None:
            return q, k
        batch_size, _, seq_len, _ = q.shape
        time_pos = self._position_tensor(
            positions,
            "time",
            batch_size=batch_size,
            seq_len=seq_len,
            device=q.device,
        )
        height_pos = self._position_tensor(
            positions,
            "height",
            batch_size=batch_size,
            seq_len=seq_len,
            device=q.device,
        )
        width_pos = self._position_tensor(
            positions,
            "width",
            batch_size=batch_size,
            seq_len=seq_len,
            device=q.device,
        )

        start = 0
        if time_pos is not None:
            q = self._apply_rope_axis(q, time_pos, start=start, dim=self.rope_time_dim)
            k = self._apply_rope_axis(k, time_pos, start=start, dim=self.rope_time_dim)
        start += self.rope_time_dim
        if height_pos is not None:
            q = self._apply_rope_axis(q, height_pos, start=start, dim=self.rope_height_dim)
            k = self._apply_rope_axis(k, height_pos, start=start, dim=self.rope_height_dim)
        start += self.rope_height_dim
        if width_pos is not None:
            q = self._apply_rope_axis(q, width_pos, start=start, dim=self.rope_width_dim)
            k = self._apply_rope_axis(k, width_pos, start=start, dim=self.rope_width_dim)
        return q, k

    def _project_qkv(
        self,
        tokens: torch.Tensor,
        positions: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_norm(tokens)
        batch_size, seq_len, _ = x.shape

        def reshape_heads(y: torch.Tensor) -> torch.Tensor:
            y = y.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            return y.transpose(1, 2).contiguous()

        q = reshape_heads(self.to_q(x))
        k = reshape_heads(self.to_k(x))
        v = reshape_heads(self.to_v(x))

        q = F.normalize(q.float(), p=2, dim=-1)
        k = F.normalize(k.float(), p=2, dim=-1)
        q, k = self._apply_global_rope(q, k, positions)
        return q, k, v.float()

    def _head_layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.float().mean(dim=-1, keepdim=True)
        var = x.float().var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x.float() - mean) * torch.rsqrt(var + 1e-6)
        weight = self.ttt_norm_weight.to(device=x.device, dtype=x_norm.dtype).view(1, self.num_heads, 1, self.head_dim)
        bias = self.ttt_norm_bias.to(device=x.device, dtype=x_norm.dtype).view(1, self.num_heads, 1, self.head_dim)
        return (x_norm * weight + bias).to(dtype=x.dtype)

    def _apply_fast_weights(
        self,
        q: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        outputs = []
        for start in range(0, q.shape[2], self.mini_batch_size):
            end = min(start + self.mini_batch_size, q.shape[2])
            q_mb = q[:, :, start:end]
            out_delta = self._head_layer_norm(torch.matmul(q_mb, W) + b)
            outputs.append(q_mb + out_delta)
        return torch.cat(outputs, dim=2)

    def _inner_update_step(
        self,
        *,
        k_mb: torch.Tensor,
        v_mb: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor,
        lr: float,
        update: bool,
        compute_loss: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        target_delta = self._head_layer_norm(v_mb - k_mb)
        pred_delta = torch.matmul(k_mb, W) + b
        err = pred_delta - target_delta

        loss = k_mb.new_tensor(0.0, dtype=torch.float32)
        loss_count = 0
        if compute_loss:
            loss = err.float().pow(2).mean()
            loss_count = 1

        if update and lr > 0:
            scale = 1.0 / max(float(k_mb.shape[2]), 1.0)
            grad_W = torch.matmul(k_mb.transpose(-2, -1), err) * scale
            grad_b = err.mean(dim=-2, keepdim=True)
            W = W - lr * grad_W
            b = b - lr * grad_b
        return W, b, loss, loss_count

    def _scan_update_and_output(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor,
        lr: float,
        update: bool,
        compute_loss: bool,
        emit_outputs: bool,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if update and emit_outputs and self.scan_kernel != "torch":
            triton_result = video_ttt_scan_triton(
                q,
                k,
                v,
                W,
                b,
                self.ttt_norm_weight,
                self.ttt_norm_bias,
                lr=lr,
                mini_batch_size=self.mini_batch_size,
            )
            if triton_result is not None:
                return triton_result
            if self.scan_kernel == "triton":
                raise RuntimeError(
                    "video_ttt.scan_kernel='triton' was requested but the Triton "
                    "same-sequence scan path is unavailable for this shape/device."
                )

        outputs = []
        loss_sum = q.new_tensor(0.0, dtype=torch.float32)
        loss_count = 0
        for start in range(0, k.shape[2], self.mini_batch_size):
            end = min(start + self.mini_batch_size, k.shape[2])
            k_mb = k[:, :, start:end]
            v_mb = v[:, :, start:end]
            W, b, loss, count = self._inner_update_step(
                k_mb=k_mb,
                v_mb=v_mb,
                W=W,
                b=b,
                lr=lr,
                update=update,
                compute_loss=compute_loss,
            )
            if compute_loss:
                loss_sum = loss_sum + loss
                loss_count += count
            if emit_outputs:
                q_mb = q[:, :, start:end]
                out_delta = self._head_layer_norm(torch.matmul(q_mb, W) + b)
                outputs.append(q_mb + out_delta)
        out = torch.cat(outputs, dim=2) if outputs else None
        return out, W, b, loss_sum, loss_count

    def _mean_update_and_output(
        self,
        *,
        q: torch.Tensor,
        update_k: torch.Tensor,
        update_v: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor,
        lr: float,
        update: bool,
        compute_loss: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        out = self._apply_fast_weights(q, W, b)
        loss_sum = q.new_tensor(0.0, dtype=torch.float32)
        grad_W_sum: Optional[torch.Tensor] = None
        grad_b_sum: Optional[torch.Tensor] = None
        num_chunks = 0
        num_update_chunks = 0

        if compute_loss or (update and lr > 0):
            for start in range(0, update_k.shape[2], self.mini_batch_size):
                end = min(start + self.mini_batch_size, update_k.shape[2])
                k_mb = update_k[:, :, start:end]
                v_mb = update_v[:, :, start:end]

                target_delta = self._head_layer_norm(v_mb - k_mb)
                pred_delta = torch.matmul(k_mb, W) + b
                err = pred_delta - target_delta
                if compute_loss:
                    loss_sum = loss_sum + err.float().pow(2).mean()
                    num_chunks += 1

                if update and lr > 0:
                    scale = 1.0 / max(float(end - start), 1.0)
                    grad_W = torch.matmul(k_mb.transpose(-2, -1), err) * scale
                    grad_b = err.mean(dim=-2, keepdim=True)
                    grad_W_sum = grad_W if grad_W_sum is None else grad_W_sum + grad_W
                    grad_b_sum = grad_b if grad_b_sum is None else grad_b_sum + grad_b
                    num_update_chunks += 1

        if update and lr > 0 and num_update_chunks > 0:
            W = W - lr * (grad_W_sum / float(num_update_chunks))
            b = b - lr * (grad_b_sum / float(num_update_chunks))
        return out, W, b, loss_sum, num_chunks

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        state: Optional[dict[str, torch.Tensor]] = None,
        update: bool = True,
        update_tokens: Optional[torch.Tensor] = None,
        positions: Optional[dict[str, torch.Tensor]] = None,
        update_positions: Optional[dict[str, torch.Tensor]] = None,
        layer_idx: int = 0,
        compute_loss: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"`tokens` must be [B, S, D], got shape {tuple(tokens.shape)}.")
        if tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"`tokens` hidden dim mismatch: expected {self.hidden_dim}, got {tokens.shape[-1]}."
            )
        if update_tokens is not None:
            if update_tokens.ndim != 3:
                raise ValueError(
                    f"`update_tokens` must be [B, S, D], got shape {tuple(update_tokens.shape)}."
                )
            if update_tokens.shape[0] != tokens.shape[0] or update_tokens.shape[-1] != self.hidden_dim:
                raise ValueError(
                    "`update_tokens` must share batch and hidden dims with `tokens`, "
                    f"got {tuple(update_tokens.shape)} vs {tuple(tokens.shape)}."
                )
        layer_idx = int(layer_idx)
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"`layer_idx` must be in [0, {self.num_layers}), got {layer_idx}.")

        if state is None or not self._state_matches(state, tokens):
            state = self.init_state(tokens.shape[0], device=tokens.device, dtype=tokens.dtype)
        if self.legacy_single_layer:
            W = state["W"]
            b = state["b"]
        else:
            W = state["W"][:, layer_idx]
            b = state["b"][:, layer_idx]

        q, k, v = self._project_qkv(tokens, positions=positions)
        if update_tokens is None:
            update_k = k
            update_v = v
        else:
            if update_positions is None:
                update_positions = positions
            _, update_k, update_v = self._project_qkv(update_tokens, positions=update_positions)

        lr = self.ttt_lr / max(float(self.head_dim), 1.0)
        same_update_sequence = update_tokens is None or update_k.shape[2] == q.shape[2]
        if self.inner_update_mode == "mean":
            out_heads, W, b, loss_sum, num_chunks = self._mean_update_and_output(
                q=q,
                update_k=update_k,
                update_v=update_v,
                W=W,
                b=b,
                lr=lr,
                update=update,
                compute_loss=compute_loss,
            )
        elif same_update_sequence:
            out_heads, W, b, loss_sum, num_chunks = self._scan_update_and_output(
                q=q,
                k=update_k,
                v=update_v,
                W=W,
                b=b,
                lr=lr,
                update=update,
                compute_loss=compute_loss,
                emit_outputs=True,
            )
            if out_heads is None:
                out_heads = self._apply_fast_weights(q, W, b)
        else:
            out_heads = self._apply_fast_weights(q, W, b)
            _, W, b, loss_sum, num_chunks = self._scan_update_and_output(
                q=update_k,
                k=update_k,
                v=update_v,
                W=W,
                b=b,
                lr=lr,
                update=update,
                compute_loss=compute_loss,
                emit_outputs=False,
            )

        out = out_heads.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[1], self.ttt_dim)
        out = self.out_proj(out.to(dtype=tokens.dtype))
        new_state = {
            "W": state["W"].clone(),
            "b": state["b"].clone(),
        }
        if self.legacy_single_layer:
            new_state["W"] = W
            new_state["b"] = b
            gate = torch.tanh(self.residual_gate).to(device=tokens.device, dtype=tokens.dtype)
            adapted = tokens + gate * out
        else:
            new_state["W"][:, layer_idx] = W
            new_state["b"][:, layer_idx] = b
            gate = torch.tanh(self.residual_gate[layer_idx]).to(device=tokens.device, dtype=tokens.dtype)
            adapted = tokens + gate.view(1, 1, self.hidden_dim) * out
        loss = loss_sum / max(num_chunks, 1)
        return adapted, new_state, loss
