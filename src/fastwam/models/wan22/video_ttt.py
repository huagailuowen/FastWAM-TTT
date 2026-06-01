from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoTTTQKVAdapter(nn.Module):
    """Video-only TTT-QKV adapter with explicit per-sequence fast weights.

    The inner update is intentionally local to observed video tokens. It mirrors
    the TTT-QKV pattern:

        K = theta_K(x), V = theta_V(x), Q = theta_Q(x)
        W' = W - lr * grad || f(K; W) - target(V, K) ||^2
        y = x + gate * f(Q; W')

    `state` carries the fast learner across observation chunks at inference.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        mini_batch_size: int = 64,
        ttt_lr: float = 0.1,
        init_std: float = 0.02,
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

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.mini_batch_size = int(mini_batch_size)
        self.ttt_lr = float(ttt_lr)

        self.input_norm = nn.LayerNorm(self.hidden_dim, eps=1e-6)
        self.to_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.to_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.to_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.W_init = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.head_dim))
        self.b_init = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))
        self.ttt_norm_weight = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.ttt_norm_bias = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        # Exact zero keeps an enabled adapter behaviorally identical at step 0.
        self.residual_gate = nn.Parameter(torch.zeros(()))

        self.reset_parameters(init_std=init_std)

    def reset_parameters(self, *, init_std: float = 0.02) -> None:
        nn.init.normal_(self.W_init, mean=0.0, std=init_std)
        nn.init.zeros_(self.b_init)
        nn.init.ones_(self.ttt_norm_weight)
        nn.init.zeros_(self.ttt_norm_bias)
        nn.init.zeros_(self.residual_gate)

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

    @staticmethod
    def detach_state(state: Optional[dict[str, torch.Tensor]]) -> Optional[dict[str, torch.Tensor]]:
        if state is None:
            return None
        return {key: value.detach() for key, value in state.items()}

    def _state_matches(self, state: dict[str, torch.Tensor], tokens: torch.Tensor) -> bool:
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

    def _project_qkv(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return q, k, v.float()

    def _head_layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.float().mean(dim=-1, keepdim=True)
        var = x.float().var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x.float() - mean) * torch.rsqrt(var + 1e-6)
        weight = self.ttt_norm_weight.to(device=x.device, dtype=x_norm.dtype).view(1, self.num_heads, 1, self.head_dim)
        bias = self.ttt_norm_bias.to(device=x.device, dtype=x_norm.dtype).view(1, self.num_heads, 1, self.head_dim)
        return (x_norm * weight + bias).to(dtype=x.dtype)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        state: Optional[dict[str, torch.Tensor]] = None,
        update: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"`tokens` must be [B, S, D], got shape {tuple(tokens.shape)}.")
        if tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"`tokens` hidden dim mismatch: expected {self.hidden_dim}, got {tokens.shape[-1]}."
            )

        if state is None or not self._state_matches(state, tokens):
            state = self.init_state(tokens.shape[0], device=tokens.device, dtype=tokens.dtype)
        W = state["W"]
        b = state["b"]

        q, k, v = self._project_qkv(tokens)

        outputs = []
        loss_sum = tokens.new_tensor(0.0, dtype=torch.float32)
        num_chunks = 0
        lr = self.ttt_lr / max(float(self.head_dim), 1.0)

        for start in range(0, tokens.shape[1], self.mini_batch_size):
            end = min(start + self.mini_batch_size, tokens.shape[1])
            q_mb = q[:, :, start:end]
            k_mb = k[:, :, start:end]
            v_mb = v[:, :, start:end]

            target_delta = self._head_layer_norm(v_mb - k_mb)
            pred_delta = torch.matmul(k_mb, W) + b
            err = pred_delta - target_delta
            loss_sum = loss_sum + err.float().pow(2).mean()
            num_chunks += 1

            if update and lr > 0:
                scale = 1.0 / max(float(end - start), 1.0)
                grad_W = torch.matmul(k_mb.transpose(-2, -1), err) * scale
                grad_b = err.mean(dim=-2, keepdim=True)
                W = W - lr * grad_W
                b = b - lr * grad_b

            out_delta = self._head_layer_norm(torch.matmul(q_mb, W) + b)
            outputs.append(q_mb + out_delta)

        out_heads = torch.cat(outputs, dim=2)
        out = out_heads.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[1], self.hidden_dim)
        out = self.out_proj(out.to(dtype=tokens.dtype))
        adapted = tokens + self.residual_gate.to(dtype=tokens.dtype) * out
        loss = loss_sum / max(num_chunks, 1)
        return adapted, {"W": W, "b": b}, loss
