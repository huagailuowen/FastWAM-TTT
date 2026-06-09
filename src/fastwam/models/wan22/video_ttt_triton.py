from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - Triton is optional.
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None and tl is not None


def _head_layer_norm_with_params(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
) -> torch.Tensor:
    mean = x.float().mean(dim=-1, keepdim=True)
    var = x.float().var(dim=-1, unbiased=False, keepdim=True)
    x_norm = (x.float() - mean) * torch.rsqrt(var + 1e-6)
    weight = norm_weight.to(device=x.device, dtype=x_norm.dtype).view(1, norm_weight.shape[0], 1, norm_weight.shape[1])
    bias = norm_bias.to(device=x.device, dtype=x_norm.dtype).view(1, norm_bias.shape[0], 1, norm_bias.shape[1])
    return (x_norm * weight + bias).to(dtype=x.dtype)


def video_ttt_scan_torch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    *,
    lr: float,
    mini_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    outputs = []
    loss_sum = q.new_tensor(0.0, dtype=torch.float32)
    loss_count = 0
    for start in range(0, k.shape[2], int(mini_batch_size)):
        end = min(start + int(mini_batch_size), k.shape[2])
        k_mb = k[:, :, start:end]
        v_mb = v[:, :, start:end]
        target_delta = _head_layer_norm_with_params(v_mb - k_mb, norm_weight, norm_bias)
        pred_delta = torch.matmul(k_mb, W) + b
        err = pred_delta - target_delta
        loss_sum = loss_sum + err.float().pow(2).mean()
        loss_count += 1
        if lr > 0:
            scale = 1.0 / max(float(end - start), 1.0)
            grad_W = torch.matmul(k_mb.transpose(-2, -1), err) * scale
            grad_b = err.mean(dim=-2, keepdim=True)
            W = W - float(lr) * grad_W
            b = b - float(lr) * grad_b
        q_mb = q[:, :, start:end]
        out_delta = _head_layer_norm_with_params(torch.matmul(q_mb, W) + b, norm_weight, norm_bias)
        outputs.append(q_mb + out_delta)
    out = torch.cat(outputs, dim=2)
    return out, W, b, loss_sum, loss_count


if triton_available():

    @triton.jit
    def _ln_rows(x, norm_weight, norm_bias, valid_tokens, valid_features, F: tl.constexpr):
        x = tl.where(valid_tokens[:, None] & valid_features[None, :], x, 0.0)
        mean = tl.sum(x, axis=1) / F
        centered = tl.where(valid_features[None, :], x - mean[:, None], 0.0)
        var = tl.sum(centered * centered, axis=1) / F
        rstd = tl.rsqrt(var + 1.0e-6)
        y = centered * rstd[:, None]
        y = y * norm_weight[None, :] + norm_bias[None, :]
        return tl.where(valid_tokens[:, None] & valid_features[None, :], y, 0.0)


    @triton.jit
    def _video_ttt_scan_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        W_ptr,
        b_ptr,
        norm_weight_ptr,
        norm_bias_ptr,
        out_ptr,
        W_last_ptr,
        b_last_ptr,
        loss_sq_ptr,
        lr: tl.constexpr,
        H: tl.constexpr,
        S: tl.constexpr,
        F: tl.constexpr,
        NC: tl.constexpr,
        CS: tl.constexpr,
        BLOCK_F: tl.constexpr,
    ):
        batch = tl.program_id(0)
        head = tl.program_id(1)

        offs_f = tl.arange(0, BLOCK_F)
        offs_i = tl.arange(0, BLOCK_F)[:, None]
        offs_j = tl.arange(0, BLOCK_F)[None, :]
        valid_f = offs_f < F
        valid_ij = (offs_i < F) & (offs_j < F)

        bh = batch * H + head
        W_offsets = bh * F * F + offs_i * F + offs_j
        b_offsets = bh * F + offs_f
        norm_offsets = head * F + offs_f

        W = tl.load(W_ptr + W_offsets, mask=valid_ij, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + b_offsets, mask=valid_f, other=0.0).to(tl.float32)
        norm_weight = tl.load(norm_weight_ptr + norm_offsets, mask=valid_f, other=0.0).to(tl.float32)
        norm_bias = tl.load(norm_bias_ptr + norm_offsets, mask=valid_f, other=0.0).to(tl.float32)

        offs_cs = tl.arange(0, CS)
        for chunk_idx in range(0, NC):
            token = chunk_idx * CS + offs_cs
            valid_t = token < S
            token_feature_offsets = bh * S * F + token[:, None] * F + offs_f[None, :]
            token_feature_mask = valid_t[:, None] & valid_f[None, :]

            q_mb = tl.load(q_ptr + token_feature_offsets, mask=token_feature_mask, other=0.0).to(tl.float32)
            k_mb = tl.load(k_ptr + token_feature_offsets, mask=token_feature_mask, other=0.0).to(tl.float32)
            v_mb = tl.load(v_ptr + token_feature_offsets, mask=token_feature_mask, other=0.0).to(tl.float32)

            target_delta = _ln_rows(v_mb - k_mb, norm_weight, norm_bias, valid_t, valid_f, F)
            pred_delta = tl.dot(k_mb, W, input_precision="ieee") + b[None, :]
            err = tl.where(token_feature_mask, pred_delta - target_delta, 0.0)
            loss_sq = tl.sum(err * err)
            tl.store(loss_sq_ptr + bh * NC + chunk_idx, loss_sq)

            chunk_remaining = S - chunk_idx * CS
            chunk_len = tl.minimum(chunk_remaining, CS)
            inv_chunk_len = 1.0 / chunk_len
            grad_W = tl.dot(tl.trans(k_mb), err, input_precision="ieee") * inv_chunk_len
            grad_b = tl.sum(err, axis=0) * inv_chunk_len
            W = W - lr * grad_W
            b = b - lr * grad_b

            out_delta = _ln_rows(
                tl.dot(q_mb, W, input_precision="ieee") + b[None, :],
                norm_weight,
                norm_bias,
                valid_t,
                valid_f,
                F,
            )
            out = q_mb + out_delta
            tl.store(out_ptr + token_feature_offsets, out, mask=token_feature_mask)

        tl.store(W_last_ptr + W_offsets, W, mask=valid_ij)
        tl.store(b_last_ptr + b_offsets, b, mask=valid_f)


class _VideoTTTLinearScanTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        W: torch.Tensor,
        b: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        lr: float,
        mini_batch_size: int,
    ):
        if not triton_available():
            raise RuntimeError("Triton is not available.")
        if q.ndim != 4 or W.ndim != 4 or b.ndim != 4:
            raise ValueError("Expected q/k/v [B,H,S,F], W [B,H,F,F], b [B,H,1,F].")
        batch_size, num_heads, seq_len, head_dim = q.shape
        if mini_batch_size <= 0:
            raise ValueError(f"`mini_batch_size` must be positive, got {mini_batch_size}.")
        num_chunks = triton.cdiv(seq_len, int(mini_batch_size))
        block_f = max(16, triton.next_power_of_2(head_dim))

        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        W_c = W.contiguous()
        b_c = b.contiguous()
        norm_weight_c = norm_weight.to(device=q.device, dtype=torch.float32).contiguous()
        norm_bias_c = norm_bias.to(device=q.device, dtype=torch.float32).contiguous()

        out = torch.empty_like(q_c)
        W_last = torch.empty_like(W_c)
        b_last = torch.empty_like(b_c)
        loss_sq = torch.empty((batch_size, num_heads, num_chunks), device=q.device, dtype=torch.float32)

        _video_ttt_scan_forward_kernel[(batch_size, num_heads)](
            q_c,
            k_c,
            v_c,
            W_c,
            b_c,
            norm_weight_c,
            norm_bias_c,
            out,
            W_last,
            b_last,
            loss_sq,
            lr=float(lr),
            H=num_heads,
            S=seq_len,
            F=head_dim,
            NC=num_chunks,
            CS=int(mini_batch_size),
            BLOCK_F=block_f,
        )

        chunk_sizes = torch.full((num_chunks,), int(mini_batch_size), device=q.device, dtype=torch.float32)
        last_chunk = seq_len - int(mini_batch_size) * (num_chunks - 1)
        chunk_sizes[-1] = float(last_chunk)
        loss_sum = (loss_sq.sum(dim=(0, 1)) / (float(batch_size * num_heads * head_dim) * chunk_sizes)).sum()

        ctx.save_for_backward(q_c, k_c, v_c, W_c, b_c, norm_weight_c, norm_bias_c)
        ctx.lr = float(lr)
        ctx.mini_batch_size = int(mini_batch_size)
        return out, W_last, b_last, loss_sum

    @staticmethod
    def backward(ctx, grad_out, grad_W_last, grad_b_last, grad_loss_sum):
        q, k, v, W, b, norm_weight, norm_bias = ctx.saved_tensors
        grad_out = torch.zeros_like(q) if grad_out is None else grad_out
        grad_W_last = torch.zeros_like(W) if grad_W_last is None else grad_W_last
        grad_b_last = torch.zeros_like(b) if grad_b_last is None else grad_b_last
        grad_loss_sum = q.new_tensor(0.0, dtype=torch.float32) if grad_loss_sum is None else grad_loss_sum

        with torch.enable_grad():
            q_ref = q.detach().requires_grad_(True)
            k_ref = k.detach().requires_grad_(True)
            v_ref = v.detach().requires_grad_(True)
            W_ref = W.detach().requires_grad_(True)
            b_ref = b.detach().requires_grad_(True)
            norm_weight_ref = norm_weight.detach().requires_grad_(True)
            norm_bias_ref = norm_bias.detach().requires_grad_(True)

            out_ref, W_last_ref, b_last_ref, loss_sum_ref, _ = video_ttt_scan_torch_reference(
                q_ref,
                k_ref,
                v_ref,
                W_ref,
                b_ref,
                norm_weight_ref,
                norm_bias_ref,
                lr=ctx.lr,
                mini_batch_size=ctx.mini_batch_size,
            )
            grads = torch.autograd.grad(
                outputs=(out_ref, W_last_ref, b_last_ref, loss_sum_ref),
                inputs=(q_ref, k_ref, v_ref, W_ref, b_ref, norm_weight_ref, norm_bias_ref),
                grad_outputs=(grad_out, grad_W_last, grad_b_last, grad_loss_sum),
                allow_unused=False,
            )
        return (*grads, None, None)


def video_ttt_scan_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    *,
    lr: float,
    mini_batch_size: int,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    if not triton_available() or q.device.type != "cuda":
        return None
    if q.dtype != torch.float32 or k.dtype != torch.float32 or v.dtype != torch.float32:
        return None
    if W.dtype != torch.float32 or b.dtype != torch.float32:
        return None
    if q.shape[-1] > 32:
        return None
    if q.shape[2] <= 0:
        return None
    out, W_last, b_last, loss_sum = _VideoTTTLinearScanTritonFn.apply(
        q,
        k,
        v,
        W,
        b,
        norm_weight,
        norm_bias,
        float(lr),
        int(mini_batch_size),
    )
    num_chunks = (q.shape[2] + int(mini_batch_size) - 1) // int(mini_batch_size)
    return out, W_last, b_last, loss_sum, int(num_chunks)
