import argparse
import copy
import time

import torch

from fastwam.models.wan22.video_ttt import VideoTTTQKVAdapter


def _sync():
    torch.cuda.synchronize()


def _ms(fn, *, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    _sync()
    return start.elapsed_time(end) / max(reps, 1)


def _clone_state(state):
    return {key: value.detach().clone().requires_grad_(value.requires_grad) for key, value in state.items()}


def _max_abs(a, b):
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return float("nan")
    return float((a.float() - b.float()).abs().max().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=3072)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--rank", type=int, default=192)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--layer-idx", type=int, default=3)
    parser.add_argument("--mini-batch", type=int, default=64)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--check-grad", action="store_true")
    parser.add_argument("--state-none", action="store_true")
    parser.add_argument("--scan-kernel", choices=("torch", "triton", "auto"), default="torch")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    dtype = torch.bfloat16
    adapter = VideoTTTQKVAdapter(
        hidden_dim=args.hidden,
        num_heads=args.heads,
        mini_batch_size=args.mini_batch,
        ttt_lr=0.5,
        residual_gate_init=0.02,
        num_layers=args.layers,
        rank=args.rank,
        inner_update_mode="scan",
        scan_kernel=args.scan_kernel,
        use_global_rope=True,
    ).to(device=device, dtype=dtype)
    adapter_ref = copy.deepcopy(adapter).to(device=device, dtype=dtype)
    adapter_ref.scan_kernel = "torch"

    tokens = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=dtype)
    base_time = torch.arange(args.batch, device=device, dtype=torch.float32).view(args.batch, 1)
    positions = {
        "time": base_time.expand(args.batch, args.seq),
        "height": torch.zeros(args.batch, args.seq, device=device),
        "width": torch.arange(args.seq, device=device, dtype=torch.float32).view(1, args.seq).expand(args.batch, -1),
    }
    state = adapter.init_state(args.batch, device=device, dtype=dtype)
    state_ref = adapter_ref.init_state(args.batch, device=device, dtype=dtype)

    def state_arg(state_value):
        return None if args.state_none else _clone_state(state_value)

    def eager_forward():
        return adapter_ref(
            tokens,
            state=state_arg(state_ref),
            update=True,
            update_tokens=None,
            positions=positions,
            layer_idx=args.layer_idx,
            compute_loss=True,
        )

    if args.compile:
        compiled_adapter = torch.compile(adapter, mode="reduce-overhead", fullgraph=False)

        def test_forward():
            return compiled_adapter(
                tokens,
                state=state_arg(state),
                update=True,
                update_tokens=None,
                positions=positions,
                layer_idx=args.layer_idx,
                compute_loss=True,
            )

        label = f"torch.compile scan_kernel={args.scan_kernel}"
    else:

        def test_forward(local_tokens=tokens):
            return adapter(
                local_tokens,
                state=state_arg(state),
                update=True,
                update_tokens=None,
                positions=positions,
                layer_idx=args.layer_idx,
                compute_loss=True,
            )

        label = f"eager scan_kernel={args.scan_kernel}"

    with torch.no_grad():
        out_ref, new_state_ref, loss_ref = eager_forward()
        out, new_state, loss = test_forward()
        print(f"variant={label}")
        print(f"shape batch={args.batch} seq={args.seq} hidden={args.hidden} heads={args.heads} rank={args.rank} mb={args.mini_batch}")
        print(f"max_abs_out={_max_abs(out_ref, out):.6e}")
        print(f"max_abs_W={_max_abs(new_state_ref['W'], new_state['W']):.6e}")
        print(f"max_abs_b={_max_abs(new_state_ref['b'], new_state['b']):.6e}")
        print(f"abs_loss={abs(float(loss_ref.item()) - float(loss.item())):.6e}")

    if args.check_grad:
        adapter_ref.zero_grad(set_to_none=True)
        adapter.zero_grad(set_to_none=True)
        tokens_ref = tokens.detach().clone().requires_grad_(True)
        tokens_test = tokens.detach().clone().requires_grad_(True)
        out_ref, new_state_ref, loss_ref = adapter_ref(
            tokens_ref,
            state=state_arg(state_ref),
            update=True,
            update_tokens=None,
            positions=positions,
            layer_idx=args.layer_idx,
            compute_loss=True,
        )
        if args.compile:
            out, new_state, loss = compiled_adapter(
                tokens_test,
                state=state_arg(state),
                update=True,
                update_tokens=None,
                positions=positions,
                layer_idx=args.layer_idx,
                compute_loss=True,
            )
        else:
            out, new_state, loss = test_forward(tokens_test)
        obj_ref = out_ref.float().square().mean() + loss_ref.float()
        obj = out.float().square().mean() + loss.float()
        obj_ref.backward()
        obj.backward()
        print(f"grad_max_abs_tokens={_max_abs(tokens_ref.grad, tokens_test.grad):.6e}")
        print(f"grad_max_abs_to_q={_max_abs(adapter_ref.to_q.weight.grad, adapter.to_q.weight.grad):.6e}")
        print(f"grad_max_abs_to_k={_max_abs(adapter_ref.to_k.weight.grad, adapter.to_k.weight.grad):.6e}")
        print(f"grad_max_abs_to_v={_max_abs(adapter_ref.to_v.weight.grad, adapter.to_v.weight.grad):.6e}")
        print(f"grad_max_abs_out_proj={_max_abs(adapter_ref.out_proj.weight.grad, adapter.out_proj.weight.grad):.6e}")
        print(f"grad_max_abs_W_init={_max_abs(adapter_ref.W_init.grad, adapter.W_init.grad):.6e}")
        print(f"grad_max_abs_gate={_max_abs(adapter_ref.residual_gate.grad, adapter.residual_gate.grad):.6e}")

    def forward_only():
        with torch.no_grad():
            out, new_state, loss = test_forward()
            # Make sure outputs are consumed.
            _ = out[0, 0, 0].float() + new_state["W"][0].float().sum() * 0.0 + loss.float()

    ms = _ms(forward_only, warmup=args.warmup, reps=args.reps)
    print(f"forward_only_ms={ms:.3f}")

    def forward_backward():
        adapter.zero_grad(set_to_none=True)
        local_tokens = tokens.detach().clone().requires_grad_(True)
        module = compiled_adapter if args.compile else adapter
        out, new_state, loss = module(
            local_tokens,
            state=state_arg(state),
            update=True,
            update_tokens=None,
            positions=positions,
            layer_idx=args.layer_idx,
            compute_loss=True,
        )
        objective = out.float().square().mean() + loss.float()
        objective.backward()

    ms_bwd = _ms(forward_backward, warmup=max(1, args.warmup // 2), reps=max(1, args.reps // 2))
    print(f"forward_backward_ms={ms_bwd:.3f}")

    print(f"max_memory_allocated_mb={torch.cuda.max_memory_allocated(device) / 1024 ** 2:.1f}")
    time.sleep(0.1)


if __name__ == "__main__":
    main()
