#!/usr/bin/env python3
import argparse
import os
import subprocess
import time

import torch


def _physical_gpu_index() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        first = visible.split(",")[0].strip()
        if first.isdigit():
            return int(first)
    return 0


def _other_compute_pids(physical_gpu: int) -> list[int]:
    cmd = [
        "nvidia-smi",
        f"--id={physical_gpu}",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    current_pid = os.getpid()
    pids: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line.split(",")[0].strip())
        except ValueError:
            continue
        if pid != current_pid:
            pids.append(pid)
    return pids


def _gpu_memory_used_mib(physical_gpu: int) -> int:
    cmd = [
        "nvidia-smi",
        f"--id={physical_gpu}",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 0
    try:
        return int(float(out.splitlines()[0].strip()))
    except (IndexError, ValueError):
        return 0


def _gpu_utilization_pct(physical_gpu: int) -> int:
    cmd = [
        "nvidia-smi",
        f"--id={physical_gpu}",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 0
    try:
        return int(float(out.splitlines()[0].strip()))
    except (IndexError, ValueError):
        return 0


def _release_memory() -> None:
    allocate_memory.tensors = []
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def allocate_memory(fraction: float, chunk_gib: float) -> float:
    device = torch.device("cuda:0")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    used_bytes = max(total_bytes - free_bytes, 0)
    target_total_used_bytes = int(total_bytes * fraction)
    target_bytes = int(max(target_total_used_bytes - used_bytes, 0))
    target_bytes = int(min(target_bytes, free_bytes * 0.92))
    chunk_bytes = int(chunk_gib * 1024**3)
    tensors = []
    allocated = 0

    while allocated < target_bytes:
        next_bytes = min(chunk_bytes, target_bytes - allocated)
        if next_bytes <= 0:
            break
        try:
            tensors.append(torch.empty((next_bytes // 2,), dtype=torch.float16, device=device))
            allocated += next_bytes
        except RuntimeError:
            if next_bytes <= 256 * 1024**2:
                raise
            chunk_bytes = max(next_bytes // 2, 256 * 1024**2)

    # Keep the tensors alive for the lifetime of the process.
    allocate_memory.tensors = tensors
    return allocated / 1024**3


def create_matmul_tensors(size: int, batches: int):
    device = torch.device("cuda:0")
    current_batches = max(int(batches), 1)
    while current_batches >= 1:
        a = []
        b = []
        c = []
        try:
            for _ in range(current_batches):
                a.append(torch.randn((size, size), device=device, dtype=torch.float16))
                b.append(torch.randn((size, size), device=device, dtype=torch.float16))
                c.append(torch.empty((size, size), device=device, dtype=torch.float16))
            return a, b, c, current_batches
        except RuntimeError:
            del a, b, c
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if current_batches == 1:
                raise
            current_batches = max(current_batches // 2, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hold a CUDA device only while it is otherwise idle."
    )
    parser.add_argument("--fraction", type=float, default=0.70)
    parser.add_argument("--chunk-gib", type=float, default=1.0)
    parser.add_argument("--matmul-size", type=int, default=2048)
    parser.add_argument(
        "--matmul-batches",
        type=int,
        default=4,
        help="Number of independent matmuls submitted per loop to keep large GPUs busy.",
    )
    parser.add_argument(
        "--sync-every",
        type=int,
        default=4,
        help="Submit this many matmul batches before synchronizing and checking for release.",
    )
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--idle-seconds", type=float, default=30.0)
    parser.add_argument(
        "--busy-memory-threshold-mib",
        type=int,
        default=4096,
        help=(
            "Treat the GPU as busy before holding when memory.used exceeds this. "
            "This avoids PID namespace issues where nvidia-smi reports host PIDs."
        ),
    )
    parser.add_argument(
        "--busy-util-threshold",
        type=int,
        default=20,
        help="Yield when GPU util is above this before holding.",
    )
    parser.add_argument(
        "--allow-memory-only-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If memory is already occupied but GPU util is low, still use remaining "
            "capacity for compute pressure. Disable to yield on memory alone."
        ),
    )
    parser.add_argument(
        "--release-extra-memory-mib",
        type=int,
        default=4096,
        help="Release while holding if total GPU memory rises this much above our held footprint.",
    )
    parser.add_argument(
        "--probe-release-seconds",
        type=float,
        default=300.0,
        help="Release the GPU periodically so a queued real job can start.",
    )
    parser.add_argument(
        "--probe-window-seconds",
        type=float,
        default=15.0,
        help="How long to stay released during each probe window.",
    )
    parser.add_argument(
        "--physical-gpu",
        type=int,
        default=None,
        help="Physical nvidia-smi GPU index. Defaults to the first CUDA_VISIBLE_DEVICES entry.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(0)
    physical_gpu = _physical_gpu_index() if args.physical_gpu is None else int(args.physical_gpu)

    size = args.matmul_size
    matmul_batches = max(int(args.matmul_batches), 1)
    sync_every = max(int(args.sync_every), 1)
    print(
        f"[gpu-hold] pid={os.getpid()} visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"physical_gpu={physical_gpu} fraction={args.fraction} matmul_size={size} "
        f"matmul_batches={matmul_batches} sync_every={sync_every} "
        f"sleep={args.sleep} poll={args.poll_seconds} "
        f"busy_memory_threshold_mib={args.busy_memory_threshold_mib} "
        f"busy_util_threshold={args.busy_util_threshold} "
        f"allow_memory_only_hold={args.allow_memory_only_hold}",
        flush=True,
    )

    while True:
        idle_started = None
        while True:
            used_mib = _gpu_memory_used_mib(physical_gpu)
            util_pct = _gpu_utilization_pct(physical_gpu)
            pids = _other_compute_pids(physical_gpu)
            memory_busy = used_mib > int(args.busy_memory_threshold_mib)
            util_busy = util_pct > int(args.busy_util_threshold)
            if util_busy or (memory_busy and not bool(args.allow_memory_only_hold)):
                idle_started = None
                print(
                    f"[gpu-hold] yielding: memory_used_mib={used_mib} "
                    f"util_pct={util_pct} pids={pids}",
                    flush=True,
                )
                time.sleep(args.poll_seconds)
                continue
            now = time.monotonic()
            idle_started = now if idle_started is None else idle_started
            if now - idle_started >= args.idle_seconds:
                break
            time.sleep(args.poll_seconds)

        a, b, c, actual_batches = create_matmul_tensors(size, matmul_batches)
        allocated_gib = allocate_memory(args.fraction, args.chunk_gib)
        held_memory_mib = _gpu_memory_used_mib(physical_gpu)
        print(
            f"[gpu-hold] holding allocated_gib={allocated_gib:.1f} "
            f"memory_used_mib={held_memory_mib} matmul_batches={actual_batches}",
            flush=True,
        )

        last_probe = time.monotonic()
        while True:
            for _ in range(sync_every):
                for idx in range(actual_batches):
                    torch.mm(a[idx], b[idx], out=c[idx])
            torch.cuda.synchronize()
            time.sleep(args.sleep)

            used_mib = _gpu_memory_used_mib(physical_gpu)
            now = time.monotonic()
            should_probe = (
                args.probe_release_seconds > 0
                and now - last_probe >= args.probe_release_seconds
            )
            should_release_for_memory = (
                used_mib
                > int(held_memory_mib) + int(args.release_extra_memory_mib)
            )
            if should_release_for_memory or should_probe:
                del a, b, c
                _release_memory()
                if should_release_for_memory:
                    pids = _other_compute_pids(physical_gpu)
                    print(
                        f"[gpu-hold] released: memory_used_mib={used_mib} "
                        f"held_memory_mib={held_memory_mib} pids={pids}",
                        flush=True,
                    )
                else:
                    print("[gpu-hold] released for probe window", flush=True)
                    time.sleep(args.probe_window_seconds)
                break


if __name__ == "__main__":
    main()
