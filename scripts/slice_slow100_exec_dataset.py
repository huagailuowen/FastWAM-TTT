#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INDEX_COLUMNS = [
    "coarse_task_index",
    "task_index",
    "coarse_quality_index",
    "quality_index",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tasks(path: Path) -> dict[int, str]:
    return {int(row["task_index"]): str(row["task"]) for row in load_jsonl(path)}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def column_matrix(series: pd.Series) -> np.ndarray:
    first = series.iloc[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return np.stack([np.asarray(x) for x in series.to_list()], axis=0)
    return np.asarray(series.to_numpy())[:, None]


def compute_stats(df: pd.DataFrame, feature_names: list[str]) -> dict[str, dict[str, list[float] | list[int]]]:
    stats: dict[str, dict[str, list[float] | list[int]]] = {}
    for name in feature_names:
        if name not in df.columns:
            continue
        values = column_matrix(df[name])
        if np.issubdtype(values.dtype, np.integer):
            values_for_stats = values.astype(np.float64)
            min_values = values.min(axis=0).astype(np.int64).tolist()
            max_values = values.max(axis=0).astype(np.int64).tolist()
        else:
            values_for_stats = values.astype(np.float64)
            min_values = values_for_stats.min(axis=0).astype(np.float64).tolist()
            max_values = values_for_stats.max(axis=0).astype(np.float64).tolist()
        stats[name] = {
            "min": min_values,
            "max": max_values,
            "mean": values_for_stats.mean(axis=0).astype(np.float64).tolist(),
            "std": values_for_stats.std(axis=0).astype(np.float64).tolist(),
            "count": [int(values.shape[0])],
        }
    return stats


def ffmpeg_slice_video(src: Path, dst: Path, start_frame: int, fps: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    expr = f"select='gte(n,{int(start_frame)})',setpts=N/FRAME_RATE/TB"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vf",
        expr,
        "-an",
        "-r",
        str(int(fps)),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def ffprobe_count_frames(path: Path) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return int(out.splitlines()[0])


def build_task_map(dataset_roots: list[Path]) -> tuple[dict[str, int], dict[tuple[Path, int], int]]:
    task_to_new: dict[str, int] = {}
    remap: dict[tuple[Path, int], int] = {}
    for root in dataset_roots:
        old_tasks = load_tasks(root / "meta" / "tasks.jsonl")
        for old_idx, task in old_tasks.items():
            if task not in task_to_new:
                task_to_new[task] = len(task_to_new)
            remap[(root, old_idx)] = task_to_new[task]
    return task_to_new, remap


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice slow100 dynamic-carrier demos from execution frame onward.")
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-frame", type=int, default=100)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--min-length", type=int, default=34)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_roots = [Path(p).resolve() for p in args.input_dirs]
    output_root = Path(args.output_dir).resolve()
    start_frame = int(args.start_frame)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    (output_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    first_info = json.loads((input_roots[0] / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = int(first_info["fps"])
    video_keys = [
        key
        for key, feature in first_info["features"].items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    feature_names = [name for name in first_info["features"].keys() if first_info["features"][name].get("dtype") != "video"]
    task_to_new, task_remap = build_task_map(input_roots)

    task_rows = [
        {"task_index": idx, "task": task}
        for task, idx in sorted(task_to_new.items(), key=lambda item: item[1])
    ]
    write_jsonl(output_root / "meta" / "tasks.jsonl", task_rows)

    episodes_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    metadata_successes: list[dict[str, Any]] = []
    metadata_failures: list[dict[str, Any]] = []
    output_episode = 0
    global_index = 0

    for input_root in input_roots:
        meta_payload = json.loads((input_root / "dynamic_carrier_generation_metadata.json").read_text(encoding="utf-8"))
        success_by_episode = {
            int(item["episode_index"]): item
            for item in meta_payload.get("successes", [])
            if "episode_index" in item
        }
        for ep_row in load_jsonl(input_root / "meta" / "episodes.jsonl"):
            if args.max_episodes is not None and output_episode >= int(args.max_episodes):
                break
            old_episode = int(ep_row["episode_index"])
            src_parquet = input_root / "data" / f"chunk-{old_episode // int(first_info['chunks_size']):03d}" / f"episode_{old_episode:06d}.parquet"
            df = pd.read_parquet(src_parquet)
            sliced = df.iloc[start_frame:].copy().reset_index(drop=True)
            if len(sliced) < int(args.min_length):
                metadata_failures.append(
                    {
                        "source_dataset": str(input_root),
                        "source_episode_index": old_episode,
                        "reason": f"sliced length {len(sliced)} < min_length {args.min_length}",
                    }
                )
                continue
            for col in INDEX_COLUMNS:
                if col in sliced.columns:
                    sliced[col] = sliced[col].map(lambda old: task_remap[(input_root, int(old))]).astype(np.int64)
            length = int(len(sliced))
            sliced["frame_index"] = np.arange(length, dtype=np.int64)
            sliced["timestamp"] = (np.arange(length, dtype=np.float32) / float(fps)).astype(np.float32)
            sliced["episode_index"] = np.full(length, output_episode, dtype=np.int64)
            sliced["index"] = np.arange(global_index, global_index + length, dtype=np.int64)

            dst_parquet = output_root / "data" / "chunk-000" / f"episode_{output_episode:06d}.parquet"
            sliced.to_parquet(dst_parquet, index=False)

            for video_key in video_keys:
                src_video = input_root / "videos" / f"chunk-{old_episode // int(first_info['chunks_size']):03d}" / video_key / f"episode_{old_episode:06d}.mp4"
                dst_video = output_root / "videos" / "chunk-000" / video_key / f"episode_{output_episode:06d}.mp4"
                ffmpeg_slice_video(src_video, dst_video, start_frame=start_frame, fps=fps)
                frame_count = ffprobe_count_frames(dst_video)
                if frame_count != length:
                    raise RuntimeError(
                        f"Video frame count mismatch for {dst_video}: got {frame_count}, expected {length}."
                    )

            tasks = []
            for col in INDEX_COLUMNS:
                if col in sliced.columns:
                    new_idx = int(sliced[col].iloc[0])
                    tasks.append(task_rows[new_idx]["task"])
            episodes_rows.append({"episode_index": output_episode, "tasks": tasks, "length": length})
            stats_rows.append({"episode_index": output_episode, "stats": to_jsonable(compute_stats(sliced, feature_names))})

            success = dict(success_by_episode.get(old_episode, {}))
            success.update(
                {
                    "episode_index": output_episode,
                    "source_dataset": str(input_root),
                    "source_episode_index": old_episode,
                    "source_start_frame": start_frame,
                    "sliced_length": length,
                    "observe_frames": 0,
                    "action_start_frame": start_frame,
                    "execution_start_frame": start_frame,
                }
            )
            metadata_successes.append(success)
            output_episode += 1
            global_index += length
        if args.max_episodes is not None and output_episode >= int(args.max_episodes):
            break

    write_jsonl(output_root / "meta" / "episodes.jsonl", episodes_rows)
    write_jsonl(output_root / "meta" / "episodes_stats.jsonl", stats_rows)

    info = dict(first_info)
    info["total_episodes"] = int(output_episode)
    info["total_frames"] = int(global_index)
    info["total_tasks"] = int(len(task_rows))
    info["total_videos"] = int(output_episode * len(video_keys))
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{output_episode}"}
    info["chunks_size"] = max(int(first_info.get("chunks_size", 1000)), int(output_episode), 1)
    (output_root / "meta" / "info.json").write_text(json.dumps(to_jsonable(info), indent=2), encoding="utf-8")

    metadata = {
        "dataset_type": "ttt_dynamic_carrier_lerobot",
        "ttt_mode": "slow100_execution_sliced",
        "created_from": [str(root) for root in input_roots],
        "source_start_frame": start_frame,
        "speed_multiplier": 0.8,
        "episodes_collected": int(output_episode),
        "episodes_requested": int(output_episode),
        "observe_frames": 0,
        "action_start_frame": start_frame,
        "execution_start_frame": start_frame,
        "successes": metadata_successes,
        "failures": metadata_failures,
    }
    (output_root / "dynamic_carrier_generation_metadata.json").write_text(
        json.dumps(to_jsonable(metadata), indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_episode} episodes, {global_index} frames to {output_root}")


if __name__ == "__main__":
    main()
