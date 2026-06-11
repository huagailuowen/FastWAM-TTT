#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
TTT_ROOT = REPO_ROOT.parent / "TTT4dynamics"
sys.path.insert(0, str(TTT_ROOT))

from ttt4dynamics.cases import DynamicCarrierCase, load_cases  # noqa: E402
from ttt4dynamics.trajectories import ProceduralTrajectory  # noqa: E402


FAMILIES = ("fast_period", "slow_period", "large_amplitude", "high_yaw", "novel_shape")


def _prompt_for_case(case: DynamicCarrierCase) -> str:
    if "box" in case.access_mode.lower() or "tray" in case.access_mode.lower():
        return "track the moving cream cheese box inside the open box and place it on the target region"
    return "track the moving cream cheese box on the platform and place it on the target region"


def _training_summary(slow_metadata_paths: list[Path]) -> dict[str, Any]:
    periods: list[float] = []
    amp_x: list[float] = []
    amp_y: list[float] = []
    yaw: list[float] = []
    for path in slow_metadata_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("successes", []):
            motion = item["case"]["motion"]
            periods.append(float(motion["period"]))
            amp_x.append(float(motion["amplitude"][0]))
            amp_y.append(float(motion["amplitude"][1]))
            yaw.append(float(motion.get("yaw", 0.0)))
    return {
        "period_range": [min(periods), max(periods)],
        "amp_x_range": [min(amp_x), max(amp_x)],
        "amp_y_range": [min(amp_y), max(amp_y)],
        "yaw_range": [min(yaw), max(yaw)],
        "num_training_successes_scanned": len(periods),
    }


def _case_distance(case: DynamicCarrierCase) -> float:
    return ProceduralTrajectory(case.motion).min_distance_to_point(case.target_xy)


def _try_case(case: DynamicCarrierCase) -> tuple[DynamicCarrierCase, float] | None:
    for shift_idx, (dx, dy) in enumerate(
        [
            (0.0, 0.0),
            (-0.02, 0.0),
            (-0.04, 0.0),
            (-0.04, -0.02),
            (-0.04, 0.02),
            (-0.06, 0.0),
            (-0.06, -0.02),
            (-0.06, 0.02),
        ]
    ):
        motion = case.motion
        if shift_idx > 0:
            motion = replace(motion, center=(float(motion.center[0] + dx), float(motion.center[1] + dy)))
        shifted = replace(case, motion=motion)
        try:
            return shifted, shifted.validate_target_separation()
        except ValueError:
            continue
    return None


def _sample_case(
    *,
    base: DynamicCarrierCase,
    family: str,
    rng: np.random.Generator,
    index: int,
) -> tuple[DynamicCarrierCase, float]:
    for _ in range(400):
        base_motion = base.motion
        base_family = base_motion.family.lower()
        direction = int(rng.choice([-1, 1]))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        yaw = float(rng.uniform(-0.30, 0.30))
        harmonics: list[dict[str, float]] = []
        motion_family = base_family
        amp_x = float(rng.uniform(0.082, 0.116))
        amp_y = 0.0 if base_family == "line" else float(rng.uniform(0.045, 0.078))
        period = float(rng.uniform(3.35, 5.20))

        if family == "fast_period":
            period = float(rng.uniform(2.15, 2.90))
            if base_family == "irregular_loop":
                harmonics = _sample_harmonics(rng)
        elif family == "slow_period":
            period = float(rng.uniform(5.70, 7.10))
            if base_family == "irregular_loop":
                harmonics = _sample_harmonics(rng)
        elif family == "large_amplitude":
            amp_x = float(rng.uniform(0.132, 0.162))
            amp_y = 0.0 if base_family == "line" else float(rng.uniform(0.092, 0.122))
            if base_family == "irregular_loop":
                harmonics = _sample_harmonics(rng)
        elif family == "high_yaw":
            yaw = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.52, 0.74))
            if base_family == "irregular_loop":
                harmonics = _sample_harmonics(rng)
        elif family == "novel_shape":
            motion_family = str(rng.choice(["ellipse", "figure8", "lissajous"]))
            amp_x = float(rng.uniform(0.087, 0.116))
            amp_y = float(rng.uniform(0.048, 0.080))
            harmonics = []
        else:
            raise ValueError(f"Unknown OOD family: {family}")

        motion = replace(
            base_motion,
            family=motion_family,
            amplitude=(amp_x, amp_y),
            period=period,
            phase=phase,
            direction=direction,
            yaw=yaw,
            phase_y=float(rng.uniform(0.5, 1.6)),
            harmonics=harmonics,
        )
        case = replace(
            base,
            case_id=f"{base.case_id}_slow100_ood_{family}_v{index:04d}",
            motion=motion,
        )
        valid = _try_case(case)
        if valid is not None:
            return valid
    raise RuntimeError(f"Could not sample valid slow100 OOD case for {base.case_id}/{family}")


def _sample_harmonics(rng: np.random.Generator) -> list[dict[str, float]]:
    return [
        {
            "order": int(rng.choice([2, 3, 4])),
            "x": float(rng.uniform(0.04, 0.16)),
            "y": float(rng.uniform(0.04, 0.16)),
            "phase": float(rng.uniform(-math.pi, math.pi)),
        }
        for _ in range(int(rng.integers(1, 3)))
    ]


def _make_entry(
    *,
    case: DynamicCarrierCase,
    distance: float,
    family: str,
    seed: int,
    episode_index: int,
) -> dict[str, Any]:
    return {
        "episode_index": int(episode_index),
        "seed": int(seed),
        "steps": int(case.max_steps),
        "success": True,
        "ood_family": family,
        "target_path_separation": float(distance),
        "task_description": _prompt_for_case(case),
        "observe_frames": 100,
        "action_start_frame": 100,
        "execution_start_frame": 100,
        "case": case.as_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate slow100-relative OOD eval metadata.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--per-family", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    base_cases = load_cases(TTT_ROOT / "configs/dynamic_carrier_cases.json")
    slow_metadata_paths = sorted(
        REPO_ROOT.glob("data/ttt_dynamic_carrier_slow100_observe10_shard*_lerobot/dynamic_carrier_generation_metadata.json")
    )
    if not slow_metadata_paths:
        raise FileNotFoundError("Could not find slow100 training metadata shards.")

    full_entries: list[dict[str, Any]] = []
    global_idx = 0
    for family_idx, family in enumerate(FAMILIES):
        for _ in range(int(args.per_family)):
            base = base_cases[global_idx % len(base_cases)]
            case, distance = _sample_case(base=base, family=family, rng=rng, index=global_idx)
            full_entries.append(
                _make_entry(
                    case=case,
                    distance=distance,
                    family=family,
                    seed=int(args.seed) + global_idx,
                    episode_index=global_idx,
                )
            )
            global_idx += 1

    sample_indices = rng.choice(len(full_entries), size=int(args.sample_count), replace=False).tolist()
    sample_entries = [full_entries[int(i)] for i in sample_indices]
    for new_idx, entry in enumerate(sample_entries):
        entry["source_index"] = int(sample_indices[new_idx])
        entry["episode_index"] = int(new_idx)

    payload = {
        "dataset_type": "ttt_dynamic_carrier_lerobot",
        "ttt_mode": "slow100_ood_eval",
        "seed": int(args.seed),
        "speed_multiplier": 0.8,
        "execution_start_frame": 100,
        "action_start_frame": 100,
        "observe_frames": 100,
        "observe_chunks": 10,
        "chunk_interval": 10,
        "camera_resolution": 224,
        "episodes_requested": int(args.sample_count),
        "successes": sample_entries,
        "failures": [],
        "base_cases": [case.as_dict() for case in base_cases],
        "ood_design": {
            "relative_to": "slow100 training metadata",
            "training_summary": _training_summary(slow_metadata_paths),
            "families": {
                "fast_period": "period sampled from [2.15, 2.90] seconds, below slow100 train min",
                "slow_period": "period sampled from [5.70, 7.10] seconds, above slow100 train max",
                "large_amplitude": "amp_x [0.132, 0.162], loop amp_y [0.092, 0.122]",
                "high_yaw": "absolute yaw [0.52, 0.74] rad",
                "novel_shape": "ellipse, figure8, or lissajous at slow100-like periods",
            },
            "full_source_count": len(full_entries),
            "sample_indices": sample_indices,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full_payload = dict(payload)
    full_payload["episodes_requested"] = len(full_entries)
    full_payload["successes"] = full_entries
    full_path = args.output_dir / "dynamic_carrier_slow100_ood_full50_seed20260610_metadata.json"
    sample_path = args.output_dir / "dynamic_carrier_slow100_ood_random20_seed20260610_metadata.json"
    full_path.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")
    sample_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote full metadata: {full_path}")
    print(f"Wrote random20 metadata: {sample_path}")
    print(f"Selected source indices: {sample_indices}")


if __name__ == "__main__":
    main()
