#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TTT_ROOT = PROJECT_ROOT.parent / "TTT4dynamics"
LIBERO_CONFIG_PATH = PROJECT_ROOT.parent / "LIBERO" / ".libero_config"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_CONFIG_PATH))

for path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "experiments" / "libero",
    TTT_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_image,
    invert_gripper_action,
    save_prediction_video,
    save_rollout_video,
)
from experiments.libero.eval_libero_single import (  # noqa: E402
    NumpyEncoder,
    _center_crop_resize,
    _compute_clip_mean_psnr,
    _denormalize_action,
    _get_future_frame_capture_steps,
    _get_num_video_frames,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _select_predicted_future_frames,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402
from ttt4dynamics.cases import DynamicCarrierCase  # noqa: E402
from ttt4dynamics.dynamic_env import DynamicCarrierEnv, create_libero_env_for_case  # noqa: E402
from ttt4dynamics.planner import PlannerConfig, ScriptedDynamicCarrierPlanner  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)], replace=True)


FLAT_PROMPT = (
    "track the moving cream cheese box on the platform, pick it up, "
    "and place it on the static target region"
)
BOX_PROMPT = (
    "track the moving cream cheese box inside the open tray, pick it from above, "
    "and place it on the static target region"
)


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_like))))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(_resolve_path(explicit))

    ckpt = _resolve_path(cfg.ckpt)
    for parent in list(ckpt.parents)[:5]:
        candidates.append(parent / "dataset_stats.json")

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate dataset_stats.json. Pass "
        "+EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _quat_to_axisangle_wxyz(quat: np.ndarray) -> np.ndarray:
    """Match the state extraction used by the dynamic-carrier dataset collector."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat.shape[0] != 4:
        raise ValueError(f"Expected quaternion with shape (4,), got {quat.shape}")
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    quat /= norm
    if quat[0] < 0.0:
        quat *= -1.0
    w = float(np.clip(quat[0], -1.0, 1.0))
    xyz = quat[1:4]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(sin_half, w)
    axis = xyz / sin_half
    return (axis * angle).astype(np.float32)


def _extract_dynamic_state(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat_to_axisangle_wxyz(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def _normalize_proprio(proprio: np.ndarray, processor: FastWAMProcessor) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError("Expected a single merged state key in shape_meta['state'].")
    state_key = state_meta[0]["key"]
    batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    batch = processor.action_state_transform(batch)
    batch = processor.normalizer.forward(batch)
    return batch["state"][state_key]


def _obs_to_model_input(
    obs: dict[str, Any],
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, np.ndarray]]:
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_hw(meta: dict[str, Any], camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    if int(processor.num_output_cameras) == 1:
        primary_h, primary_w = _meta_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif int(processor.num_output_cameras) == 2:
        primary_h, primary_w = _meta_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"Expected one or two cameras, got {processor.num_output_cameras}.")

    if rgb.shape[:2] != (int(height), int(width)):
        raise ValueError(
            f"Input image size mismatch: got {rgb.shape[:2]}, expected {(int(height), int(width))}."
        )

    image = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    image = image * (2.0 / 255.0) - 1.0
    proprio = _normalize_proprio(_extract_dynamic_state(obs), processor)
    return image, proprio, imgs


def _prompt_for_case(case: DynamicCarrierCase, cfg: DictConfig) -> str:
    prompt_override = cfg.EVALUATION.get("prompt_override")
    if prompt_override:
        return str(prompt_override)
    if "box" in case.access_mode.lower() or "tray" in case.access_mode.lower():
        return BOX_PROMPT
    return FLAT_PROMPT


def _load_eval_cases(cfg: DictConfig) -> list[tuple[DynamicCarrierCase, int, int]]:
    metadata_path = _resolve_path(
        cfg.EVALUATION.get(
            "dynamic_metadata_path",
            "./data/ttt_dynamic_carrier_cream_200_lerobot/dynamic_carrier_generation_metadata.json",
        )
    )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    successes = payload.get("successes", [])
    if not isinstance(successes, list) or len(successes) == 0:
        raise ValueError(f"No successful cases found in {metadata_path}")

    start = int(cfg.EVALUATION.get("case_start", 0))
    num_trials = int(cfg.EVALUATION.num_trials)
    selected = successes[start : start + num_trials]
    if len(selected) < num_trials:
        repeats = (num_trials - len(selected))
        selected.extend(successes[:repeats])

    cases = []
    for fallback_idx, item in enumerate(selected):
        cases.append(
            (
                DynamicCarrierCase.from_dict(item["case"]),
                int(item.get("seed", payload.get("seed", 0) + fallback_idx)),
                int(item.get("episode_index", start + fallback_idx)),
            )
        )
    return cases


def _predict_action_chunk(
    obs: dict[str, Any],
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], Optional[list[Image.Image]]]:
    num_inference_steps = int(
        cfg.EVALUATION.get("num_inference_steps", cfg.get("eval_num_inference_steps", 20))
    )
    prompt = DEFAULT_PROMPT.format(task=task_description)
    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    seed = None if cfg.get("seed") is None else int(cfg.seed)
    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": seed,
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }

    predicted_future_frames = None
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
        with torch.no_grad():
            pred = model.infer_joint(**infer_kwargs)
        predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
    else:
        with torch.no_grad():
            pred = model.infer_action(**infer_kwargs)

    action = _denormalize_action(pred["action"], processor)[0]
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", True)):
        action[..., -1] = np.sign(action[..., -1])
    return action, imgs, predicted_future_frames


def _save_action_trace(trace_dir: Path, episode_result: dict[str, Any]) -> str:
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / f"episode_{episode_result['trial_idx']:04d}_actions.npz"
    np.savez_compressed(
        path,
        actions=np.asarray(episode_result["actions"], dtype=np.float32),
        eef_xyz=np.asarray(episode_result["eef_xyz"], dtype=np.float32),
        payload_xyz=np.asarray(episode_result["payload_xyz"], dtype=np.float32),
        carrier_xyz=np.asarray(episode_result["carrier_xyz"], dtype=np.float32),
    )
    return str(path)


def run_episode(
    *,
    case: DynamicCarrierCase,
    seed: int,
    dataset_episode_index: int,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    trial_idx: int,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    reset_ttt_state: bool = True,
) -> tuple[dict[str, Any], list[Any], list[dict[str, Any]], Optional[float]]:
    base_env, init_state, _ = create_libero_env_for_case(
        case,
        repo_root=TTT_ROOT,
        camera_resolution=int(cfg.EVALUATION.get("camera_resolution", 224)),
        seed=seed,
    )
    env = DynamicCarrierEnv(base_env, case)
    task_description = _prompt_for_case(case, cfg)
    max_steps = int(cfg.EVALUATION.get("max_steps", case.max_steps))
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 10))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 0))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    if reset_ttt_state and hasattr(model, "reset_video_ttt_state"):
        model.reset_video_ttt_state()

    replay_images: list[Any] = []
    pending_actions: list[list[float]] = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    future_clip_psnr: list[float] = []
    current_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1
    action_trace: list[list[float]] = []
    eef_trace: list[list[float]] = []
    payload_trace: list[list[float]] = []
    carrier_trace: list[list[float]] = []

    try:
        obs = env.reset(init_state=init_state)
        done = False
        success = False
        pbar = tqdm(total=max_steps + num_steps_wait, desc=f"trial {trial_idx + 1}")
        for t in range(max_steps + num_steps_wait):
            pbar.update(1)
            if t < num_steps_wait:
                action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
                obs, _, done, _ = env.step(action)
                action_trace.append(action.tolist())
                continue

            if len(pending_actions) == 0:
                action_chunk, imgs, predicted_future_frames = _predict_action_chunk(
                    obs=obs,
                    task_description=task_description,
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    action_horizon=action_horizon,
                    input_w=input_w,
                    input_h=input_h,
                    model_device=model_device,
                )
                replay_images.append(imgs.copy())
                if predicted_future_frames is not None:
                    current_replan_idx += 1
                    current_clip = {
                        "replan_idx": current_replan_idx,
                        "gt_frames": [imgs.copy()],
                        "pred_frames": predicted_future_frames,
                    }
                else:
                    current_clip = None
                current_replan_step = 0
                pending_actions = action_chunk[:replan_steps].tolist()
            else:
                replay_images.append(get_libero_image(obs).copy())

            action = np.asarray(pending_actions.pop(0), dtype=np.float32)
            eef_trace.append(env.eef_position(obs).astype(float).tolist())
            payload_trace.append(env.payload_position().astype(float).tolist())
            carrier_trace.append(env.carrier_position().astype(float).tolist())
            action_trace.append(action.astype(float).tolist())
            obs, _, done, _ = env.step(action)

            if visualize_future_video and current_clip is not None:
                current_replan_step += 1
                if current_replan_step in capture_steps:
                    current_clip["gt_frames"].append(get_libero_image(obs))
                if done or len(pending_actions) == 0:
                    expected = 1 + sum(1 for step in capture_steps if step <= current_replan_step)
                    current_clip["pred_frames"] = current_clip["pred_frames"][:expected]
                    current_clip["gt_frames"] = current_clip["gt_frames"][:expected]
                    if len(current_clip["gt_frames"]) == len(current_clip["pred_frames"]):
                        clip_psnr = _compute_clip_mean_psnr(
                            current_clip["gt_frames"], current_clip["pred_frames"]
                        )
                        if clip_psnr is not None:
                            future_clip_psnr.append(float(clip_psnr))
                    predicted_future_video_clips.append(current_clip)
                    current_clip = None

            success = bool(env.check_success())
            if done or success:
                success = bool(env.check_success())
                break
        pbar.close()

        result = {
            "trial_idx": int(trial_idx),
            "dataset_episode_index": int(dataset_episode_index),
            "case_id": case.case_id,
            "access_mode": case.access_mode,
            "trajectory_family": case.motion.family,
            "seed": int(seed),
            "task_description": task_description,
            "success": bool(success),
            "steps": int(len(action_trace)),
            "actions": action_trace,
            "eef_xyz": eef_trace,
            "payload_xyz": payload_trace,
            "carrier_xyz": carrier_trace,
        }
        mean_psnr = float(np.mean(future_clip_psnr)) if len(future_clip_psnr) > 0 else None
        return result, replay_images, predicted_future_video_clips, mean_psnr
    finally:
        env.close()


def run_scripted_ttt_warmup_pass(
    *,
    case: DynamicCarrierCase,
    seed: int,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    trial_idx: int,
    pass_idx: int,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict[str, Any]:
    """Run one guided demo pass while preserving the model's TTT fast state.

    The scripted planner supplies the executed actions. Model actions are
    requested only at the configured interval so `infer_action` applies the same
    observation-frame video TTT update used during normal inference.
    """
    base_env, init_state, _ = create_libero_env_for_case(
        case,
        repo_root=TTT_ROOT,
        camera_resolution=int(cfg.EVALUATION.get("camera_resolution", 224)),
        seed=seed,
    )
    env = DynamicCarrierEnv(base_env, case)
    planner = ScriptedDynamicCarrierPlanner(env, PlannerConfig())
    task_description = _prompt_for_case(case, cfg)
    warmup_max_steps = cfg.EVALUATION.get("warmup_max_steps", None)
    if warmup_max_steps is None:
        warmup_max_steps = cfg.EVALUATION.get("max_steps", case.max_steps)
    max_steps = int(warmup_max_steps)
    update_interval = int(cfg.EVALUATION.get("warmup_update_interval", cfg.EVALUATION.get("replan_steps", 10)))
    if update_interval <= 0:
        raise ValueError(f"EVALUATION.warmup_update_interval must be positive, got {update_interval}.")
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 0))

    try:
        obs = env.reset(init_state=init_state)
        planner.reset()
        done = False
        success = False
        steps = 0
        ttt_updates = 0
        pbar = tqdm(
            total=max_steps + num_steps_wait,
            desc=f"trial {trial_idx + 1} warmup {pass_idx + 1}",
            leave=False,
        )
        for t in range(max_steps + num_steps_wait):
            pbar.update(1)
            if t < num_steps_wait:
                action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
            else:
                if (t - num_steps_wait) % update_interval == 0:
                    _predict_action_chunk(
                        obs=obs,
                        task_description=task_description,
                        model=model,
                        processor=processor,
                        cfg=cfg,
                        action_horizon=action_horizon,
                        input_w=input_w,
                        input_h=input_h,
                        model_device=model_device,
                    )
                    ttt_updates += 1
                action = planner.act(obs).astype(np.float32)

            obs, _, done, _ = env.step(action)
            steps += 1
            if t >= num_steps_wait:
                success = bool(env.check_success())
                if done or success or planner.is_done():
                    success = bool(env.check_success())
                    break
        pbar.close()

        return {
            "pass_idx": int(pass_idx),
            "success": bool(success),
            "steps": int(steps),
            "ttt_updates": int(ttt_updates),
            "final_phase": str(planner.phase.value),
        }
    finally:
        env.close()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig) -> dict[str, Any]:
    start_time = time.time()
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")

    device = str(cfg.EVALUATION.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    _load_model_checkpoint(model, str(_resolve_path(cfg.ckpt)))
    model = model.to(device).eval()

    stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = (
        int(cfg.data.train.num_frames) - 1 if action_horizon_cfg is None else int(action_horizon_cfg)
    )
    video_size = cfg.data.train.get("video_size", [224, 448])
    input_h, input_w = int(video_size[0]), int(video_size[1])

    cases = _load_eval_cases(cfg)
    output_dir = _resolve_path(cfg.EVALUATION.output_dir)
    run_name = str(cfg.EVALUATION.get("run_name", "dynamic_carrier"))
    run_dir = output_dir / run_name
    video_dir = run_dir / "rollout_videos"
    predicted_video_dir = run_dir / "predicted_videos"
    action_trace_dir = run_dir / "action_traces"
    video_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)
    action_trace_dir.mkdir(parents=True, exist_ok=True)

    max_rollout_videos = int(cfg.EVALUATION.get("max_rollout_videos", len(cases)))
    max_prediction_videos = int(cfg.EVALUATION.get("max_prediction_videos", len(cases)))
    save_per_replan_prediction = bool(cfg.EVALUATION.get("save_per_replan_prediction_videos", False))
    warmup_passes = int(cfg.EVALUATION.get("warmup_passes", 0))
    warmup_policy = str(cfg.EVALUATION.get("warmup_policy", "scripted"))
    warmup_update_interval = int(
        cfg.EVALUATION.get("warmup_update_interval", cfg.EVALUATION.get("replan_steps", 10))
    )
    if warmup_passes < 0:
        raise ValueError(f"EVALUATION.warmup_passes must be non-negative, got {warmup_passes}.")
    if warmup_passes > 0 and warmup_policy != "scripted":
        raise ValueError(f"Only EVALUATION.warmup_policy=scripted is supported, got {warmup_policy}.")

    results: dict[str, Any] = {
        "checkpoint": str(_resolve_path(cfg.ckpt)),
        "dataset_stats_path": str(stats_path),
        "total_episodes": int(len(cases)),
        "successes": 0,
        "success_rate": 0.0,
        "future_video_psnr_mean": None,
        "episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0.0,
        "gpu_id": int(cfg.get("gpu_id", 0)),
        "visualize_future_video": bool(cfg.EVALUATION.get("visualize_future_video", False)),
        "warmup_passes": int(warmup_passes),
        "warmup_policy": warmup_policy if warmup_passes > 0 else None,
        "warmup_update_interval": int(warmup_update_interval) if warmup_passes > 0 else None,
    }
    psnr_values: list[float] = []

    for trial_idx, (case, seed, dataset_episode_index) in enumerate(cases):
        warmup_summaries: list[dict[str, Any]] = []
        reset_ttt_state = True
        if warmup_passes > 0:
            if hasattr(model, "reset_video_ttt_state"):
                model.reset_video_ttt_state()
            for pass_idx in range(warmup_passes):
                warmup_summaries.append(
                    run_scripted_ttt_warmup_pass(
                        case=case,
                        seed=seed,
                        model=model,
                        processor=processor,
                        cfg=cfg,
                        trial_idx=trial_idx,
                        pass_idx=pass_idx,
                        action_horizon=action_horizon,
                        input_w=input_w,
                        input_h=input_h,
                        model_device=device,
                    )
                )
            reset_ttt_state = False

        episode, replay_images, future_clips, episode_psnr = run_episode(
            case=case,
            seed=seed,
            dataset_episode_index=dataset_episode_index,
            model=model,
            processor=processor,
            cfg=cfg,
            trial_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=device,
            reset_ttt_state=reset_ttt_state,
        )
        if warmup_passes > 0:
            episode["warmup_passes"] = int(warmup_passes)
            episode["warmup_summaries"] = warmup_summaries
        if episode["success"]:
            results["successes"] += 1
        if episode_psnr is not None:
            episode["future_video_psnr"] = float(episode_psnr)
            psnr_values.append(float(episode_psnr))

        episode["action_trace_path"] = _save_action_trace(action_trace_dir, episode)
        if trial_idx < max_rollout_videos:
            episode["rollout_video_path"] = save_rollout_video(
                str(video_dir),
                replay_images,
                f"trial{trial_idx:04d}_{case.case_id}",
                success=bool(episode["success"]),
                task_description=episode["task_description"],
                fps=int(cfg.EVALUATION.get("rollout_fps", 20)),
            )

        if bool(cfg.EVALUATION.get("visualize_future_video", False)) and len(future_clips) > 0:
            all_gt_frames = []
            all_pred_frames = []
            for clip in future_clips:
                all_gt_frames.extend(clip["gt_frames"])
                all_pred_frames.extend(clip["pred_frames"])
                if save_per_replan_prediction and trial_idx < max_prediction_videos:
                    save_prediction_video(
                        str(predicted_video_dir),
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"trial{trial_idx:04d}_{case.case_id}",
                        clip["replan_idx"],
                        success=bool(episode["success"]),
                        task_description=episode["task_description"],
                    )
            if trial_idx < max_prediction_videos and len(all_gt_frames) > 0:
                episode["prediction_video_path"] = save_prediction_video(
                    str(predicted_video_dir),
                    all_gt_frames,
                    all_pred_frames,
                    f"trial{trial_idx:04d}_{case.case_id}",
                    "all",
                    success=bool(episode["success"]),
                    task_description=episode["task_description"],
                )

        slim_episode = dict(episode)
        for heavy_key in ("actions", "eef_xyz", "payload_xyz", "carrier_xyz"):
            slim_episode.pop(heavy_key, None)
        results["episodes"].append(slim_episode)

        results["success_rate"] = results["successes"] / max(1, trial_idx + 1)
        with (run_dir / "results_partial.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

    results["duration"] = time.time() - start_time
    results["success_rate"] = results["successes"] / max(1, len(cases))
    if psnr_values:
        results["future_video_psnr_mean"] = float(np.mean(psnr_values))

    output_file = run_dir / "results.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(
        f"Dynamic carrier eval completed: {results['successes']}/{len(cases)} "
        f"successes ({results['success_rate']:.3f})"
    )
    if results["future_video_psnr_mean"] is not None:
        print(f"Future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Results: {output_file}")
    return results


if __name__ == "__main__":
    main()
