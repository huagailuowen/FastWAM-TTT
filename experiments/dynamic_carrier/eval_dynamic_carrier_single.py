#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from dataclasses import replace
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
from fastwam.datasets.lerobot.robot_video_dataset import (  # noqa: E402
    DEFAULT_PROMPT,
    DEFAULT_TTT_OBSERVATION_INSTRUCTION,
)
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


def _observation_prompt_for_ttt(cfg: DictConfig) -> str:
    instruction = str(
        cfg.EVALUATION.get(
            "ttt_observation_instruction",
            cfg.data.train.get(
                "ttt_observation_instruction",
                DEFAULT_TTT_OBSERVATION_INSTRUCTION,
            ),
        )
    )
    return DEFAULT_PROMPT.format(task=instruction)


def _planner_config_from_eval_cfg(cfg: DictConfig) -> PlannerConfig:
    return PlannerConfig(
        intercept_lead_s=float(cfg.EVALUATION.get("intercept_lead_s", 0.42)),
        position_gain=float(cfg.EVALUATION.get("position_gain", 10.0)),
        max_pos_action=float(cfg.EVALUATION.get("max_pos_action", 1.0)),
        xy_tolerance=float(cfg.EVALUATION.get("xy_tolerance", 0.035)),
        target_xy_tolerance=float(cfg.EVALUATION.get("target_xy_tolerance", 0.055)),
        z_tolerance=float(cfg.EVALUATION.get("z_tolerance", 0.035)),
    )


def _get_observe_then_act_interval(cfg: DictConfig) -> int:
    return int(
        cfg.EVALUATION.get(
            "observe_then_act_interval",
            cfg.data.train.get("chunk_interval", cfg.EVALUATION.get("replan_steps", 10)),
        )
    )


def _get_observe_then_act_update_interval(cfg: DictConfig) -> int:
    return int(
        cfg.EVALUATION.get(
            "observe_then_act_update_interval",
            cfg.data.train.get("chunk_interval", _get_observe_then_act_interval(cfg)),
        )
    )


def _select_observe_then_act_frames(cfg: DictConfig, *, update_interval: int, max_chunks: int) -> int:
    explicit = cfg.EVALUATION.get("observe_then_act_frames", None)
    if explicit is not None:
        observe_frames = int(explicit)
    else:
        default_frames = int(max_chunks * update_interval)
        min_frames = int(
            cfg.EVALUATION.get(
                "observe_then_act_frames_min",
                cfg.data.train.get("observe_frames_min", default_frames),
            )
        )
        max_frames = int(
            cfg.EVALUATION.get(
                "observe_then_act_frames_max",
                cfg.data.train.get("observe_frames_max", default_frames),
            )
        )
        if min_frames > max_frames:
            raise ValueError("observe_then_act_frames_min cannot be greater than observe_then_act_frames_max.")
        random_frames = bool(
            cfg.EVALUATION.get(
                "observe_then_act_random_frames",
                cfg.data.train.get("random_observe_frames", False),
            )
        )
        if random_frames and min_frames < max_frames:
            choices = np.arange(min_frames, max_frames + 1, int(update_interval), dtype=np.int64)
            observe_frames = int(np.random.choice(choices))
        else:
            observe_frames = int(max_frames)
    if observe_frames <= 0:
        raise ValueError(f"observe_then_act_frames must be positive, got {observe_frames}.")
    if observe_frames % int(update_interval) != 0:
        raise ValueError(
            f"observe_then_act_frames={observe_frames} must be divisible by update_interval={update_interval}."
        )
    return int(observe_frames)


def _model_has_video_ttt_adapter(model: torch.nn.Module) -> bool:
    enabled = getattr(model, "video_ttt_enabled", False)
    if callable(enabled):
        enabled = enabled()
    return bool(enabled)


def _phase_shift_case(
    case: DynamicCarrierCase,
    *,
    group_index: int,
    try_index: int,
    tries_per_group: int,
) -> DynamicCarrierCase:
    offset = 2.0 * math.pi * float(try_index) / max(float(tries_per_group), 1.0)
    motion = replace(case.motion, phase=float((case.motion.phase + offset) % (2.0 * math.pi)))
    return replace(
        case,
        case_id=f"{case.case_id}_evalgroup{group_index:04d}_try{try_index:02d}",
        motion=motion,
    )


def _load_eval_cases(cfg: DictConfig) -> list[dict[str, Any]]:
    metadata_path = _resolve_path(
        cfg.EVALUATION.get(
            "dynamic_metadata_path",
            "./data/ttt_dynamic_carrier_cream_200_lerobot/dynamic_carrier_generation_metadata.json",
        )
    )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    start = int(cfg.EVALUATION.get("case_start", 0))
    num_trials = int(cfg.EVALUATION.num_trials)
    default_repeat = str(payload.get("ttt_mode", "")) == "repeat_attempt"
    use_groups = bool(cfg.EVALUATION.get("repeat_eval_from_groups", default_repeat))

    if use_groups and payload.get("groups"):
        groups = list(payload.get("groups") or [])
        selected = groups[start : start + num_trials]
        if len(selected) < num_trials:
            selected.extend(groups[: num_trials - len(selected)])
        records: list[dict[str, Any]] = []
        for fallback_idx, group in enumerate(selected):
            tries = list(group.get("tries") or [])
            try_cases = [
                DynamicCarrierCase.from_dict(item["case"])
                for item in tries
                if isinstance(item, dict) and item.get("case") is not None
            ]
            try_seeds = [
                int(item.get("seed", int(payload.get("seed", 0)) + fallback_idx + try_idx))
                for try_idx, item in enumerate(tries)
                if isinstance(item, dict)
            ]
            records.append(
                {
                    "case": DynamicCarrierCase.from_dict(group["case"]),
                    "seed": int(try_seeds[0] if try_seeds else int(payload.get("seed", 0)) + fallback_idx),
                    "dataset_episode_index": int((group.get("episode_indices") or [start + fallback_idx])[0]),
                    "group_id": int(group.get("group_id", start + fallback_idx)),
                    "try_cases": try_cases,
                    "try_seeds": try_seeds,
                    "episode_indices": [int(x) for x in group.get("episode_indices", [])],
                    "repeat_group": True,
                    "metadata_path": str(metadata_path),
                }
            )
        return records

    successes = payload.get("successes", [])
    if not isinstance(successes, list) or len(successes) == 0:
        raise ValueError(f"No successful cases found in {metadata_path}")
    selected = successes[start : start + num_trials]
    if len(selected) < num_trials:
        selected.extend(successes[: num_trials - len(selected)])

    records = []
    for fallback_idx, item in enumerate(selected):
        records.append(
            {
                "case": DynamicCarrierCase.from_dict(item["case"]),
                "seed": int(item.get("seed", payload.get("seed", 0) + fallback_idx)),
                "dataset_episode_index": int(item.get("episode_index", start + fallback_idx)),
                "group_id": int(start + fallback_idx),
                "try_cases": [],
                "try_seeds": [],
                "episode_indices": [],
                "repeat_group": False,
                "metadata_path": str(metadata_path),
            }
        )
    return records


def _case_for_try(record: dict[str, Any], try_idx: int, tries_per_group: int) -> DynamicCarrierCase:
    try_cases = record.get("try_cases") or []
    if try_idx < len(try_cases):
        return try_cases[try_idx]
    return _phase_shift_case(
        record["case"],
        group_index=int(record.get("group_id", 0)),
        try_index=int(try_idx),
        tries_per_group=int(tries_per_group),
    )


def _seed_for_try(record: dict[str, Any], try_idx: int) -> int:
    try_seeds = record.get("try_seeds") or []
    if try_idx < len(try_seeds):
        return int(try_seeds[try_idx])
    return int(record["seed"]) + int(try_idx)


def _episode_index_for_try(record: dict[str, Any], try_idx: int) -> int:
    episode_indices = record.get("episode_indices") or []
    if try_idx < len(episode_indices):
        return int(episode_indices[try_idx])
    return int(record.get("dataset_episode_index", 0))


@torch.no_grad()
def _apply_restart_ttt_marker(
    *,
    model: torch.nn.Module,
    cfg: DictConfig,
    input_w: int,
    input_h: int,
    model_device: str,
) -> Optional[float]:
    """Apply the repeat-attempt black-frame switch update used during training."""
    if not hasattr(model, "_apply_video_ttt_observation"):
        return None
    restart_instruction = str(
        cfg.data.train.get(
            "restart_instruction",
            "restart the same dynamic carrier task and try again in the same environment",
        )
    )
    prompt = DEFAULT_PROMPT.format(task=restart_instruction)
    context, context_mask = model.encode_prompt(prompt)
    if getattr(model, "proprio_encoder", None) is not None:
        zero_proprio = torch.zeros(
            (1, int(model.proprio_dim)),
            device=model.device,
            dtype=model.torch_dtype,
        )
        context, context_mask = model._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=zero_proprio,
        )
    black_image = torch.full(
        (1, 3, int(input_h), int(input_w)),
        -1.0,
        device=model_device,
        dtype=model.torch_dtype,
    )
    first_frame_latents = model._encode_input_image_latents_tensor(
        input_image=black_image,
        tiled=bool(cfg.EVALUATION.get("tiled", False)),
    )
    tokens = model._build_video_ttt_observation_tokens(
        first_frame_latents=first_frame_latents,
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)),
        global_time=0,
    )
    _, _, ttt_loss = model._apply_video_ttt_observation(
        tokens,
        state=model._video_ttt_inference_state,
        persist_state=True,
        update=True,
        update_tokens=tokens,
    )
    if ttt_loss is None:
        return None
    return float(ttt_loss.detach().float().cpu().item())


@torch.no_grad()
def _apply_observation_ttt_update(
    *,
    obs: dict[str, Any],
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    input_w: int,
    input_h: int,
    model_device: str,
    ttt_global_time: int = 0,
) -> tuple[Optional[float], float, dict[str, np.ndarray]]:
    if not _model_has_video_ttt_adapter(model) or not hasattr(model, "_apply_video_ttt_observation"):
        return None, 0.0, {}
    start_time = time.perf_counter()
    del task_description
    prompt = _observation_prompt_for_ttt(cfg)
    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )
    context, context_mask = model.encode_prompt(prompt)
    if getattr(model, "proprio_encoder", None) is not None:
        context, context_mask = model._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
    first_frame_latents = model._encode_input_image_latents_tensor(
        input_image=image,
        tiled=bool(cfg.EVALUATION.get("tiled", False)),
    )
    tokens = model._build_video_ttt_observation_tokens(
        first_frame_latents=first_frame_latents,
        context=context,
        context_mask=context_mask,
        fuse_vae_embedding_in_latents=bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)),
        global_time=int(ttt_global_time),
    )
    _, _, ttt_loss = model._apply_video_ttt_observation(
        tokens,
        state=model._video_ttt_inference_state,
        persist_state=True,
        update=True,
        update_tokens=tokens,
    )
    elapsed = float(time.perf_counter() - start_time)
    loss = None if ttt_loss is None else float(ttt_loss.detach().float().cpu().item())
    return loss, elapsed, imgs


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
    update_video_ttt: bool = True,
    ttt_global_time: int = 0,
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
        "update_video_ttt": bool(update_video_ttt),
        "ttt_global_time": int(ttt_global_time),
    }

    predicted_future_frames = None
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
        infer_kwargs["test_action_with_infer_action"] = False
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
    pickup_success = False
    first_pickup_step: Optional[int] = None

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
                    ttt_global_time=len(action_trace) // 2,
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
            if env.payload_attached_to_gripper and not pickup_success:
                pickup_success = True
                first_pickup_step = len(action_trace)

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
            "pickup_success": bool(pickup_success),
            "first_pickup_step": int(first_pickup_step) if first_pickup_step is not None else None,
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


def run_observe_then_act_episode(
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
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 0))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])
    observe_chunks = int(cfg.EVALUATION.get("observe_then_act_chunks", 0))
    observe_interval = _get_observe_then_act_interval(cfg)
    observe_update_interval = _get_observe_then_act_update_interval(cfg)
    observe_frames = _select_observe_then_act_frames(
        cfg,
        update_interval=observe_update_interval,
        max_chunks=observe_chunks,
    )
    observe_update_chunks = int(observe_frames // observe_update_interval)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", observe_interval))
    observe_policy = str(cfg.EVALUATION.get("observe_then_act_observe_policy", "dummy"))
    count_observe_steps = bool(cfg.EVALUATION.get("observe_then_act_count_observe_steps", True))
    update_during_observe = bool(
        cfg.EVALUATION.get("observe_then_act_update_during_observe", True)
    ) and _model_has_video_ttt_adapter(model)
    update_during_policy = bool(
        cfg.EVALUATION.get("observe_then_act_update_during_policy", True)
    ) and _model_has_video_ttt_adapter(model)
    action_infer_updates_ttt = bool(
        cfg.EVALUATION.get("observe_then_act_action_infer_updates_ttt", False)
    )

    if observe_chunks <= 0:
        raise ValueError(f"EVALUATION.observe_then_act_chunks must be positive, got {observe_chunks}.")
    if observe_interval <= 0:
        raise ValueError(
            f"EVALUATION.observe_then_act_interval must be positive, got {observe_interval}."
        )
    if observe_update_interval <= 0:
        raise ValueError(
            f"EVALUATION.observe_then_act_update_interval must be positive, got {observe_update_interval}."
        )
    if observe_update_chunks > observe_chunks:
        raise ValueError(
            f"Selected observe warmup needs {observe_update_chunks} updates, "
            f"but observe_then_act_chunks only allows {observe_chunks}."
        )
    if observe_policy not in {"dummy", "scripted"}:
        raise ValueError(
            "EVALUATION.observe_then_act_observe_policy must be 'dummy' or 'scripted', "
            f"got {observe_policy!r}."
        )

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
    pickup_success = False
    first_pickup_step: Optional[int] = None

    total_budget = (
        max_steps
        if count_observe_steps
        else num_steps_wait + observe_frames + max_steps
    )
    pbar = None
    try:
        obs = env.reset(init_state=init_state)
        planner = (
            ScriptedDynamicCarrierPlanner(env, _planner_config_from_eval_cfg(cfg))
            if observe_policy == "scripted"
            else None
        )
        if planner is not None:
            planner.reset()

        done = False
        success = False
        elapsed_steps = 0
        observe_steps_done = 0
        observe_ttt_updates = 0
        policy_ttt_updates = 0
        ttt_update_seconds: list[float] = []
        action_infer_seconds: list[float] = []
        pbar = tqdm(total=total_budget, desc=f"trial {trial_idx + 1} observe_then_act")

        for _ in range(num_steps_wait):
            replay_images.append(get_libero_image(obs).copy())
            action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
            action_trace.append(action.astype(float).tolist())
            obs, _, done, _ = env.step(action)
            if env.payload_attached_to_gripper and not pickup_success:
                pickup_success = True
                first_pickup_step = len(action_trace)
            elapsed_steps += 1
            pbar.update(1)
            if done:
                break

        ttt_time_origin_step = int(elapsed_steps)
        ttt_time_stride = max(int(cfg.data.train.get("ttt_time_stride", 2)), 1)

        def relative_ttt_time() -> int:
            return max(int(elapsed_steps) - ttt_time_origin_step, 0) // ttt_time_stride

        if not done:
            for chunk_idx in range(observe_update_chunks):
                if update_during_observe:
                    _, update_elapsed, _ = _apply_observation_ttt_update(
                        obs=obs,
                        task_description=task_description,
                        model=model,
                        processor=processor,
                        cfg=cfg,
                        input_w=input_w,
                        input_h=input_h,
                        model_device=model_device,
                        ttt_global_time=relative_ttt_time(),
                    )
                    ttt_update_seconds.append(update_elapsed)
                    observe_ttt_updates += 1

                for _ in range(observe_update_interval):
                    replay_images.append(get_libero_image(obs).copy())
                    if planner is not None:
                        action = planner.act(obs).astype(np.float32)
                    else:
                        action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
                    eef_trace.append(env.eef_position(obs).astype(float).tolist())
                    payload_trace.append(env.payload_position().astype(float).tolist())
                    carrier_trace.append(env.carrier_position().astype(float).tolist())
                    action_trace.append(action.astype(float).tolist())
                    obs, _, done, _ = env.step(action)
                    if env.payload_attached_to_gripper and not pickup_success:
                        pickup_success = True
                        first_pickup_step = len(action_trace)
                    elapsed_steps += 1
                    observe_steps_done += 1
                    pbar.update(1)
                    if done:
                        break
                if done:
                    break

        policy_budget = max_steps - elapsed_steps if count_observe_steps else max_steps
        policy_budget = max(int(policy_budget), 0)
        policy_steps = 0

        for _ in range(policy_budget):
            if done:
                break
            if (
                update_during_policy
                and policy_steps % observe_update_interval == 0
            ):
                _, update_elapsed, _ = _apply_observation_ttt_update(
                    obs=obs,
                    task_description=task_description,
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    input_w=input_w,
                    input_h=input_h,
                    model_device=model_device,
                    ttt_global_time=relative_ttt_time(),
                )
                ttt_update_seconds.append(update_elapsed)
                policy_ttt_updates += 1
            if len(pending_actions) == 0:
                infer_start = time.perf_counter()
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
                    update_video_ttt=action_infer_updates_ttt,
                    ttt_global_time=relative_ttt_time(),
                )
                action_infer_seconds.append(float(time.perf_counter() - infer_start))
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
            if env.payload_attached_to_gripper and not pickup_success:
                pickup_success = True
                first_pickup_step = len(action_trace)
            elapsed_steps += 1
            policy_steps += 1
            pbar.update(1)

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
        if pbar is not None:
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
            "pickup_success": bool(pickup_success),
            "first_pickup_step": int(first_pickup_step) if first_pickup_step is not None else None,
            "steps": int(len(action_trace)),
            "observe_then_act_chunks": int(observe_update_chunks),
            "observe_then_act_max_chunks": int(observe_chunks),
            "observe_then_act_interval": int(observe_interval),
            "observe_then_act_update_interval": int(observe_update_interval),
            "observe_then_act_observe_frames": int(observe_frames),
            "observe_then_act_policy": observe_policy,
            "observe_then_act_ttt_updates": int(observe_ttt_updates),
            "observe_then_act_policy_ttt_updates": int(policy_ttt_updates),
            "observe_then_act_update_during_policy": bool(update_during_policy),
            "observe_then_act_action_infer_updates_ttt": bool(action_infer_updates_ttt),
            "observe_then_act_mean_ttt_update_s": (
                float(np.mean(ttt_update_seconds)) if len(ttt_update_seconds) > 0 else None
            ),
            "observe_then_act_mean_action_infer_s": (
                float(np.mean(action_infer_seconds)) if len(action_infer_seconds) > 0 else None
            ),
            "observe_steps": int(observe_steps_done),
            "policy_steps": int(policy_steps),
            "actions": action_trace,
            "eef_xyz": eef_trace,
            "payload_xyz": payload_trace,
            "carrier_xyz": carrier_trace,
        }
        mean_psnr = float(np.mean(future_clip_psnr)) if len(future_clip_psnr) > 0 else None
        return result, replay_images, predicted_future_video_clips, mean_psnr
    finally:
        if pbar is not None:
            pbar.close()
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
    planner = ScriptedDynamicCarrierPlanner(env, _planner_config_from_eval_cfg(cfg))
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
                        ttt_global_time=t // 2,
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
    observe_then_act_chunks = int(cfg.EVALUATION.get("observe_then_act_chunks", 0))
    if warmup_passes < 0:
        raise ValueError(f"EVALUATION.warmup_passes must be non-negative, got {warmup_passes}.")
    if warmup_passes > 0 and warmup_policy != "scripted":
        raise ValueError(f"Only EVALUATION.warmup_policy=scripted is supported, got {warmup_policy}.")
    if warmup_passes > 0 and observe_then_act_chunks > 0:
        raise ValueError(
            "Use only one adaptation protocol: set either EVALUATION.warmup_passes "
            "or EVALUATION.observe_then_act_chunks, not both."
        )

    results: dict[str, Any] = {
        "checkpoint": str(_resolve_path(cfg.ckpt)),
        "dataset_stats_path": str(stats_path),
        "total_episodes": int(len(cases)),
        "successes": 0,
        "success_rate": 0.0,
        "pickup_successes": 0,
        "pickup_success_rate": 0.0,
        "future_video_psnr_mean": None,
        "episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0.0,
        "gpu_id": int(cfg.get("gpu_id", 0)),
        "visualize_future_video": bool(cfg.EVALUATION.get("visualize_future_video", False)),
        "warmup_passes": int(warmup_passes),
        "warmup_policy": warmup_policy if warmup_passes > 0 else None,
        "warmup_update_interval": int(warmup_update_interval) if warmup_passes > 0 else None,
        "repeat_eval_from_groups": bool(cfg.EVALUATION.get("repeat_eval_from_groups", False)),
        "restart_markers": bool(warmup_passes > 0),
        "observe_then_act_chunks": int(observe_then_act_chunks),
        "observe_then_act_interval": (
            int(_get_observe_then_act_interval(cfg)) if observe_then_act_chunks > 0 else None
        ),
        "observe_then_act_update_interval": (
            int(_get_observe_then_act_update_interval(cfg)) if observe_then_act_chunks > 0 else None
        ),
    }
    psnr_values: list[float] = []
    tries_per_group = int(
        cfg.EVALUATION.get(
            "tries_per_group",
            cfg.data.train.get("tries_per_group", max(warmup_passes + 1, 1)),
        )
    )

    for trial_idx, record in enumerate(cases):
        base_case: DynamicCarrierCase = record["case"]
        case = base_case
        seed = int(record["seed"])
        dataset_episode_index = int(record["dataset_episode_index"])
        warmup_summaries: list[dict[str, Any]] = []
        restart_marker_losses: list[Optional[float]] = []
        reset_ttt_state = True
        if observe_then_act_chunks > 0:
            episode, replay_images, future_clips, episode_psnr = run_observe_then_act_episode(
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
                reset_ttt_state=True,
            )
        elif warmup_passes > 0:
            if hasattr(model, "reset_video_ttt_state"):
                model.reset_video_ttt_state()
            for pass_idx in range(warmup_passes):
                if pass_idx > 0:
                    restart_marker_losses.append(
                        _apply_restart_ttt_marker(
                            model=model,
                            cfg=cfg,
                            input_w=input_w,
                            input_h=input_h,
                            model_device=device,
                        )
                    )
                warmup_case = _case_for_try(record, pass_idx, tries_per_group=tries_per_group)
                warmup_summaries.append(
                    run_scripted_ttt_warmup_pass(
                        case=warmup_case,
                        seed=_seed_for_try(record, pass_idx),
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
            restart_marker_losses.append(
                _apply_restart_ttt_marker(
                    model=model,
                    cfg=cfg,
                    input_w=input_w,
                    input_h=input_h,
                    model_device=device,
                )
            )
            case = _case_for_try(record, warmup_passes, tries_per_group=tries_per_group)
            seed = _seed_for_try(record, warmup_passes)
            dataset_episode_index = _episode_index_for_try(record, warmup_passes)
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
        else:
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
            episode["restart_marker_losses"] = restart_marker_losses
            episode["repeat_group_id"] = int(record.get("group_id", trial_idx))
            episode["base_case_id"] = base_case.case_id
            episode["execution_try_index"] = int(warmup_passes)
            episode["execution_case_id"] = case.case_id
        if episode["success"]:
            results["successes"] += 1
        if episode.get("pickup_success", False):
            results["pickup_successes"] += 1
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
        results["pickup_success_rate"] = results["pickup_successes"] / max(1, trial_idx + 1)
        with (run_dir / "results_partial.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

    results["duration"] = time.time() - start_time
    results["success_rate"] = results["successes"] / max(1, len(cases))
    results["pickup_success_rate"] = results["pickup_successes"] / max(1, len(cases))
    if psnr_values:
        results["future_video_psnr_mean"] = float(np.mean(psnr_values))

    output_file = run_dir / "results.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(
        f"Dynamic carrier eval completed: {results['successes']}/{len(cases)} "
        f"successes ({results['success_rate']:.3f})"
    )
    print(
        f"Dynamic carrier pickup completed: {results['pickup_successes']}/{len(cases)} "
        f"pickups ({results['pickup_success_rate']:.3f})"
    )
    if results["future_video_psnr_mean"] is not None:
        print(f"Future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Results: {output_file}")
    return results


if __name__ == "__main__":
    main()
