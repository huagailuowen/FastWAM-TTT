import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
DEFAULT_TTT_OBSERVATION_INSTRUCTION = (
    "observe the motion trajectory of the object that needs to be grasped; do not take any action"
)

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        restart_instruction: Optional[str] = None,
        observation_instruction: Optional[str] = None,
    ):
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )
    
        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        
        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction
        self.restart_instruction = restart_instruction
        self.observation_instruction = observation_instruction

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad.any().item()):
                has_pad = True
            if bool(proprio_is_pad.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))
        
        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot): 
        #   action: [num_frames-1, action_dim] # start from t0, except the last frame
        #   proprio: [num_frames, proprio_dim] # start from t0 to the last frame, aligned with video frames
        action = sample["action"] # [T-1, action_dim]
        proprio = sample["proprio"][:-1, :] # [T-1, state_dim]， to align with action
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        if action.shape[0] % (video.shape[1] - 1) != 0:
            raise ValueError(
                f"`action` horizon must be divisible by `video` transitions, got {action.shape[0]} and {video.shape[1] - 1}"
            )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if self.observation_instruction is not None:
            observation_prompt = DEFAULT_PROMPT.format(task=str(self.observation_instruction))
            observation_context, observation_context_mask = self._get_cached_text_context(observation_prompt)
            observation_context[~observation_context_mask] = 0.0
            observation_context_mask = torch.ones_like(observation_context_mask)
            data.update(
                {
                    "observation_prompt": observation_prompt,
                    "observation_context": observation_context,
                    "observation_context_mask": observation_context_mask,
                }
            )
        if self.restart_instruction is not None:
            restart_instruction = DEFAULT_PROMPT.format(task=str(self.restart_instruction))
            restart_context, restart_context_mask = self._get_cached_text_context(restart_instruction)
            restart_context[~restart_context_mask] = 0.0
            restart_context_mask = torch.ones_like(restart_context_mask)
            data.update(
                {
                    "restart_prompt": restart_instruction,
                    "restart_context": restart_context,
                    "restart_context_mask": restart_context_mask,
                }
            )
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data


class RobotVideoTTTGroupedDataset(RobotVideoDataset):
    """LeRobot dynamic-carrier dataset with explicit TTT episode structure.

    repeat_attempt mode returns a same-environment group as tensors with try
    and chunk dimensions: video=[tries, chunks, 3, T, H, W].
    observe_then_act mode returns a continuous execution sequence plus warmup
    observation frames sampled before execution:
    ttt_warmup_video=[chunks, 3, H, W],
    ttt_execution_video=[exec_chunks, 3, T, H, W].
    When enabled, policy update frames are sampled separately so TTT can update
    more often than action/video chunks are trained.
    """

    def __init__(
        self,
        *args,
        ttt_metadata_path: Optional[str] = None,
        ttt_mode: Optional[str] = None,
        tries_per_group: int = 6,
        observe_chunks: int = 20,
        chunk_interval: int = 10,
        execution_chunk_interval: Optional[int] = None,
        observe_frames: Optional[int] = None,
        observe_frames_min: Optional[int] = None,
        observe_frames_max: Optional[int] = None,
        random_observe_frames: bool = False,
        policy_update_during_execution: bool = False,
        stage1_execution_only: bool = False,
        ttt_time_stride: int = 2,
        ttt_observation_instruction: Optional[str] = None,
        entry_repeat: int = 1,
        **kwargs,
    ):
        if ttt_observation_instruction is None and not bool(stage1_execution_only):
            ttt_observation_instruction = DEFAULT_TTT_OBSERVATION_INSTRUCTION
        if ttt_observation_instruction is not None:
            kwargs.setdefault("observation_instruction", str(ttt_observation_instruction))
        super().__init__(*args, **kwargs)
        if ttt_metadata_path is None:
            dataset_dirs = kwargs.get("dataset_dirs")
            if dataset_dirs is None and len(args) >= 1:
                dataset_dirs = args[0]
            if not dataset_dirs:
                raise ValueError("`ttt_metadata_path` is required when dataset_dirs is empty.")
            ttt_metadata_path = str(Path(str(dataset_dirs[0])) / "dynamic_carrier_generation_metadata.json")
        self.ttt_metadata_path = Path(ttt_metadata_path)
        self.ttt_metadata = json.loads(self.ttt_metadata_path.read_text(encoding="utf-8"))
        self.ttt_mode = str(ttt_mode or self.ttt_metadata.get("ttt_mode", "")).strip()
        if self.ttt_mode not in {"repeat_attempt", "observe_then_act"}:
            raise ValueError(
                f"`ttt_mode` must be repeat_attempt or observe_then_act, got {self.ttt_mode!r}."
            )
        self.tries_per_group = int(tries_per_group)
        self.observe_chunks = int(observe_chunks)
        self.chunk_interval = int(chunk_interval)
        self.execution_chunk_interval = int(
            self.chunk_interval if execution_chunk_interval is None else execution_chunk_interval
        )
        self.observe_frames_override = None if observe_frames is None else int(observe_frames)
        self.observe_frames_min = None if observe_frames_min is None else int(observe_frames_min)
        self.observe_frames_max = None if observe_frames_max is None else int(observe_frames_max)
        self.random_observe_frames = bool(random_observe_frames)
        self.policy_update_during_execution = bool(policy_update_during_execution)
        self.stage1_execution_only = bool(stage1_execution_only)
        self.ttt_time_stride = int(ttt_time_stride)
        self.entry_repeat = int(entry_repeat)
        if self.tries_per_group <= 0:
            raise ValueError("`tries_per_group` must be positive.")
        if self.observe_chunks <= 0:
            raise ValueError("`observe_chunks` must be positive.")
        if self.chunk_interval <= 0:
            raise ValueError("`chunk_interval` must be positive.")
        if self.execution_chunk_interval <= 0:
            raise ValueError("`execution_chunk_interval` must be positive.")
        if self.observe_frames_override is not None and self.observe_frames_override <= 0:
            raise ValueError("`observe_frames` must be positive when provided.")
        if self.observe_frames_min is not None and self.observe_frames_min <= 0:
            raise ValueError("`observe_frames_min` must be positive when provided.")
        if self.observe_frames_max is not None and self.observe_frames_max <= 0:
            raise ValueError("`observe_frames_max` must be positive when provided.")
        if (self.observe_frames_min is None) != (self.observe_frames_max is None):
            raise ValueError("`observe_frames_min` and `observe_frames_max` must be provided together.")
        if self.observe_frames_min is not None:
            if self.observe_frames_min > self.observe_frames_max:
                raise ValueError("`observe_frames_min` cannot be greater than `observe_frames_max`.")
            if self.observe_frames_min % self.chunk_interval != 0 or self.observe_frames_max % self.chunk_interval != 0:
                raise ValueError("observe frame range must align with `chunk_interval`.")
        if self.policy_update_during_execution and self.execution_chunk_interval % self.chunk_interval != 0:
            raise ValueError(
                "`execution_chunk_interval` must be divisible by `chunk_interval` "
                "when `policy_update_during_execution=true`."
            )
        if self.ttt_time_stride <= 0:
            raise ValueError("`ttt_time_stride` must be positive.")
        if self.entry_repeat <= 0:
            raise ValueError("`entry_repeat` must be positive.")
        self.ttt_entries = self._build_ttt_entries()
        if self.entry_repeat > 1:
            self.ttt_entries = self.ttt_entries * self.entry_repeat
        self.max_execution_chunks = max(
            len(entry.get("execution_offsets", [])) for entry in self.ttt_entries
        )
        self.max_policy_update_chunks = max(
            len(entry.get("policy_update_offsets", [])) for entry in self.ttt_entries
        )

    def _episode_bounds(self, episode_index: int) -> tuple[int, int]:
        starts = self.lerobot_dataset.episode_data_index["from"]
        ends = self.lerobot_dataset.episode_data_index["to"]
        if episode_index < 0 or episode_index >= len(starts):
            raise IndexError(f"Episode index {episode_index} out of bounds for {len(starts)} episodes.")
        return int(starts[episode_index].item()), int(ends[episode_index].item())

    def _max_valid_start(self, episode_indices: list[int]) -> int:
        stride = int(self.lerobot_dataset.global_sample_stride)
        required_tail = (int(self.num_frames) - 1) * stride
        lengths = []
        for episode_index in episode_indices:
            start, end = self._episode_bounds(int(episode_index))
            lengths.append(max(int(end - start), 1))
        return max(min(lengths) - 1 - required_tail, 0)

    def _global_index(self, episode_index: int, offset: int) -> int:
        start, _ = self._episode_bounds(int(episode_index))
        return int(start + max(int(offset), 0))

    def _num_warmup_chunks_for_frames(self, observe_frames: int) -> int:
        return min(
            int(math.ceil(float(observe_frames) / float(self.chunk_interval))),
            int(self.observe_chunks),
        )

    def _ttt_time_origin_frame(self, action_start_frame: int, observe_frames: int) -> int:
        valid_chunks = self._num_warmup_chunks_for_frames(int(observe_frames))
        return max(int(action_start_frame) - valid_chunks * int(self.chunk_interval), 0)

    def _offsets_to_ttt_times(
        self,
        offsets: list[int] | torch.Tensor,
        origin_frame: int = 0,
    ) -> torch.Tensor:
        if isinstance(offsets, torch.Tensor):
            offsets_tensor = offsets.to(dtype=torch.long)
        else:
            offsets_tensor = torch.tensor([int(offset) for offset in offsets], dtype=torch.long)
        relative_offsets = (offsets_tensor - int(origin_frame)).clamp(min=0)
        return torch.div(relative_offsets, int(self.ttt_time_stride), rounding_mode="floor")

    def _execution_sample_offsets(self, start: int, max_start: int) -> list[int]:
        offsets = list(range(int(start), int(max_start) + 1, int(self.chunk_interval)))
        return [int(offset) for offset in offsets] or [int(start)]

    def _prediction_phase_offsets(self, execution_start: int, max_start: int) -> list[int]:
        phase_limit = min(
            int(max_start),
            int(execution_start) + int(self.execution_chunk_interval) - int(self.chunk_interval),
        )
        offsets = list(range(int(execution_start), int(phase_limit) + 1, int(self.chunk_interval)))
        return [int(offset) for offset in offsets] or [int(execution_start)]

    def _execution_prediction_offsets(self, first_prediction_offset: int, max_start: int) -> list[int]:
        offsets = list(
            range(int(first_prediction_offset), int(max_start) + 1, int(self.execution_chunk_interval))
        )
        return [int(offset) for offset in offsets] or [int(first_prediction_offset)]

    def _policy_update_schedule(
        self,
        execution_start: int,
        execution_offsets: list[int],
    ) -> tuple[list[int], list[tuple[int, int]]]:
        update_offsets: list[int] = []
        update_ranges: list[tuple[int, int]] = []
        next_update_offset = int(execution_start)
        for prediction_offset in execution_offsets:
            start_idx = len(update_offsets)
            current = max(int(next_update_offset), int(execution_start))
            while current <= int(prediction_offset):
                update_offsets.append(int(current))
                current += int(self.chunk_interval)
            update_ranges.append((int(start_idx), int(len(update_offsets))))
            next_update_offset = int(prediction_offset) + int(self.chunk_interval)
        return update_offsets, update_ranges

    def _build_ttt_entries(self) -> list[dict]:
        groups = list(self.ttt_metadata.get("groups") or [])
        entries: list[dict] = []
        if self.ttt_mode == "repeat_attempt":
            for group in groups:
                episode_indices = [int(x) for x in group.get("episode_indices", [])[: self.tries_per_group]]
                if len(episode_indices) < self.tries_per_group:
                    continue
                max_start = self._max_valid_start(episode_indices)
                offsets = list(range(0, max_start + 1, self.execution_chunk_interval)) or [0]
                entries.append(
                    {
                        "mode": "repeat_attempt",
                        "group_id": int(group.get("group_id", len(entries))),
                        "episode_indices": episode_indices,
                        "offset": int(offsets[0]),
                        "execution_offsets": [int(offset) for offset in offsets],
                    }
                )
        else:
            if not groups:
                groups = [
                    {
                        "group_id": idx,
                        "episode_indices": [int(item["episode_index"])],
                        "observe_frames": int(item.get("observe_frames", 0)),
                        "action_start_frame": int(
                            item.get(
                                "action_start_frame",
                                item.get("execution_start_frame", item.get("observe_frames", 0)),
                            )
                        ),
                    }
                    for idx, item in enumerate(self.ttt_metadata.get("successes", []))
                    if item.get("episode_index") is not None
                ]
            default_observe_frames = int(self.observe_chunks * self.chunk_interval)
            for group in groups:
                episode_indices = [int(x) for x in group.get("episode_indices", [])]
                if not episode_indices:
                    continue
                episode_index = episode_indices[0]
                metadata_observe_frames = int(group.get("observe_frames", default_observe_frames))
                if self.observe_frames_override is not None:
                    observe_frames_min = int(self.observe_frames_override)
                    observe_frames_max = int(self.observe_frames_override)
                elif self.observe_frames_min is not None and self.observe_frames_max is not None:
                    observe_frames_min = int(self.observe_frames_min)
                    observe_frames_max = int(self.observe_frames_max)
                else:
                    observe_frames_min = int(metadata_observe_frames)
                    observe_frames_max = int(metadata_observe_frames)
                max_start = self._max_valid_start([episode_index])
                action_start_frame = int(
                    group.get(
                        "action_start_frame",
                        group.get("execution_start_frame", metadata_observe_frames),
                    )
                )
                start = min(max(action_start_frame, 0), max_start)
                offsets = list(range(start, max_start + 1, self.execution_chunk_interval)) or [start]
                execution_sample_offsets = self._execution_sample_offsets(start, max_start)
                policy_update_offsets = []
                if self.policy_update_during_execution:
                    policy_update_offsets = list(range(start, max_start + 1, self.chunk_interval)) or [start]
                entries.append(
                    {
                        "mode": "observe_then_act",
                        "group_id": int(group.get("group_id", len(entries))),
                        "episode_index": int(episode_index),
                        "offset": int(offsets[0]),
                        "execution_offsets": [int(offset) for offset in offsets],
                        "execution_sample_offsets": [int(offset) for offset in execution_sample_offsets],
                        "policy_update_offsets": [int(offset) for offset in policy_update_offsets],
                        "action_start_frame": int(start),
                        "execution_start_frame": int(start),
                        "observe_frames": int(observe_frames_max),
                        "observe_frames_min": int(observe_frames_min),
                        "observe_frames_max": int(observe_frames_max),
                    }
                )
        if not entries:
            raise ValueError(f"No TTT entries could be built from {self.ttt_metadata_path}.")
        return entries

    def __len__(self):
        return len(self.ttt_entries)

    @staticmethod
    def _stack_try_samples(samples: list[dict]) -> dict:
        tensor_keys = [
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
        ]
        data = {
            key: torch.stack([sample[key] for sample in samples], dim=0)
            for key in tensor_keys
            if key in samples[0]
        }
        data["prompt"] = [sample.get("prompt", "") for sample in samples]
        if "observation_prompt" in samples[0]:
            data["observation_prompt"] = samples[0].get("observation_prompt", "")
        if "restart_context" in samples[0]:
            data["restart_context"] = samples[0]["restart_context"]
            data["restart_context_mask"] = samples[0]["restart_context_mask"]
            data["restart_prompt"] = samples[0].get("restart_prompt", "")
        return data

    def _stack_try_sequence_samples(
        self,
        try_sequences: list[list[dict]],
        start_offsets: list[int],
    ) -> dict:
        if len(try_sequences) <= 0:
            raise ValueError("repeat-attempt sequence must contain at least one try.")
        tensor_keys = [
            "video",
            "action",
            "proprio",
            "image_is_pad",
            "action_is_pad",
            "proprio_is_pad",
        ]
        sequence_data = {}
        masks = []
        for sequence in try_sequences:
            if len(sequence) <= 0:
                raise ValueError("repeat-attempt try sequence must contain at least one chunk.")
            padded = list(sequence)
            while len(padded) < self.max_execution_chunks:
                padded.append(sequence[-1])
            masks.append([idx < len(sequence) for idx in range(self.max_execution_chunks)])
            for key in tensor_keys:
                if key not in sequence[0]:
                    continue
                sequence_data.setdefault(key, []).append(
                    torch.stack([sample[key] for sample in padded], dim=0)
                )

        data = {
            key: torch.stack(values, dim=0)
            for key, values in sequence_data.items()
        }
        data["context"] = torch.stack([sequence[0]["context"] for sequence in try_sequences], dim=0)
        data["context_mask"] = torch.stack([sequence[0]["context_mask"] for sequence in try_sequences], dim=0)
        if "observation_context" in try_sequences[0][0]:
            data["observation_context"] = torch.stack(
                [sequence[0]["observation_context"] for sequence in try_sequences],
                dim=0,
            )
            data["observation_context_mask"] = torch.stack(
                [sequence[0]["observation_context_mask"] for sequence in try_sequences],
                dim=0,
            )
            data["observation_prompt"] = try_sequences[0][0].get("observation_prompt", "")
        data["prompt"] = [sequence[0].get("prompt", "") for sequence in try_sequences]
        data["ttt_try_chunk_mask"] = torch.tensor(masks, dtype=torch.bool)
        data["ttt_try_start_offsets"] = torch.tensor(start_offsets, dtype=torch.long)
        if "restart_context" in try_sequences[0][0]:
            data["restart_context"] = try_sequences[0][0]["restart_context"]
            data["restart_context_mask"] = try_sequences[0][0]["restart_context_mask"]
            data["restart_prompt"] = try_sequences[0][0].get("restart_prompt", "")
        return data

    def _select_observe_frames(self, entry: dict) -> int:
        min_frames = int(entry.get("observe_frames_min", entry.get("observe_frames", self.observe_chunks * self.chunk_interval)))
        max_frames = int(entry.get("observe_frames_max", entry.get("observe_frames", min_frames)))
        if not self.random_observe_frames or min_frames == max_frames:
            return max_frames
        num_choices = int((max_frames - min_frames) // self.chunk_interval) + 1
        return int(min_frames + self.chunk_interval * random.randrange(num_choices))

    def _get_observe_warmup(
        self,
        episode_index: int,
        observe_frames: int,
        action_start_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        frames = []
        proprios = []
        valid_chunks = self._num_warmup_chunks_for_frames(int(observe_frames))
        action_start_frame = max(int(action_start_frame), 0)
        offsets = [
            max(action_start_frame - (valid_chunks - idx) * self.chunk_interval, 0)
            for idx in range(valid_chunks)
        ]
        for offset in offsets:
            sample = self._get(self._global_index(episode_index, offset))
            frames.append(sample["video"][:, 0])
            proprios.append(sample["proprio"][0])
        padded_frames = list(frames)
        padded_proprios = list(proprios)
        padded_offsets = [int(offset) for offset in offsets]
        while len(padded_frames) < self.observe_chunks:
            padded_frames.append(frames[-1])
            padded_proprios.append(proprios[-1])
            padded_offsets.append(int(offsets[-1]))
        mask = torch.tensor([idx < valid_chunks for idx in range(self.observe_chunks)], dtype=torch.bool)
        return (
            torch.stack(padded_frames, dim=0),
            torch.stack(padded_proprios, dim=0),
            mask,
            torch.tensor(padded_offsets, dtype=torch.long),
        )

    def _get_policy_updates(
        self,
        episode_index: int,
        offsets: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(offsets) <= 0:
            raise ValueError("policy update offsets must contain at least one entry.")
        frames = []
        proprios = []
        for offset in offsets:
            sample = self._get(self._global_index(episode_index, offset))
            frames.append(sample["video"][:, 0])
            proprios.append(sample["proprio"][0])

        padded_frames = list(frames)
        padded_proprios = list(proprios)
        padded_offsets = [int(offset) for offset in offsets]
        while len(padded_frames) < self.max_policy_update_chunks:
            padded_frames.append(frames[-1])
            padded_proprios.append(proprios[-1])
            padded_offsets.append(int(offsets[-1]))

        mask = torch.tensor(
            [idx < len(offsets) for idx in range(self.max_policy_update_chunks)],
            dtype=torch.bool,
        )
        return (
            torch.stack(padded_frames, dim=0),
            torch.stack(padded_proprios, dim=0),
            mask,
            torch.tensor(padded_offsets, dtype=torch.long),
        )

    def _stack_execution_samples(self, samples: list[dict]) -> dict:
        if len(samples) <= 0:
            raise ValueError("observe-then-act execution sequence must contain at least one chunk.")
        tensor_keys = [
            "video",
            "action",
            "proprio",
            "image_is_pad",
            "action_is_pad",
            "proprio_is_pad",
        ]
        padded_samples = list(samples)
        while len(padded_samples) < self.max_execution_chunks:
            padded_samples.append(samples[-1])

        data = {
            f"ttt_execution_{key}": torch.stack([sample[key] for sample in padded_samples], dim=0)
            for key in tensor_keys
            if key in samples[0]
        }
        data["ttt_execution_mask"] = torch.tensor(
            [idx < len(samples) for idx in range(self.max_execution_chunks)],
            dtype=torch.bool,
        )
        return data

    def __getitem__(self, idx):
        try:
            entry = self.ttt_entries[int(idx)]
            if self.stage1_execution_only:
                offsets = [
                    int(offset)
                    for offset in entry.get(
                        "execution_sample_offsets",
                        entry.get("execution_offsets", [entry["offset"]]),
                    )
                ]
                if len(offsets) <= 0:
                    raise ValueError("stage1 execution-only sample requires at least one execution offset.")
                offset = int(random.choice(offsets))
                if entry["mode"] == "repeat_attempt":
                    episode_index = int(random.choice(entry["episode_indices"]))
                else:
                    episode_index = int(entry["episode_index"])
                data = self._get(self._global_index(episode_index, offset))
                data["ttt_sequence_mode_id"] = torch.tensor(3, dtype=torch.long)
                data["ttt_group_id"] = torch.tensor(int(entry["group_id"]), dtype=torch.long)
                data["ttt_chunk_offset"] = torch.tensor(offset, dtype=torch.long)
                observe_frames = self._select_observe_frames(entry)
                action_start_frame = int(
                    entry.get(
                        "action_start_frame",
                        entry.get("execution_start_frame", min(offsets)),
                    )
                )
                time_origin_frame = self._ttt_time_origin_frame(action_start_frame, observe_frames)
                data["ttt_time_origin_frame"] = torch.tensor(int(time_origin_frame), dtype=torch.long)
                data["ttt_observe_frames"] = torch.tensor(int(observe_frames), dtype=torch.long)
                data["ttt_global_time"] = self._offsets_to_ttt_times([offset], time_origin_frame)[0]
                if "action_start_frame" in entry:
                    data["ttt_action_start_frame"] = torch.tensor(int(entry["action_start_frame"]), dtype=torch.long)
                if "execution_start_frame" in entry:
                    data["ttt_execution_start_frame"] = torch.tensor(
                        int(entry["execution_start_frame"]),
                        dtype=torch.long,
                    )
                return data

            if entry["mode"] == "repeat_attempt":
                all_offsets = [int(offset) for offset in entry.get("execution_offsets", [entry["offset"]])]
                try_sequences = []
                start_offsets = []
                for episode_index in entry["episode_indices"]:
                    offsets = all_offsets
                    start_offsets.append(int(offsets[0]))
                    try_sequences.append(
                        [
                            self._get(self._global_index(episode_index, offset))
                            for offset in offsets
                        ]
                    )
                data = self._stack_try_sequence_samples(try_sequences, start_offsets)
                data["ttt_sequence_mode_id"] = torch.tensor(1, dtype=torch.long)
                data["ttt_group_id"] = torch.tensor(entry["group_id"], dtype=torch.long)
                data["ttt_chunk_offset"] = torch.tensor(min(start_offsets), dtype=torch.long)
                return data

            observe_frames = self._select_observe_frames(entry)
            max_start = self._max_valid_start([int(entry["episode_index"])])
            execution_start = min(
                max(
                    int(
                        entry.get(
                            "action_start_frame",
                            entry.get("execution_start_frame", observe_frames),
                        )
                    ),
                    0,
                ),
                max_start,
            )
            first_prediction_offset = int(random.choice(self._prediction_phase_offsets(execution_start, max_start)))
            execution_offsets = self._execution_prediction_offsets(first_prediction_offset, max_start)
            execution_samples = [
                self._get(self._global_index(entry["episode_index"], offset))
                for offset in execution_offsets
            ]
            data = execution_samples[0]
            warmup_video, warmup_proprio, warmup_mask, warmup_offsets = self._get_observe_warmup(
                entry["episode_index"],
                observe_frames,
                execution_start,
            )
            time_origin_frame = int(warmup_offsets[0].item())
            data.update(self._stack_execution_samples(execution_samples))
            data["ttt_warmup_video"] = warmup_video
            data["ttt_warmup_proprio"] = warmup_proprio
            data["ttt_warmup_mask"] = warmup_mask
            data["ttt_warmup_offsets"] = warmup_offsets
            data["ttt_time_origin_frame"] = torch.tensor(int(time_origin_frame), dtype=torch.long)
            data["ttt_warmup_times"] = self._offsets_to_ttt_times(warmup_offsets, time_origin_frame)
            data["ttt_observe_frames"] = torch.tensor(int(observe_frames), dtype=torch.long)
            data["ttt_action_start_frame"] = torch.tensor(int(execution_start), dtype=torch.long)
            data["ttt_execution_start_frame"] = torch.tensor(int(execution_start), dtype=torch.long)
            data["ttt_sequence_mode_id"] = torch.tensor(2, dtype=torch.long)
            data["ttt_group_id"] = torch.tensor(entry["group_id"], dtype=torch.long)
            data["ttt_chunk_offset"] = torch.tensor(execution_offsets[0], dtype=torch.long)
            data["ttt_first_prediction_offset"] = torch.tensor(first_prediction_offset, dtype=torch.long)
            padded_offsets = list(execution_offsets)
            while len(padded_offsets) < self.max_execution_chunks:
                padded_offsets.append(execution_offsets[-1])
            data["ttt_execution_offsets"] = torch.tensor(padded_offsets, dtype=torch.long)
            data["ttt_execution_times"] = self._offsets_to_ttt_times(padded_offsets, time_origin_frame)
            policy_update_offsets = []
            policy_update_ranges: list[tuple[int, int]] = []
            if self.policy_update_during_execution:
                policy_update_offsets, policy_update_ranges = self._policy_update_schedule(
                    execution_start,
                    execution_offsets,
                )
            if self.policy_update_during_execution:
                (
                    policy_update_video,
                    policy_update_proprio,
                    policy_update_mask,
                    policy_update_offsets_tensor,
                ) = self._get_policy_updates(entry["episode_index"], policy_update_offsets)
                data["ttt_policy_update_video"] = policy_update_video
                data["ttt_policy_update_proprio"] = policy_update_proprio
                data["ttt_policy_update_mask"] = policy_update_mask
                data["ttt_policy_update_offsets"] = policy_update_offsets_tensor
                data["ttt_policy_update_times"] = self._offsets_to_ttt_times(
                    policy_update_offsets_tensor,
                    time_origin_frame,
                )
                padded_ranges = list(policy_update_ranges)
                while len(padded_ranges) < self.max_execution_chunks:
                    padded_ranges.append((0, 0))
                data["ttt_policy_update_ranges"] = torch.tensor(padded_ranges, dtype=torch.long)
                data["ttt_policy_updates_per_execution_chunk"] = torch.tensor(
                    self.execution_chunk_interval // self.chunk_interval,
                    dtype=torch.long,
                )
            return data
        except Exception as e:
            print(f"Error processing TTT sample idx {idx}: {e}. Returning a random sample instead.")
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            return self.__getitem__(random_idx)
