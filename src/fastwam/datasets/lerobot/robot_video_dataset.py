import hashlib
import json
import os
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
    """

    def __init__(
        self,
        *args,
        ttt_metadata_path: Optional[str] = None,
        ttt_mode: Optional[str] = None,
        tries_per_group: int = 6,
        observe_chunks: int = 20,
        chunk_interval: int = 10,
        entry_repeat: int = 1,
        **kwargs,
    ):
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
        self.entry_repeat = int(entry_repeat)
        if self.tries_per_group <= 0:
            raise ValueError("`tries_per_group` must be positive.")
        if self.observe_chunks <= 0:
            raise ValueError("`observe_chunks` must be positive.")
        if self.chunk_interval <= 0:
            raise ValueError("`chunk_interval` must be positive.")
        if self.entry_repeat <= 0:
            raise ValueError("`entry_repeat` must be positive.")
        self.ttt_entries = self._build_ttt_entries()
        if self.entry_repeat > 1:
            self.ttt_entries = self.ttt_entries * self.entry_repeat
        self.max_execution_chunks = max(
            len(entry.get("execution_offsets", [])) for entry in self.ttt_entries
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

    def _build_ttt_entries(self) -> list[dict]:
        groups = list(self.ttt_metadata.get("groups") or [])
        entries: list[dict] = []
        if self.ttt_mode == "repeat_attempt":
            for group in groups:
                episode_indices = [int(x) for x in group.get("episode_indices", [])[: self.tries_per_group]]
                if len(episode_indices) < self.tries_per_group:
                    continue
                max_start = self._max_valid_start(episode_indices)
                offsets = list(range(0, max_start + 1, self.chunk_interval)) or [0]
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
                observe_frames = int(group.get("observe_frames", default_observe_frames))
                max_start = self._max_valid_start([episode_index])
                start = min(max(observe_frames, 0), max_start)
                offsets = list(range(start, max_start + 1, self.chunk_interval)) or [start]
                entries.append(
                    {
                        "mode": "observe_then_act",
                        "group_id": int(group.get("group_id", len(entries))),
                        "episode_index": int(episode_index),
                        "offset": int(offsets[0]),
                        "execution_offsets": [int(offset) for offset in offsets],
                        "observe_frames": int(observe_frames),
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
        data["prompt"] = [sequence[0].get("prompt", "") for sequence in try_sequences]
        data["ttt_try_chunk_mask"] = torch.tensor(masks, dtype=torch.bool)
        data["ttt_try_start_offsets"] = torch.tensor(start_offsets, dtype=torch.long)
        if "restart_context" in try_sequences[0][0]:
            data["restart_context"] = try_sequences[0][0]["restart_context"]
            data["restart_context_mask"] = try_sequences[0][0]["restart_context_mask"]
            data["restart_prompt"] = try_sequences[0][0].get("restart_prompt", "")
        return data

    def _get_observe_warmup(self, episode_index: int, observe_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames = []
        proprios = []
        max_offset = max(int(observe_frames) - 1, 0)
        offsets = [min(i * self.chunk_interval, max_offset) for i in range(self.observe_chunks)]
        for offset in offsets:
            sample = self._get(self._global_index(episode_index, offset))
            frames.append(sample["video"][:, 0])
            proprios.append(sample["proprio"][0])
        return torch.stack(frames, dim=0), torch.stack(proprios, dim=0)

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

            execution_offsets = [int(offset) for offset in entry.get("execution_offsets", [entry["offset"]])]
            execution_samples = [
                self._get(self._global_index(entry["episode_index"], offset))
                for offset in execution_offsets
            ]
            data = execution_samples[0]
            warmup_video, warmup_proprio = self._get_observe_warmup(
                entry["episode_index"],
                entry["observe_frames"],
            )
            data.update(self._stack_execution_samples(execution_samples))
            data["ttt_warmup_video"] = warmup_video
            data["ttt_warmup_proprio"] = warmup_proprio
            data["ttt_sequence_mode_id"] = torch.tensor(2, dtype=torch.long)
            data["ttt_group_id"] = torch.tensor(entry["group_id"], dtype=torch.long)
            data["ttt_chunk_offset"] = torch.tensor(entry["offset"], dtype=torch.long)
            padded_offsets = list(execution_offsets)
            while len(padded_offsets) < self.max_execution_chunks:
                padded_offsets.append(execution_offsets[-1])
            data["ttt_execution_offsets"] = torch.tensor(padded_offsets, dtype=torch.long)
            return data
        except Exception as e:
            print(f"Error processing TTT sample idx {idx}: {e}. Returning a random sample instead.")
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            return self.__getitem__(random_idx)
