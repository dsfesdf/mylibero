"""最小视觉 Behavior Cloning (BC) 实现。

从 LIBERO HDF5 demonstrations 读取 (image, state, action)，使用小型 CNN + MLP
预测当前时刻的 7 维动作。Dataset 默认不保留 h5py 文件句柄，避免多进程 pickle 错误。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


@dataclass
class NormalizationStats:
    """训练集统计量；验证和 rollout 必须复用同一组统计量。"""
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def to_dict(self) -> Dict[str, np.ndarray]:
        return self.__dict__.copy()


PathLike = Union[str, Path]


def list_demos(hdf5_path: PathLike) -> List[str]:
    """返回 HDF5 中按 demo 编号排序的轨迹名。"""
    with h5py.File(hdf5_path, "r") as handle:
        return sorted(handle["data"].keys(), key=lambda name: int(name.split("_")[-1]))


def read_demo_arrays(hdf5_path: PathLike, demo_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取一个 demo，并将 h5py 数据复制成普通 NumPy 数组。"""
    with h5py.File(hdf5_path, "r") as handle:
        demo = handle["data"][demo_name]
        images = np.asarray(demo["obs"]["agentview_rgb"], dtype=np.uint8)
        joint_states = np.asarray(demo["obs"]["joint_states"], dtype=np.float32)
        gripper_states = np.asarray(demo["obs"]["gripper_states"], dtype=np.float32)
        actions = np.asarray(demo["actions"], dtype=np.float32)
    states = np.concatenate([joint_states, gripper_states], axis=-1)
    if not (len(images) == len(states) == len(actions)):
        raise ValueError(f"时间长度不一致: images={len(images)}, states={len(states)}, actions={len(actions)}")
    return images, states, actions


def compute_stats(hdf5_path: PathLike, demo_names: Sequence[str]) -> NormalizationStats:
    """只用训练 demos 计算 state/action 的均值和标准差。"""
    states, actions = [], []
    for demo_name in demo_names:
        _, state, action = read_demo_arrays(hdf5_path, demo_name)
        states.append(state)
        actions.append(action)
    state_all, action_all = np.concatenate(states), np.concatenate(actions)
    return NormalizationStats(
        state_mean=state_all.mean(0).astype(np.float32),
        state_std=np.maximum(state_all.std(0), 1e-6).astype(np.float32),
        action_mean=action_all.mean(0).astype(np.float32),
        action_std=np.maximum(action_all.std(0), 1e-6).astype(np.float32),
    )


class LIBEROFrameDataset(Dataset):
    """将指定 demos 展平为逐帧监督样本。"""
    def __init__(self, hdf5_path: PathLike, demo_names: Sequence[str], stats: NormalizationStats, image_size: int = 84):
        self.hdf5_path, self.stats, self.image_size = str(hdf5_path), stats, image_size
        self.index: List[Tuple[str, int]] = []
        for demo_name in demo_names:
            # 只读取长度，不保存 h5py.File 对象。
            with h5py.File(self.hdf5_path, "r") as handle:
                length = handle["data"][demo_name]["actions"].shape[0]
            self.index.extend((demo_name, t) for t in range(length))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        demo_name, timestep = self.index[index]
        images, states, actions = read_demo_arrays(self.hdf5_path, demo_name)
        # HWC uint8 -> CHW float32，并把图像缩放到 [0, 1] 和 84x84。
        image = torch.from_numpy(images[timestep]).permute(2, 0, 1).float() / 255.0
        image = torch.nn.functional.interpolate(image[None], (self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
        state = (states[timestep] - self.stats.state_mean) / self.stats.state_std
        action = (actions[timestep] - self.stats.action_mean) / self.stats.action_std
        return image, torch.from_numpy(state).float(), torch.from_numpy(action).float()


class SmallVisualBC(nn.Module):
    """CNN 提取图像特征，MLP 提取状态特征，拼接后回归 action。"""
    def __init__(self, state_dim: int = 9, action_dim: int = 7):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
        ##更大池化
        #     nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        # )
        # self.state_encoder = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True))
        # self.action_head = nn.Sequential(nn.Linear(2112, 128), nn.ReLU(inplace=True), nn.Linear(128, action_dim))

        ##2*2池化
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.action_head = nn.Sequential(nn.Linear(576, 128), nn.ReLU(inplace=True), nn.Linear(128, action_dim))
        #     nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        # )
        # self.state_encoder = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, 64), nn.ReLU(inplace=True))
        # self.action_head = nn.Sequential(nn.Linear(192, 128), nn.ReLU(inplace=True), nn.Linear(128, action_dim))


    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.action_head(torch.cat([self.image_encoder(image), self.state_encoder(state)], dim=-1))
