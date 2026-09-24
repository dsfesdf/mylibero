"""最小 action chunking 组件。

与原有单步 BC 不同，这里的一个样本是：

    当前图像 + 当前状态 -> 未来 K 个动作

轨迹末尾不足 K 帧时，用最后一帧动作占位，同时返回 mask；loss 只计算真实动作，
不会让补齐动作影响训练。
"""
from __future__ import annotations

import os
import sys
from typing import List, Sequence, Tuple

# 让本文件既能作为包内模块，也能直接运行时找到 bc/ 目录下的 bc.py。
_HERE = os.path.dirname(os.path.abspath(__file__))
_BC_DIR = os.path.join(os.path.dirname(_HERE), "bc")
for _path in (_HERE, _BC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from bc import NormalizationStats, PathLike, prepare_demo_tensors


class LIBEROChunkDataset(Dataset):
    """将 LIBERO demos 转为 (image_t, state_t, action[t:t+K], mask) 样本。"""

    def __init__(
        self,
        hdf5_path: PathLike,
        demo_names: Sequence[str],
        stats: NormalizationStats,
        chunk_size: int = 10,
        image_size: int = 84,
    ):
        if chunk_size < 1:
            raise ValueError("chunk_size 必须 >= 1")
        self.hdf5_path = str(hdf5_path)
        self.stats = stats
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.demos = {}
        self.index: List[Tuple[str, int]] = []
        for demo_name in demo_names:
            self.demos[demo_name] = prepare_demo_tensors(
                self.hdf5_path, demo_name, self.stats, self.image_size
            )
            length = self.demos[demo_name][2].shape[0]
            self.index.extend((demo_name, t) for t in range(length))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        demo_name, timestep = self.index[index]
        images, states, actions = self.demos[demo_name]
        image = images[timestep]
        state = states[timestep]

        # 轨迹末尾用最后动作填充，但 mask=0，训练 loss 会忽略这些位置。
        end = min(timestep + self.chunk_size, len(actions))
        valid = end - timestep
        chunk = torch.empty((self.chunk_size, actions.shape[-1]), dtype=actions.dtype)
        mask = torch.zeros(self.chunk_size, dtype=torch.float32)
        chunk[:valid] = actions[timestep:end]
        chunk[valid:] = actions[end - 1]
        mask[:valid] = 1.0
        return (
            image,
            state,
            chunk,
            mask,
        )


class SmallChunkBC(nn.Module):
    """复用小 CNN/MLP 编码器，一次输出 chunk_size * action_dim 个动作值。"""

    def __init__(self, state_dim: int = 9, action_dim: int = 7, chunk_size: int = 10, pool_size: int = 2):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.pool_size = pool_size
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((pool_size, pool_size)), nn.Flatten(),
        )
        image_dim = 128 * pool_size * pool_size
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
        )
        self.action_head = nn.Sequential(
            nn.Linear(image_dim + 64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, chunk_size * action_dim),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        features = torch.cat([self.image_encoder(image), self.state_encoder(state)], dim=-1)
        return self.action_head(features).view(-1, self.chunk_size, self.action_dim)


def masked_chunk_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """只在真实未来动作位置计算 MSE。"""
    squared_error = (prediction - target).pow(2).mean(dim=-1)
    return (squared_error * mask).sum() / mask.sum().clamp_min(1.0)
