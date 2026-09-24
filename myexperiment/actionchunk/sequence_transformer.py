"""小型时序 Transformer：历史观测序列 -> 未来 action chunk。"""
from __future__ import annotations

import os
import sys
from typing import List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
BC_DIR = os.path.join(os.path.dirname(HERE), "bc")
for path in (HERE, BC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from bc import NormalizationStats, PathLike, prepare_demo_tensors
from action_chunking import masked_chunk_mse


class LIBEROSequenceChunkDataset(Dataset):
    """返回历史 H 帧观测、未来 K 个动作和动作有效 mask。

    初始化时一次性读取、resize、归一化指定 demos 并缓存到内存，
    __getitem__ 只做内存索引，避免每个样本重复打开 HDF5 并读取整条轨迹。
    """

    def __init__(
        self,
        hdf5_path: PathLike,
        demo_names: Sequence[str],
        stats: NormalizationStats,
        history_length: int = 5,
        chunk_size: int = 10,
        image_size: int = 84,
    ):
        if history_length < 1 or chunk_size < 1:
            raise ValueError("history_length 和 chunk_size 必须 >= 1")
        self.hdf5_path = str(hdf5_path)
        self.stats = stats
        self.history_length = history_length
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

        # 历史不足 H 帧时重复最早一帧，保持张量尺寸固定。
        # images/states/actions 已在初始化时 resize 并归一化。
        history_indices = [max(0, timestep - self.history_length + 1 + i)
                           for i in range(self.history_length)]
        image_batch = images[history_indices]
        state_batch = states[history_indices]

        end = min(timestep + self.chunk_size, len(actions))
        valid = end - timestep
        chunk = torch.empty((self.chunk_size, actions.shape[-1]), dtype=actions.dtype)
        mask = torch.zeros(self.chunk_size, dtype=torch.float32)
        chunk[:valid] = actions[timestep:end]
        chunk[valid:] = actions[end - 1]
        mask[:valid] = 1.0
        return (
            image_batch,
            state_batch,
            chunk,
            mask,
        )


class SmallSequenceTransformer(nn.Module):
    """用历史观测 token 和 K 个 action query 预测未来动作序列。

    这比 ``SmallChunkBC`` 多了一层真正的序列建模：Transformer 不仅在
    H 个历史观测之间做注意力，K 个可学习 action query 之间也可以互相
    交换信息，并共同读取历史观测。每个 query 最终负责一个未来时间步。
    """

    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 7,
        history_length: int = 5,
        chunk_size: int = 10,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.history_length = history_length
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(inplace=True),
        )
        self.token_projection = nn.Sequential(
            nn.Linear(128 + 64, d_model), nn.LayerNorm(d_model), nn.ReLU(inplace=True)
        )
        # 两类可学习 token：历史位置编码 + 未来 K 步 action query。
        self.observation_position_embedding = nn.Parameter(
            torch.zeros(1, history_length, d_model)
        )
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        nn.init.normal_(self.observation_position_embedding, std=0.02)
        nn.init.normal_(self.action_queries, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        # 对每个 action query 独立映射为一个 7 维动作，而不是展平输出 K*7。
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

    def forward(self, images: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        batch, history = images.shape[:2]
        image_features = self.image_encoder(images.reshape(batch * history, *images.shape[2:]))
        state_features = self.state_encoder(states.reshape(batch * history, -1))
        observation_tokens = self.token_projection(
            torch.cat([image_features, state_features], dim=-1)
        )
        observation_tokens = observation_tokens.view(batch, history, -1)
        observation_tokens = (
            observation_tokens + self.observation_position_embedding[:, :history]
        )

        # [H 个观测 token, K 个动作 query] 一起进入 Transformer。action query
        # 没有包含真实未来动作，因此不会把标签泄漏给模型。
        action_queries = self.action_queries.expand(batch, -1, -1)
        tokens = torch.cat([observation_tokens, action_queries], dim=1)
        encoded = self.transformer(tokens)
        action_tokens = encoded[:, -self.chunk_size:]
        return self.action_head(action_tokens)


__all__ = ["LIBEROSequenceChunkDataset", "SmallSequenceTransformer", "masked_chunk_mse"]
