"""一个可训练、可推理的精简完整 ACT 框架。

实现对应 ACT 的核心训练链路：

    observation -> Transformer decoder -> K 个动作
    observation + future actions -> CVAE posterior -> latent z
    reconstruction loss + beta * KL(q(z|o,a)||N(0,I))

训练时 posterior encoder 可以看到 demonstration 的未来动作；推理时没有动作标签，
因此使用 z=0（ACT 的标准做法）。这不是论文代码的逐行复刻，但保留了 ACT 的
CVAE、latent、Transformer encoder/decoder、action query 和 padding mask 机制。
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
BC_DIR = os.path.join(os.path.dirname(HERE), "bc")
for path in (HERE, BC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from action_chunking import masked_chunk_mse
from sequence_transformer import LIBEROSequenceChunkDataset


class LIBEROACTDataset(LIBEROSequenceChunkDataset):
    """ACT 使用的 HDF5 dataset。

    复用已经验证过的 sequence dataset：返回
    ``images[H,3,h,w]``、``states[H,state_dim]``、``actions[K,7]`` 和
    ``action_mask[K]``。mask=0 的尾部 padding 不参与重建损失。
    """


class ACT(nn.Module):
    """带 CVAE posterior 和 Transformer decoder 的小型 ACT。

    Args:
        history_length: 输入的历史观测帧数 H。
        chunk_size: 每次并行预测的动作数 K。
        latent_dim: CVAE latent z 的维度。
        d_model/num_heads/num_layers: Transformer decoder 配置。
    """

    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 7,
        history_length: int = 1,
        chunk_size: int = 10,
        latent_dim: int = 32,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        visual_representation: str = "global_pool",
        spatial_grid: int = 6,
        observation_mode: str = "both",
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")
        if visual_representation not in {"global_pool", "spatial_tokens"}:
            raise ValueError("visual_representation 必须是 global_pool 或 spatial_tokens")
        if observation_mode not in {"both", "image", "state"}:
            raise ValueError("observation_mode 必须是 both、image 或 state")
        if spatial_grid < 1:
            raise ValueError("spatial_grid 必须 >= 1")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_length = history_length
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.visual_representation = visual_representation
        self.spatial_grid = spatial_grid
        self.observation_mode = observation_mode

        # 保留 global_pool + both 的层命名和结构，以兼容已有 ACT checkpoint。
        if visual_representation == "global_pool":
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            )
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, 64), nn.ReLU(inplace=True),
            )
            if observation_mode == "both":
                input_dim = 128 + 64
            elif observation_mode == "image":
                input_dim = 128
            else:
                input_dim = 64
            self.observation_projection = nn.Sequential(
                nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
            )
            self.spatial_pool = None
            self.spatial_projection = None
            self.spatial_position = None
            self.state_token_projection = None
            self.observation_position = nn.Parameter(torch.zeros(1, history_length, d_model))
            nn.init.normal_(self.observation_position, std=0.02)
            tokens_per_frame = 1
            visual_token_count = 0
        else:
            self.image_encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 5, 2, 2), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
            )
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, 64), nn.ReLU(inplace=True),
            )
            self.spatial_pool = nn.AdaptiveAvgPool2d((spatial_grid, spatial_grid))
            self.spatial_projection = nn.Linear(128, d_model)
            self.spatial_position = nn.Parameter(
                torch.zeros(1, spatial_grid * spatial_grid, d_model)
            )
            self.state_token_projection = nn.Linear(64, d_model)
            nn.init.normal_(self.spatial_position, std=0.02)
            self.observation_position = nn.Parameter(torch.zeros(1, history_length, d_model))
            nn.init.normal_(self.observation_position, std=0.02)
            if observation_mode == "state":
                tokens_per_frame = 1
                visual_token_count = 0
            elif observation_mode == "image":
                tokens_per_frame = spatial_grid * spatial_grid
                visual_token_count = spatial_grid * spatial_grid
            else:
                tokens_per_frame = spatial_grid * spatial_grid + 1
                visual_token_count = spatial_grid * spatial_grid
        self.visual_token_count = visual_token_count
        self.tokens_per_frame = tokens_per_frame
        self.observation_token_count = history_length * tokens_per_frame

        # CVAE posterior q(z | observation, future action chunk)。
        self.posterior_action_projection = nn.Linear(action_dim, d_model)
        self.posterior_state_projection = nn.Linear(state_dim, d_model)
        self.posterior_cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.posterior_position = nn.Parameter(torch.zeros(1, chunk_size + 3, d_model))
        nn.init.normal_(self.posterior_cls, std=0.02)
        nn.init.normal_(self.posterior_position, std=0.02)
        posterior_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.posterior_encoder = nn.TransformerEncoder(posterior_layer, num_layers=num_layers)
        self.posterior_stats = nn.Linear(d_model, 2 * latent_dim)

        # decoder memory = observation tokens + 1 个 latent token。
        self.latent_projection = nn.Linear(latent_dim, d_model)
        self.memory_position = nn.Parameter(
            torch.zeros(1, self.observation_token_count + 1, d_model)
        )
        nn.init.normal_(self.memory_position, std=0.02)
        self.action_queries = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        self.action_position = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        nn.init.normal_(self.action_queries, std=0.02)
        nn.init.normal_(self.action_position, std=0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

    def encode_observation(self, images: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """把 ``[B,H,...]`` 观测编码成 decoder memory tokens。"""
        batch, history = images.shape[:2]
        if history != self.history_length:
            raise ValueError(f"需要 history_length={self.history_length}，实际得到 {history}")
        flat_images = images.reshape(batch * history, *images.shape[2:])
        flat_states = states.reshape(batch * history, -1)

        if self.visual_representation == "global_pool":
            image_features = self.image_encoder(flat_images)
            state_features = self.state_encoder(flat_states)
            if self.observation_mode == "both":
                inputs = torch.cat([image_features, state_features], dim=-1)
            elif self.observation_mode == "image":
                inputs = image_features
            else:
                inputs = state_features
            tokens = self.observation_projection(inputs).view(batch, history, self.d_model)
            return tokens + self.observation_position[:, :history]

        state_features = self.state_encoder(flat_states)
        frame_position = self.observation_position[:, :history].unsqueeze(2)
        if self.observation_mode == "state":
            tokens = self.state_token_projection(state_features).view(
                batch, history, 1, self.d_model
            )
            return (tokens + frame_position).reshape(batch, history, self.d_model)

        image_features = self.image_encoder(flat_images)
        image_features = self.spatial_pool(image_features)
        image_features = image_features.flatten(2).transpose(1, 2)
        visual_tokens = self.spatial_projection(image_features)
        visual_tokens = visual_tokens.view(
            batch, history, self.visual_token_count, self.d_model
        )
        visual_tokens = (
            visual_tokens
            + self.spatial_position[:, : self.visual_token_count].unsqueeze(1)
            + frame_position
        )
        if self.observation_mode == "image":
            return visual_tokens.reshape(batch, self.observation_token_count, self.d_model)

        # A state token is appended to each frame's spatial visual tokens.
        state_tokens = self.state_token_projection(state_features).view(
            batch, history, 1, self.d_model
        )
        state_tokens = state_tokens + frame_position
        tokens = torch.cat([visual_tokens, state_tokens], dim=2)
        return tokens.reshape(batch, self.observation_token_count, self.d_model)

    def posterior(self, observation_tokens: torch.Tensor, states: torch.Tensor, actions: torch.Tensor,
                  action_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 q(z|o,a) 的均值和 log variance。"""
        batch, chunk = actions.shape[:2]
        if chunk != self.chunk_size:
            raise ValueError(f"需要 chunk_size={self.chunk_size}，实际得到 {chunk}")
        cls = self.posterior_cls.expand(batch, -1, -1)
        # posterior 也看到视觉观测；否则它只能根据动作本身编码 z，
        # 不符合 q(z | observation, action) 的条件 VAE 定义。
        observation_token = observation_tokens.mean(dim=1, keepdim=True)
        # 使用最新一帧 state 作为 qpos token；未来动作 token 提供 demonstration 信息。
        state_token = self.posterior_state_projection(states[:, -1]).unsqueeze(1)
        action_tokens = self.posterior_action_projection(actions)
        tokens = torch.cat([cls, observation_token, state_token, action_tokens], dim=1)
        tokens = tokens + self.posterior_position[:, : tokens.size(1)]
        key_padding = None
        if action_mask is not None:
            key_padding = torch.cat([
                torch.zeros(batch, 3, dtype=torch.bool, device=actions.device),
                ~action_mask.bool(),
            ], dim=1)
        encoded = self.posterior_encoder(tokens, src_key_padding_mask=key_padding)
        mean, logvar = self.posterior_stats(encoded[:, 0]).chunk(2, dim=-1)
        return mean, logvar.clamp(-10.0, 10.0)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, observation_tokens: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        batch = observation_tokens.size(0)
        latent_token = self.latent_projection(latent).unsqueeze(1)
        memory = torch.cat([observation_tokens, latent_token], dim=1)
        memory = memory + self.memory_position[:, : memory.size(1)]
        queries = self.action_queries.expand(batch, -1, -1) + self.action_position
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ):
        """训练返回 ``pred, mean, logvar``；推理 actions=None 时使用 z=0。"""
        observation_tokens = self.encode_observation(images, states)
        if actions is not None:
            mean, logvar = self.posterior(observation_tokens, states, actions, action_mask)
            latent = self.reparameterize(mean, logvar)
        else:
            batch = images.size(0)
            mean = images.new_zeros(batch, self.latent_dim)
            logvar = images.new_zeros(batch, self.latent_dim)
            latent = mean
        prediction = self.decode(observation_tokens, latent)
        return prediction, mean, logvar

    @torch.no_grad()
    def predict(self, images: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """推理接口：不需要未来动作，返回归一化的 ``[B,K,action_dim]``。"""
        prediction, _, _ = self(images, states, actions=None, action_mask=None)
        return prediction


def act_loss(prediction: torch.Tensor, target: torch.Tensor, action_mask: torch.Tensor,
             mean: torch.Tensor, logvar: torch.Tensor, kl_weight: float = 10.0):
    """ACT 的重建 MSE、KL 和总 loss。"""
    reconstruction = masked_chunk_mse(prediction, target, action_mask)
    kl = -0.5 * (1.0 + logvar - mean.pow(2) - logvar.exp()).sum(dim=-1).mean()
    total = reconstruction + kl_weight * kl / max(mean.size(-1), 1)
    return total, reconstruction, kl


__all__ = ["LIBEROACTDataset", "ACT", "act_loss"]
