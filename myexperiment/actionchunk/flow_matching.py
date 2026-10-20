"""条件 rectified flow：高斯噪声 -> 连续动作 chunk。

x_t=(1-t)*noise+t*action，监督速度为 action-noise。
训练损失只覆盖真实动作；padding 也在注意力中屏蔽，避免尾部标签泄漏。
"""
import math

import torch
from torch import nn


class FlowMatchingHead(nn.Module):
    def __init__(self, action_dim, chunk_size, d_model, num_heads, num_layers, dropout):
        super().__init__()
        self.action_dim, self.chunk_size = action_dim, chunk_size
        self.action_projection = nn.Linear(action_dim, d_model)
        self.position = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)
        self.time_projection = nn.Sequential(nn.Linear(32, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), 16)))
        layer = nn.TransformerDecoderLayer(d_model, num_heads, 4 * d_model, dropout, activation="gelu", batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.velocity = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, action_dim))

    def forward(self, noisy_actions, time, memory, mask=None):
        phase = time[:, None].float() * self.frequencies[None]
        embedding = self.time_projection(torch.cat([phase.sin(), phase.cos()], dim=-1))
        tokens = self.action_projection(noisy_actions) + self.position + embedding[:, None]
        padding = ~mask.bool() if mask is not None else None
        return self.velocity(self.decoder(tokens, memory, tgt_key_padding_mask=padding))

    def loss(self, actions, mask, memory):
        if not mask.bool().any(dim=1).all():
            raise ValueError("每个样本至少需要一个有效动作")
        # AMP 下插值与平方误差保持 float32，以避免大速度误差溢出。
        actions = actions.float()
        noise = torch.randn_like(actions)
        time = torch.rand(actions.size(0), device=actions.device)
        point = (1 - time[:, None, None]) * noise + time[:, None, None] * actions
        point = point * mask[..., None]
        prediction = self(point, time, memory, mask)
        error = (prediction.float() - (actions - noise)).square().mean(-1)
        return (error * mask).sum() / mask.sum().clamp_min(1)

    def sample(self, memory, steps=10, noise=None):
        """Euler 积分；可传固定 noise 做可重复的离线对照。"""
        if steps < 1:
            raise ValueError("flow_steps 必须 >= 1")
        shape = (memory.size(0), self.chunk_size, self.action_dim)
        actions = torch.randn(shape, device=memory.device, dtype=memory.dtype) if noise is None else noise
        for index in range(steps):
            time = torch.full((memory.size(0),), index / steps, device=memory.device)
            actions = actions + self(actions, time, memory) / steps
        return actions
