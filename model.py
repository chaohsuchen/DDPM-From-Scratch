"""
The noise predictor eps_theta(x_t, t).

For 2-D points a small MLP is enough. The only "trick" is telling the network
which timestep it is at: t is turned into a sinusoidal embedding (same idea as
Transformer position encodings) and fed in alongside x_t.
"""

import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    """t (integer) -> [sin(t * w_1), ..., sin(t * w_k), cos(t * w_1), ..., cos(t * w_k)]."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden))

    def forward(self, h):
        return h + self.net(h)


class NoisePredictor(nn.Module):
    """eps_theta(x_t, t): input a noisy point and its timestep, output the predicted noise."""

    def __init__(self, data_dim=2, hidden=256, n_blocks=4, t_dim=64):
        super().__init__()
        self.t_embed = nn.Sequential(SinusoidalEmbedding(t_dim), nn.Linear(t_dim, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden))
        self.x_in = nn.Linear(data_dim, hidden)
        self.blocks = nn.ModuleList([ResidualBlock(hidden) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, data_dim))

    def forward(self, x_t, t):
        h = self.x_in(x_t) + self.t_embed(t)
        for block in self.blocks:
            h = block(h)
        return self.out(h)
