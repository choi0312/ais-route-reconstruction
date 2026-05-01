from __future__ import annotations

import torch
import torch.nn as nn


class TCMBlock(nn.Module):
    def __init__(self, hidden_dim: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.temporal_mixer = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, seq_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len * 2, seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
        x = residual + y
        x = x + self.channel_mixer(x)
        x_t = x.transpose(1, 2)
        x_t = x_t + self.temporal_mixer(x_t)
        return x_t.transpose(1, 2)


class BiTCMDeltaModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        seq_len: int = 20,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.forward_blocks = nn.ModuleList(
            [TCMBlock(hidden_dim, seq_len, dropout) for _ in range(num_layers)]
        )
        self.backward_blocks = nn.ModuleList(
            [TCMBlock(hidden_dim, seq_len, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward_stream(self, x: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        h = self.proj(x)
        for block in blocks:
            h = block(h)
        return h[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fwd = self.forward_stream(x, self.forward_blocks)
        bwd = self.forward_stream(torch.flip(x, dims=[1]), self.backward_blocks)
        h = 0.5 * (fwd + bwd)
        return self.head(h)


class GraphTemporalRefiner(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        radius: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.radius = radius
        self.theta = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def build_adj(self, T: int, device) -> torch.Tensor:
        adj = torch.zeros(T, T, device=device)
        for i in range(T):
            for j in range(max(0, i - self.radius), min(T, i + self.radius + 1)):
                adj[i, j] = 1.0 / (1.0 + abs(i - j))
        adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.theta(x)
        B, T, _ = h.shape
        adj = self.build_adj(T, h.device)
        g = torch.einsum("ij,bjh->bih", adj, h)
        a, _ = self.attn(g, g, g)
        return self.head(a[:, -1, :])


class HybridDeltaModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        seq_len: int = 20,
        graph_radius: int = 3,
        tcn_weight: float = 0.6,
        graph_weight: float = 0.4,
    ):
        super().__init__()
        self.tcn = BiTCMDeltaModel(input_dim, hidden_dim, seq_len)
        self.graph = GraphTemporalRefiner(input_dim, hidden_dim, graph_radius)
        self.tcn_weight = tcn_weight
        self.graph_weight = graph_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_tcn = self.tcn(x)
        delta_graph = self.graph(x)
        return self.tcn_weight * delta_tcn + self.graph_weight * delta_graph
