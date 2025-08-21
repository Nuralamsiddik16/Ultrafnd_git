# src/models/gnn/gnn_model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSageLayer(nn.Module):
    """A basic mean-aggregator GraphSAGE layer."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        deg = A_hat.sum(dim=1, keepdim=True).clamp_min(1)
        neigh = A_hat @ X / deg
        H = torch.cat([X, neigh], dim=-1)
        return self.linear(H)


class GNNModel(nn.Module):
    """Multi-layer GraphSAGE-style network with dropout and ReLU."""

    def __init__(
        self,
        in_dim: int,
        hid: int = 256,
        out_dim: int = 128,
        dropout: float = 0.2,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hid] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [GraphSageLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        X: (N, F) node features
        A: (N, N) adjacency (0/1 or weighted)
        Returns embeddings of shape (N, out_dim).
        """
        I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        A_hat = A + I
        H = X
        for layer in self.layers[:-1]:
            H = layer(H, A_hat)
            H = self.act(H)
            H = self.drop(H)
        H = self.layers[-1](H, A_hat)
        return H
