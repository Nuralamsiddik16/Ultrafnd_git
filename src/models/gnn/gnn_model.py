# src/models/gnn/gnn_model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNModel(nn.Module):
    """
    Minimal 2-layer GCN compatible with the rest of the codebase.
    Not wired into the trainer (the trainer uses its own SimpleGCN),
    but this prevents import errors where GNNModel is referenced.
    """
    def __init__(self, in_dim: int, hid: int = 256, out_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hid)
        self.lin2 = nn.Linear(hid, out_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()  # Using ReLU activation for better non-linearity

    def _norm_adj(self, A: torch.Tensor) -> torch.Tensor:
        # A_hat = A + I
        I = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        A_hat = A + I
        deg = A_hat.sum(dim=-1).clamp_min(1e-9)
        D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
        return D_inv_sqrt @ A_hat @ D_inv_sqrt

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        X: (N, F) node features
        A: (N, N) adjacency (0/1 or weighted)
        """
        A_norm = self._norm_adj(A)
        H = F.relu(A_norm @ self.lin1(X))  # Applying ReLU after the first layer
        H = self.drop(H)
        Z = self.lin2(A_norm @ H)
        return Z  # (N, out_dim)
