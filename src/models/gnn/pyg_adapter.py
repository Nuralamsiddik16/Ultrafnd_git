# src/models/gnn/pyg_adapter.py
"""
PyTorch Geometric scaffold — use on your Mac for the hetero graph version.
"""
try:
    import torch
    from torch import nn
    from torch_geometric.nn import SAGEConv, to_hetero
    TORCH_PYG_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_PYG_AVAILABLE = False

if TORCH_PYG_AVAILABLE:
    class PostEncoder(nn.Module):
        def __init__(self, in_dim, hid=256, out_dim=256, layers=2, dropout=0.3):
            super().__init__()
            self.convs = nn.ModuleList([SAGEConv(in_dim, hid)] + [SAGEConv(hid, hid) for _ in range(layers - 1)])
            self.proj = nn.Linear(hid, out_dim)
            self.dropout = nn.Dropout(dropout)
            self.act = nn.ReLU()  # ReLU for non-linearity

        def forward(self, x, edge_index):
            h = x
            for conv in self.convs:
                h = self.act(conv(h, edge_index))
                h = self.dropout(h)
            return self.proj(h)

    class HeteroFGHGNN(nn.Module):
        """
        Heterogeneous FG-HGNN (posts/phrases/sources) — build with torch_geometric.to_hetero
        """
        def __init__(self, metadata, dims):
            super().__init__()
            self.base = PostEncoder(dims["post_in"], hid=dims["hid"], out_dim=dims["out"], layers=dims["layers"])
            self.model = to_hetero(self.base, metadata=metadata)

        def forward(self, x_dict, edge_index_dict):
            h_dict = self.model(x_dict, edge_index_dict)
            return h_dict["post"]

else:
    # Placeholders to keep imports error-free where PyG isn't installed
    class PostEncoder: ...
    class HeteroFGHGNN: ...
