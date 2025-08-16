# src/models/fusion/cross_modal_transformer.py
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config_utils import ConfigManager
from src.models.affective_forensics import AffectiveForensics
from src.models.semantic_forgery import SemanticForgeryAnalyzer
from src.models.chronos_guard import ChronosGuard


# -----------------------------
# Forensic Co-Attention (evidence-gated)
# -----------------------------

class ForensicCoAttention(nn.Module):
    """
    Evidence-gated co-attention over two modality vectors.
    Given x (B,H) and y (B,H), we compute:
      - q = Wq x, k = Wk y, v = Wv y
      - attn = sigmoid( (q·k)/sqrt(H) )
      - gate = sigmoid(Ge * evidence)  -> (B,1)
      - out = gate * (attn * v) + (1-gate) * (x + y)/2
    Returns (B,H).
    """
    def __init__(self, hidden_dim: int, evidence_dim: int = 3):
        super().__init__()
        self.h = hidden_dim
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.evidence_proj = nn.Sequential(
            nn.Linear(evidence_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor, evidence: torch.Tensor) -> torch.Tensor:
        """
        x: (B,H), y: (B,H), evidence: (B,E)
        """
        B, H = x.shape
        q = self.q(x)               # (B,H)
        k = self.k(y)               # (B,H)
        v = self.v(y)               # (B,H)
        scale = (H ** 0.5)
        score = (q * k).sum(dim=-1, keepdim=True) / scale     # (B,1)
        attn = torch.sigmoid(score)                           # (B,1) ∈ (0,1)
        gated = torch.sigmoid(self.evidence_proj(evidence))   # (B,1)

        attended = attn * v                                   # (B,H), broadcast
        base = 0.5 * (x + y)
        out = gated * attended + (1.0 - gated) * base         # (B,H)
        return out


# -----------------------------
# Cross-Modal Transformer (fusion)
# -----------------------------

class CrossModalTransformer(nn.Module):
    """
    Fuses modality vectors (text/audio/visual/temporal) with evidence-gated co-attention.
    Accepts optional GNN feature vector per sample.

    Forward expects a dict with:
      {
        "text_features":     (B, 768),
        "audio_features":    (B, 128),
        "visual_features":   (B, 512),
        "temporal_features": (B, 256),
        # optional:
        "gnn_feat":          (B, gnn_dim)    # from graph module, if available
      }
    Returns:
      {
        "fused":  (B, F_after_mlp=hidden),
        "logits": (B, 2),
        "forensic": { "emotion_intensity":..., "semantic_conflict":..., "temporal_delay":... }
      }
    """
    def __init__(self, config_path: str = "configs/model_configs/fusion.yaml"):
        super().__init__()
        cfg = ConfigManager().load_config(config_path)
        self.hidden = int(cfg.get("hidden_dim", 512))
        self.dropout = float(cfg.get("dropout", 0.3))  # Increased dropout rate
        self.use_gnn = bool(cfg.get("use_gnn", True))
        self.gnn_dim = int(cfg.get("gnn_dim", 128))

        # Device & dtype
        self.dtype = torch.float32
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        # Projections to shared space
        self.text_proj     = nn.Linear(768, self.hidden, dtype=self.dtype)
        self.audio_proj    = nn.Linear(128, self.hidden, dtype=self.dtype)
        self.visual_proj   = nn.Linear(512, self.hidden, dtype=self.dtype)
        self.temporal_proj = nn.Linear(256, self.hidden, dtype=self.dtype)

        if self.use_gnn:
            self.gnn_proj = nn.Linear(self.gnn_dim, self.hidden, dtype=self.dtype)

        # Evidence modules (kept here for potential future use)
        self.affective = AffectiveForensics()
        self.semantic  = SemanticForgeryAnalyzer()
        self.chronos   = ChronosGuard()

        # Forensic co-attention blocks
        self.attn_tv = ForensicCoAttention(self.hidden, evidence_dim=3)
        self.attn_ta = ForensicCoAttention(self.hidden, evidence_dim=3)
        self.attn_vu = ForensicCoAttention(self.hidden, evidence_dim=3)

        # Fusion head
        self.include_pairs = True
        base = 4
        pairs = 8
        co = 3
        gnn = 1 if self.use_gnn else 0
        self.fused_dim = (base + (pairs if self.include_pairs else 0) + co + gnn) * self.hidden

        self.fuse_mlp = nn.Sequential(
            nn.Linear(self.fused_dim, 2 * self.hidden, dtype=self.dtype),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(2 * self.hidden, self.hidden, dtype=self.dtype),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.classifier = nn.Linear(self.hidden, 2, dtype=self.dtype)

        self.to(self.device)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        feats: dict of modality features (B, dim)
        """
        # Move to device/dtype
        def prep(x): return x.to(self.device, dtype=self.dtype)

        t = prep(feats["text_features"])       # (B,768)
        a = prep(feats["audio_features"])      # (B,128)
        v = prep(feats["visual_features"])     # (B,512)
        u = prep(feats["temporal_features"])   # (B,256)

        # Project
        t = self.text_proj(t)      # (B,H)
        a = self.audio_proj(a)     # (B,H)
        v = self.visual_proj(v)    # (B,H)
        u = self.temporal_proj(u)  # (B,H)

        # --- Evidence scalars (computed from current projected features) ---
        with torch.no_grad():
            # cosine mapped to [0,1]
            def cos01(x1, x2):
                xn = F.normalize(x1, dim=-1)
                yn = F.normalize(x2, dim=-1)
                c = (xn * yn).sum(dim=-1, keepdim=True)
                return 0.5 * (c.clamp(-1, 1) + 1.0)

            semantic_sim = cos01(t, v)                 # (B,1)
            semantic_conflict = 1.0 - semantic_sim     # (B,1)
            emo_proxy = (t.abs().mean(dim=-1, keepdim=True)).tanh()  # (B,1)
            delay_proxy = 1.0 - cos01(t, u)            # (B,1)

        # --- Co-attention with evidence ---
        tv_star = self.attn_tv(t, v, torch.cat([semantic_conflict, emo_proxy, torch.zeros_like(emo_proxy)], dim=-1))
        ta_star = self.attn_ta(t, a, torch.cat([emo_proxy, torch.zeros_like(emo_proxy), torch.zeros_like(emo_proxy)], dim=-1))
        vu_star = self.attn_vu(v, u, torch.cat([delay_proxy, torch.zeros_like(emo_proxy), torch.zeros_like(emo_proxy)], dim=-1))

        # --- Pairwise interactions ---
        if self.include_pairs:
            feats_pairs = [
                t + a, t * a, (t - a).abs(),
                t + v, t * v, (t - v).abs(),
                t + u, v + u
            ]
            pairs_cat = torch.cat(feats_pairs, dim=-1)  # (B, 8H)
        else:
            pairs_cat = torch.empty(t.shape[0], 0, device=self.device, dtype=self.dtype)

        # Optional GNN path
        gnn_cat = []
        if self.use_gnn and ("gnn_feat" in feats) and feats["gnn_feat"] is not None:
            g = prep(feats["gnn_feat"])                # (B, gnn_dim)
            g = self.gnn_proj(g)                       # (B,H)
            gnn_cat = [g]

        # Concatenate everything
        fused_cat = torch.cat([
            t, a, v, u,                 # 4H
            pairs_cat,                  # 8H
            tv_star, ta_star, vu_star,  # 3H
            *gnn_cat                    # H (optional)
        ], dim=-1)                      # (B, F)

        fused = self.fuse_mlp(fused_cat)    # (B, H)
        logits = self.classifier(fused)     # (B, 2)

        forensic = {
            "emotion_intensity": emo_proxy.squeeze(-1),    # (B,)
            "semantic_conflict": semantic_conflict.squeeze(-1),
            "temporal_delay": delay_proxy.squeeze(-1),
        }

        return {
            "fused": fused,
            "logits": logits,
            "forensic": forensic
        }
