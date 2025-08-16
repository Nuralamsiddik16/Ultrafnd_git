# src/training/forensic_trainer.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from src.data_pipeline.fakesv_dataset import FakeSVRawDataset, build_gnn_cache_from_raw_dataset
from src.models.fusion.cross_modal_transformer import CrossModalTransformer
from src.models.fusion.deep_truth_classifier import DeepTruthClassifier
from src.training.metrics.forensic_metrics import (
    aggregate_epoch_metrics,
    pretty_print,
)

# -----------------------------
# Small, dependency-free GCN
# -----------------------------

class SimpleGCN(nn.Module):
    """
    Two-layer GCN over post nodes. No PyG dependency.
    Input: X (N, F), adjacency A (N, N) sparse/dense 0/1.
    """
    def __init__(self, in_dim: int, hid: int = 128, out_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hid)
        self.lin2 = nn.Linear(hid, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   (N, F)
        adj: (N, N) binary/float, will be sym-norm’d inside
        """
        # A_hat = A + I
        N = adj.shape[0]
        device = x.device
        A_hat = adj + torch.eye(N, device=device, dtype=x.dtype)

        # D^{-1/2} A_hat D^{-1/2}
        deg = A_hat.sum(dim=-1) + 1e-9
        D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt

        h = self.drop(F.gelu(self.lin1(A_norm @ x)))
        z = self.lin2(A_norm @ h)
        return z  # (N, out_dim)


# -----------------------------
# Dataset from cache (tensorized)
# -----------------------------

class CachedTensorDataset(torch.utils.data.Dataset):
    def __init__(self, cache: Dict, indices: np.ndarray):
        self.ids = cache["ids"][indices]
        self.T = torch.from_numpy(cache["text"][indices])
        self.A = torch.from_numpy(cache["audio"][indices])
        self.V = torch.from_numpy(cache["visual"][indices])
        self.U = torch.from_numpy(cache["temporal"][indices])
        self.AUX = torch.from_numpy(cache["aux"][indices])
        self.y = torch.from_numpy(cache["labels"][indices]).long()
        # gnn features will be injected later (same order)

    def __len__(self):
        return self.T.shape[0]

    def __getitem__(self, i):
        return {
            "text_features": self.T[i],
            "audio_features": self.A[i],
            "visual_features": self.V[i],
            "temporal_features": self.U[i],
            "aux": self.AUX[i],
            "label": self.y[i],
            "index": i  # local index within this split
        }


# -----------------------------
# Config dataclass
# -----------------------------

@dataclass
class TrainConfig:
    data_root: str
    ocr_phrase_pkl: Optional[str]
    out_dir: str = "outputs"
    batch_size: int = 16
    epochs: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    gnn_dim: int = 128
    gnn_overlap_thresh: float = 0.12
    seed: int = 42
    use_mps: bool = True
    use_gnn: bool = True
    save_best: bool = True
    # nice-to-haves
    grad_clip: float = 5.0
    early_stop_patience: int = 3  # stop if no AUC improvement this many epochs


# -----------------------------
# Utility: build adjacency from OCR phrase sets
# -----------------------------

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b)) + 1e-9
    return float(inter / union)

def build_adj_from_ocr(ocr_sets, thresh: float = 0.12) -> np.ndarray:
    N = len(ocr_sets)
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        A[i, i] = 1.0
        si = ocr_sets[i]
        for j in range(i + 1, N):
            sj = ocr_sets[j]
            jac = jaccard(si, sj)
            if jac >= thresh:
                A[i, j] = A[j, i] = 1.0
    return A


# -----------------------------
# Trainer
# -----------------------------

class ForensicTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)

        # Device
        if cfg.use_mps and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.dtype = torch.float32
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # 1) Load raw dataset + build cache
        raw = FakeSVRawDataset(cfg.data_root)
        self.cache = build_gnn_cache_from_raw_dataset(
            raw,
            ocr_phrase_pkl=cfg.ocr_phrase_pkl,
            text_dim=768, audio_dim=128, visual_dim=512, temporal_dim=256, seed=cfg.seed
        )
        (self.tr_idx, self.va_idx, self.te_idx) = self.cache["split"]

        # 2) Build graph (train+val+test combined, embeddings learned transductively)
        self._build_gnn()

        # 3) Build data loaders
        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders()

        # 4) Models
        self.fusion = CrossModalTransformer(config_path="configs/model_configs/fusion.yaml").to(self.device)
        self.clf = DeepTruthClassifier(config_path="configs/model_configs/classifier.yaml").to(self.device)

        # 5) Optimizer and Scheduler (with lr decay)
        params = list(self.fusion.parameters()) + list(self.clf.parameters()) + (
            list(self.gnn.parameters()) if self.cfg.use_gnn else []
        )
        self.optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = StepLR(self.optim, step_size=3, gamma=0.7)  # Decay LR every 3 epochs by 30%

        self.best_val_auc = -1.0
        self.no_improve = 0
        self.ckpt_path = os.path.join(cfg.out_dir, "best.pt")

    # ---------- GNN ----------
    def _build_gnn(self):
        T = self.cache["text"]     # (N, 768)
        A = self.cache["audio"]    # (N, 128)
        V = self.cache["visual"]   # (N, 512)
        U = self.cache["temporal"] # (N, 256)
        ocr_sets = self.cache["ocr_sets"]

        # Node features for the graph: a compact projection of modalities (no gradients needed here)
        with torch.no_grad():
            X = np.concatenate([T[:, :192], A[:, :32], V[:, :128], U[:, :64]], axis=1).astype(np.float32)  # (N, 416)
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

        # Adjacency by OCR Jaccard
        Adj = build_adj_from_ocr(ocr_sets, thresh=self.cfg.gnn_overlap_thresh)  # (N,N) 0/1

        self.X = torch.from_numpy(X).to(self.device, dtype=self.dtype)     # (N, 416)
        self.Adj = torch.from_numpy(Adj).to(self.device, dtype=self.dtype) # (N, N)

        # Initialize GCN
        self.gnn = SimpleGCN(in_dim=self.X.shape[1], hid=2*self.cfg.gnn_dim, out_dim=self.cfg.gnn_dim, dropout=0.2).to(self.device)

        # Optional quick pre-train for stability (degree reconstruction)
        self._pretrain_gnn(epochs=2)

        # Cache initial node embeddings (kept in cache order)
        with torch.no_grad():
            Z = self.gnn(self.X, self.Adj)  # (N, gnn_dim)
        self.cache["gnn_Z"] = Z.detach()

    def _pretrain_gnn(self, epochs: int = 2):
        if not self.cfg.use_gnn:
            return
        opt = torch.optim.Adam(self.gnn.parameters(), lr=1e-3, weight_decay=1e-4)
        target_deg = self.Adj.sum(dim=-1, keepdim=True) / max(1.0, self.Adj.shape[0])
        head = nn.Linear(self.cfg.gnn_dim, 1, device=self.device, dtype=self.dtype)
        for _ in range(epochs):
            self.gnn.train()
            Z = self.gnn(self.X, self.Adj)        # (N, g)
            pred = torch.sigmoid(head(Z))
            loss = F.mse_loss(pred, target_deg)
            opt.zero_grad(); loss.backward(); opt.step()

    # ---------- Data ----------
    def _build_dataloaders(self):
        tr = CachedTensorDataset(self.cache, self.tr_idx)
        va = CachedTensorDataset(self.cache, self.va_idx)
        te = CachedTensorDataset(self.cache, self.te_idx)

        train_loader = torch.utils.data.DataLoader(tr, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(va, batch_size=self.cfg.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(te, batch_size=self.cfg.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    # ---------- Train / Eval ----------
    def _forward_batch(self, batch, split: str) -> Dict[str, torch.Tensor]:
        # Map local split indices back to global indices:
        idx_local = batch["index"]  # indices within this split
        if split == "train":
            global_idx = torch.tensor(self.tr_idx, device=self.device)[idx_local]
        elif split == "val":
            global_idx = torch.tensor(self.va_idx, device=self.device)[idx_local]
        else:
            global_idx = torch.tensor(self.te_idx, device=self.device)[idx_local]

        # Gather GNN features for these nodes
        gnn_feat = None
        if self.cfg.use_gnn:
            Z_all = self.cache["gnn_Z"]  # (N, g)
            gnn_feat = Z_all[global_idx] # (B, g)

        feats = {
            "text_features": batch["text_features"].to(self.device, dtype=self.dtype),
            "audio_features": batch["audio_features"].to(self.device, dtype=self.dtype),
            "visual_features": batch["visual_features"].to(self.device, dtype=self.dtype),
            "temporal_features": batch["temporal_features"].to(self.device, dtype=self.dtype),
            "gnn_feat": gnn_feat
        }
        fused_out = self.fusion(feats)        # dict with "fused", "logits", "forensic"
        aux = batch["aux"].to(self.device, dtype=self.dtype)
        y = batch["label"].to(self.device)
        clf_out = self.clf(fused_out["fused"], aux)

        return {
            "logits": clf_out["logits"],        # (B,2)
            "probs": clf_out["probs"],          # (B,2)
            "y": y,                             
            "forensic": fused_out.get("forensic", {})  # emotion_intensity, semantic_conflict, temporal_delay
        }

    def _epoch_loop(self, loader, split: str) -> Tuple[float, Dict[str, float]]:
        is_train = (split == "train")
        self.fusion.train(is_train)
        self.clf.train(is_train)
        if self.cfg.use_gnn:
            self.gnn.train(is_train)

        losses: List[float] = []
        y_all: List[np.ndarray] = []
        p1_all: List[np.ndarray] = []
        forensic_buf = {"semantic_conflict": [], "temporal_delay": [], "emotion_intensity": []}

        for batch in loader:
            out = self._forward_batch(batch, split)
            loss = F.cross_entropy(out["logits"], out["y"])

            if is_train:
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        list(self.fusion.parameters()) + list(self.clf.parameters()) +
                        (list(self.gnn.parameters()) if self.cfg.use_gnn else []),
                        max_norm=self.cfg.grad_clip
                    )
                self.optim.step()

            # collect
            losses.append(float(loss.detach().cpu()))
            y_np = out["y"].detach().cpu().numpy()
            p1_np = out["probs"][:, 1].detach().cpu().numpy()
            y_all.append(y_np); p1_all.append(p1_np)

            # forensic (each present as (B,))
            f = out["forensic"]
            if "semantic_conflict" in f:
                forensic_buf["semantic_conflict"].append(f["semantic_conflict"].detach().cpu().numpy())
            if "temporal_delay" in f:
                forensic_buf["temporal_delay"].append(f["temporal_delay"].detach().cpu().numpy())
            if "emotion_intensity" in f:
                forensic_buf["emotion_intensity"].append(f["emotion_intensity"].detach().cpu().numpy())

        # aggregate
        loss_mean = float(np.mean(losses)) if losses else 0.0
        y_cat = np.concatenate(y_all) if y_all else np.array([], dtype=int)
        p1_cat = np.concatenate(p1_all) if p1_all else np.array([], dtype=float)

        forensic_cat = None
        if forensic_buf["semantic_conflict"] or forensic_buf["temporal_delay"]:
            forensic_cat = {
                k: (np.concatenate(v) if len(v) else np.array([], dtype=float))
                for k, v in forensic_buf.items()
            }

        metrics = aggregate_epoch_metrics(
            y_true=y_cat, y_score=p1_cat, forensic=forensic_cat, threshold=0.5, include_cm=False
        )
        return loss_mean, metrics

    def fit(self):
        best_score = -1.0
        self.no_improve = 0

        for epoch in range(1, self.cfg.epochs + 1):
            tr_loss, tr_metrics = self._epoch_loop(self.train_loader, "train")
            va_loss, va_metrics = self._epoch_loop(self.val_loader, "val")

            # Step scheduler after each epoch
            self.scheduler.step()

            # Logging metrics (already added)
            print(f"[Epoch {epoch:02d}] train_loss={tr_loss:.4f} | ", end="")
            pretty_print("train", tr_metrics)
            print(f"           val_loss={va_loss:.4f} | ", end="")
            pretty_print("val", va_metrics)

            # Early stopping (already in place)
            val_auc = float(va_metrics.get("auc", 0.5))
            improved = val_auc > (self.best_val_auc + 1e-4)
            if improved and self.cfg.save_best:
                self.best_val_auc = val_auc
                self.no_improve = 0
                torch.save({
                    "fusion": self.fusion.state_dict(),
                    "clf": self.clf.state_dict(),
                    "gnn": self.gnn.state_dict() if self.cfg.use_gnn else None,
                    "cfg": self.cfg.__dict__
                }, self.ckpt_path)
                print(f"  ↳ saved best checkpoint to {self.ckpt_path} (val_auc={self.best_val_auc:.3f})")
            else:
                self.no_improve += 1
                if self.no_improve >= self.cfg.early_stop_patience:
                    print(f"↳ Early stopping (no val AUC improvement for {self.cfg.early_stop_patience} epochs)")
                    break

        return self.best_val_auc

    def test(self) -> Dict[str, float]:
        # load best
        if os.path.exists(self.ckpt_path):
            ck = torch.load(self.ckpt_path, map_location=self.device)
            self.fusion.load_state_dict(ck["fusion"])
            self.clf.load_state_dict(ck["clf"])
            if self.cfg.use_gnn and ck.get("gnn") is not None:
                self.gnn.load_state_dict(ck["gnn"])

        self.fusion.eval(); self.clf.eval()
        if self.cfg.use_gnn: self.gnn.eval()

        ts_loss, ts_metrics = self._epoch_loop(self.test_loader, "test")
        print(f"[Test] loss={ts_loss:.4f} | ", end="")
        pretty_print("test", ts_metrics)

        # For convenience keep old keys too
        return {
            "test_loss": ts_loss,
            "test_acc": ts_metrics.get("accuracy", 0.0),
            "test_auc": ts_metrics.get("auc", 0.5),
            "test_precision": ts_metrics.get("precision", 0.0),
            "test_recall": ts_metrics.get("recall", 0.0),
            "test_f1": ts_metrics.get("f1", 0.0),
            "test_cmcs": ts_metrics.get("cmcs", 0.0),
            "test_dfdr": ts_metrics.get("dfdr", 0.0),
        }
