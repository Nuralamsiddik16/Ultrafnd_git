# src/training/forensic_trainer.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score

from src.data_pipeline.fakesv_dataset import FakeSVRawDataset, build_gnn_cache_from_raw_dataset
from src.models.fusion.cross_modal_transformer import CrossModalTransformer
from src.models.fusion.deep_truth_classifier import DeepTruthClassifier
from src.models.gnn import GNNModel


# ----------------------- Config -----------------------

@dataclass
class TrainConfig:
    data_root: str
    ocr_phrase_pkl: Optional[str]
    out_dir: str = "outputs_v2"

    epochs: int = 12
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-4
    seed: int = 42

    # GNN
    use_gnn: bool = True
    gnn_dim: int = 128
    gnn_overlap_thresh: float = 0.12

    # Device
    use_mps: bool = False

    # Checkpointing
    save_best: bool = True

    # Loss options
    label_smoothing: float = 0.05
    class_weighting: bool = False  # turn on if your dataset is imbalanced

    # Encoder freeze (epochs)
    freeze_epochs: int = 0  # set to >0 if you want to freeze encoders early

    # Grad clipping
    grad_clip: float = 1.0

    # Scheduler
    use_cosine: bool = True
    min_lr_scale: float = 0.1  # eta_min = lr * min_lr_scale


# ----------------------- Utilities -----------------------

def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def auc_safe(y_true: np.ndarray, y_prob_pos: np.ndarray) -> float:
    """Return ROC-AUC if both classes present; else return 0.5 (neutral)."""
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_prob_pos))
    except Exception:
        return 0.5


def build_adj_from_ocr_sets(ocr_sets: List[set], overlap_thresh: float = 0.12) -> torch.Tensor:
    """
    Build an undirected weighted adjacency from OCR token overlap Jaccard.
    """
    N = len(ocr_sets)
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        si = ocr_sets[i]
        if not si:
            continue
        for j in range(i + 1, N):
            sj = ocr_sets[j]
            if not sj:
                continue
            inter = len(si & sj)
            union = len(si | sj)
            s = (inter / union) if union > 0 else 0.0
            if s >= overlap_thresh:
                A[i, j] = s
                A[j, i] = s
    # add small self-loops via the GNN’s normalization if needed
    return torch.from_numpy(A)


# ----------------------- Trainer -----------------------

class ForensicTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        _set_seed(cfg.seed)

        # Device
        if cfg.use_mps and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Data / cache
        raw = FakeSVRawDataset(cfg.data_root)
        self.cache = build_gnn_cache_from_raw_dataset(
            raw,
            ocr_phrase_pkl=cfg.ocr_phrase_pkl,
            text_dim=768, audio_dim=128, visual_dim=512, temporal_dim=256, seed=cfg.seed
        )
        (self.train_idx, self.val_idx, self.test_idx) = self.cache["split"]

        # Print distributions once (sanity)
        def dist(idx):
            y = self.cache["labels"][idx]
            return {int(c): int((y == c).sum()) for c in np.unique(y)}
        print(f"[Split] sizes train/val/test = {len(self.train_idx)}/{len(self.val_idx)}/{len(self.test_idx)}")
        print(f"[Split] label dist train: {dist(self.train_idx)} | val: {dist(self.val_idx)} | test: {dist(self.test_idx)}")

        # Models
        # GNN for sample graph features
        self.use_gnn = cfg.use_gnn
        if self.use_gnn:
            # Node feature is a compact concat: [mean(text,audio,visual,temporal), aux(2)]
            self.gnn_in_dim = 416  #  ( (768+128+512+256)/4 = 416 )  + 2 aux is folded later
            self.gnn = GNNModel(in_dim=self.gnn_in_dim, hid=256, out_dim=cfg.gnn_dim, dropout=0.1).to(self.device)
        else:
            self.gnn = None

        # Fusion and classifier
        self.fusion = CrossModalTransformer("configs/model_configs/fusion.yaml").to(self.device)
        self.classifier = DeepTruthClassifier("configs/model_configs/classifier.yaml").to(self.device)

        # Params & optimizer
        params = []
        if self.gnn is not None:
            params += list(self.gnn.parameters())
        params += list(self.fusion.parameters()) + list(self.classifier.parameters())

        self.optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        if cfg.use_cosine:
            eta_min = cfg.lr * cfg.min_lr_scale
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epochs, eta_min=eta_min)
        else:
            self.scheduler = None

        # Loss
        if cfg.class_weighting:
            y = self.cache["labels"]
            pos = float((y == 1).sum())
            neg = float((y == 0).sum())
            total = max(1.0, pos + neg)
            w0 = 0.5 * total / max(1.0, neg)
            w1 = 0.5 * total / max(1.0, pos)
            weight = torch.tensor([w0, w1], device=self.device, dtype=torch.float32)
        else:
            weight = None
        self.criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=self.cfg.label_smoothing)

        # Early stopping state
        self.best_score = -1.0
        self.no_improve = 0

        # Optional: freeze encoders early (if your fusion exposes them)
        self._frozen = False
        if self.cfg.freeze_epochs > 0:
            self._freeze_encoders(True)

        os.makedirs(self.cfg.out_dir, exist_ok=True)

    # --------- Optional encoder (un)freeze stubs, safe if modules don’t exist ----------
    def _set_requires_grad(self, module: Optional[nn.Module], flag: bool):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = flag

    def _freeze_encoders(self, flag: bool):
        # Adjust attribute names if your fusion exposes encoders directly.
        self._set_requires_grad(getattr(self.fusion, "text_encoder", None), not flag)
        self._set_requires_grad(getattr(self.fusion, "audio_encoder", None), not flag)
        self._set_requires_grad(getattr(self.fusion, "visual_backbone", None), not flag)
        self._frozen = flag

    # -----------------------------------------------------------------------------------

    def _iter_batches(self, idx: np.ndarray, batch_size: int):
        N = len(idx)
        for i in range(0, N, batch_size):
            yield idx[i:i + batch_size]

    def _pack_batch(self, sel: np.ndarray) -> Dict[str, torch.Tensor]:
        T = torch.from_numpy(self.cache["text"][sel]).to(self.device)
        A = torch.from_numpy(self.cache["audio"][sel]).to(self.device)
        V = torch.from_numpy(self.cache["visual"][sel]).to(self.device)
        U = torch.from_numpy(self.cache["temporal"][sel]).to(self.device)
        AUX = torch.from_numpy(self.cache["aux"][sel]).to(self.device)
        y = torch.from_numpy(self.cache["labels"][sel]).long().to(self.device)
        batch = dict(text=T, audio=A, visual=V, temporal=U, aux=AUX, y=y)

        if self.use_gnn:
            # Node features: mean over modalities, then concat aux
            Xmean = torch.stack([T, A, V, U], dim=0).mean(dim=0)  # (B, 416)
            gnn_X = Xmean
            # Build adjacency from OCR overlap for THIS mini-batch
            ocr_sub = [self.cache["ocr_sets"][int(i)] for i in sel]
            # anneal threshold per epoch if set through self._curr_epoch
            thr0 = self.cfg.gnn_overlap_thresh
            epoch = getattr(self, "_curr_epoch", 0)
            thr = max(0.05, thr0 * (0.95 ** epoch))
            A_mat = build_adj_from_ocr_sets(ocr_sub, overlap_thresh=thr).to(self.device)
            batch.update(gnn_X=gnn_X, gnn_A=A_mat)
        return batch

    def _forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # GNN forward (if used)
        if self.use_gnn and self.gnn is not None:
            gnn_feat = self.gnn(batch["gnn_X"], batch["gnn_A"])  # (B, gnn_dim)
        else:
            B = batch["text"].size(0)
            gnn_feat = torch.zeros(B, self.cfg.gnn_dim, device=self.device, dtype=batch["text"].dtype)

        fused = self.fusion({
            "text_features": batch["text"],
            "audio_features": batch["audio"],
            "visual_features": batch["visual"],
            "temporal_features": batch["temporal"],
            "gnn_feat": gnn_feat
        })
        # Fused representation and classifier
        logits, probs = self.classifier(fused["fused"], batch["aux"])["logits"], self.classifier(fused["fused"], batch["aux"])["probs"]
        return {"logits": logits, "probs": probs}

    def _step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self._forward(batch)
        loss = self.criterion(out["logits"], batch["y"])
        with torch.no_grad():
            preds = out["probs"].argmax(dim=1).detach().cpu().numpy()
            y_true = batch["y"].detach().cpu().numpy()
            prob_pos = out["probs"][:, 1].detach().cpu().numpy()
            acc = float(accuracy_score(y_true, preds))
            auc = float(auc_safe(y_true, prob_pos))
        return loss, {"acc": acc, "auc": auc}

    def train(self) -> Dict[str, float]:
        print("\n>>> Training...")
        best_path = os.path.join(self.cfg.out_dir, "best.pt")
        for epoch in range(1, self.cfg.epochs + 1):
            self._curr_epoch = epoch - 1  # zero-based for annealing
            if self.cfg.freeze_epochs > 0 and self._frozen and epoch > self.cfg.freeze_epochs:
                print("→ Unfreezing encoders")
                self._freeze_encoders(False)

            # -- train --
            self.gnn.train() if self.gnn is not None else None
            self.fusion.train()
            self.classifier.train()

            # shuffle train indices
            tr = self.train_idx.copy()
            np.random.shuffle(tr)

            losses, accs, aucs = [], [], []
            for sel in self._iter_batches(tr, self.cfg.batch_size):
                batch = self._pack_batch(sel)
                self.optimizer.zero_grad(set_to_none=True)
                loss, metrics = self._step(batch)
                loss.backward()
                if self.cfg.grad_clip and self.cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(list(self.fusion.parameters()) + list(self.classifier.parameters()) +
                                             (list(self.gnn.parameters()) if self.gnn is not None else []),
                                             max_norm=self.cfg.grad_clip)
                self.optimizer.step()

                losses.append(float(loss.detach().cpu()))
                accs.append(metrics["acc"]); aucs.append(metrics["auc"])

            tr_loss = float(np.mean(losses)) if losses else 0.0
            tr_acc = float(np.mean(accs)) if accs else 0.0
            tr_auc = float(np.mean(aucs)) if aucs else 0.5

            # -- val --
            val_loss, val_acc, val_auc = self._evaluate(self.val_idx)

            print(f"[Epoch {epoch:02d}] train: loss={tr_loss:.4f} acc={tr_acc:.3f} auc={tr_auc:.3f} | "
                  f"val: loss={val_loss:.4f} acc={val_acc:.3f} auc={val_auc:.3f}")

            # early stopping on val AUC
            improved = val_auc > (self.best_score + 1e-4)
            if improved and self.cfg.save_best:
                self.best_score = val_auc
                self.no_improve = 0
                torch.save({
                    "fusion": self.fusion.state_dict(),
                    "classifier": self.classifier.state_dict(),
                    "gnn": (self.gnn.state_dict() if self.gnn is not None else None),
                    "cfg": self.cfg.__dict__,
                }, best_path)
                print(f"  ↳ saved best checkpoint to {best_path} (score={self.best_score:.3f})")
            else:
                self.no_improve += 1
                if self.no_improve >= 3:
                    print("↳ Early stopping (no val AUC improvement 3 epochs)")
                    break

            if self.scheduler is not None:
                self.scheduler.step()

        # test best
        return self.test()

    @torch.no_grad()
    def _evaluate(self, idx: np.ndarray) -> Tuple[float, float, float]:
        self.gnn.eval() if self.gnn is not None else None
        self.fusion.eval(); self.classifier.eval()

        losses, y_true_all, prob_pos_all, preds_all = [], [], [], []
        for sel in self._iter_batches(idx, self.cfg.batch_size):
            batch = self._pack_batch(sel)
            out = self._forward(batch)
            loss = self.criterion(out["logits"], batch["y"])
            losses.append(float(loss.detach().cpu()))
            y = batch["y"].detach().cpu().numpy()
            p1 = out["probs"][:, 1].detach().cpu().numpy()
            pred = out["probs"].argmax(dim=1).detach().cpu().numpy()

            y_true_all.append(y); prob_pos_all.append(p1); preds_all.append(pred)

        if not losses:
            return 0.0, 0.0, 0.5

        y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=int)
        prob_pos = np.concatenate(prob_pos_all) if prob_pos_all else np.array([], dtype=float)
        preds = np.concatenate(preds_all) if preds_all else np.array([], dtype=int)

        loss = float(np.mean(losses))
        acc = float(accuracy_score(y_true, preds)) if y_true.size else 0.0
        auc = float(auc_safe(y_true, prob_pos)) if y_true.size else 0.5
        return loss, acc, auc

    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        print("\n>>> Testing best checkpoint...")
        best_path = os.path.join(self.cfg.out_dir, "best.pt")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device)
            self.fusion.load_state_dict(ckpt["fusion"])
            self.classifier.load_state_dict(ckpt["classifier"])
            if self.gnn is not None and ckpt.get("gnn") is not None:
                self.gnn.load_state_dict(ckpt["gnn"])

        loss, acc, auc = self._evaluate(self.test_idx)
        print(f"[Test] loss={loss:.4f} acc={acc:.3f} auc={auc:.3f}\n")
        return {"test_loss": loss, "test_acc": acc, "test_auc": auc}
