from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


class MetricsCalculator:
    """Accumulates predictions and computes common classification metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        self.all_preds.extend(preds.detach().cpu().numpy())
        self.all_labels.extend(labels.detach().cpu().numpy())
        self.all_probs.extend(probs.detach().cpu().numpy())

    def compute(self) -> Dict[str, float]:
        labels = np.array(self.all_labels)
        preds = np.array(self.all_preds)
        probs = np.array(self.all_probs)
        metrics = {
            "accuracy": float(np.mean(labels == preds)),
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall": recall_score(labels, preds, average="macro", zero_division=0),
            "f1_score": f1_score(labels, preds, average="macro", zero_division=0),
        }
        # Binary classification AUC requires probabilities of positive class
        if probs.ndim == 2 and probs.shape[1] == 2:
            metrics["auc_roc"] = roc_auc_score(labels, probs[:, 1])
        return metrics
