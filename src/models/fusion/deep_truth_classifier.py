# src/models/fusion/deep_truth_classifier.py
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config_utils import ConfigManager


# -----------------------------
# Small utilities
# -----------------------------

def _device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _init_lin(m: nn.Linear):
    nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)


# -----------------------------
# NODE-lite (Oblivious Trees)
# -----------------------------

class _ObliviousTree(nn.Module):
    """
    A single oblivious tree:
      - depth D
      - at each depth, a feature gate (softmax over input dims) selects a (soft) feature,
        a threshold is learned, and a soft left/right routing is computed with temperature tau.
      - outputs class logits via a leaf table (2^D leaves, C classes).
    """
    def __init__(self, in_dim: int, num_classes: int = 2, depth: int = 4, tau: float = 10.0, dropout: float = 0.3):
        super().__init__()
        self.in_dim = in_dim
        self.depth = depth
        self.num_classes = num_classes
        self.tau = nn.Parameter(torch.tensor(float(tau)), requires_grad=False)

        # feature gates and thresholds per depth
        self.gates = nn.ParameterList([nn.Parameter(torch.zeros(in_dim)) for _ in range(depth)])  # softmax over features
        self.thresh = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(depth)])      # scalar threshold per depth

        # leaf logits: [2^depth, C]
        self.num_leaves = 1 << depth
        self.leaf_logits = nn.Parameter(torch.zeros(self.num_leaves, num_classes))
        nn.init.zeros_(self.leaf_logits)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, F)
        returns: [B, num_classes]
        """
        B, _ = x.shape

        # decisions per depth
        probs = x.new_ones((B, 1))
        for k in range(self.depth):
            alpha = torch.softmax(self.gates[k], dim=0)        # [in_dim]
            feat = (x * alpha).sum(dim=-1, keepdim=True)       # [B,1] soft-chosen feature
            t = self.thresh[k]                                 # scalar
            s = torch.sigmoid(self.tau * (feat - t))           # [B,1] right prob
            left = (1.0 - s)
            right = s
            probs = torch.cat([probs * left, probs * right], dim=1)  # double leaves

        # probs: [B, 2^depth]
        logits = probs @ self.leaf_logits                      # [B, C]
        return self.dropout(logits)  # (B, C)


class NODEEnsemble(nn.Module):
    """
    Ensemble of oblivious trees; average logits from each tree.
    """
    def __init__(self, in_dim: int, num_classes: int = 2, num_trees: int = 6, depth: int = 4, tau: float = 10.0, dropout: float = 0.3):
        super().__init__()
        self.trees = nn.ModuleList([
            _ObliviousTree(in_dim, num_classes, depth=depth, tau=tau, dropout=dropout)
            for _ in range(num_trees)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [t(x) for t in self.trees]    # list of [B,C]
        return torch.stack(outs, dim=0).mean(dim=0)  # [B,C]


# -----------------------------
# DeepTruthClassifier
# -----------------------------

class DeepTruthClassifier(nn.Module):
    """
    Final binary classifier with interpretability.
    Inputs:
      - fused: (B, F) from CrossModalTransformer
      - aux (optional): (B, A) e.g., [delay, emotion]
    """
    def __init__(self, config_path: str = "configs/model_configs/classifier.yaml"):
        super().__init__()
        cfg = ConfigManager().load_config(config_path)
        self.hidden = int(cfg.get("hidden_dim", 512))
        self.dropout = float(cfg.get("dropout", 0.3))  # Increased dropout for better regularization
        self.num_classes = int(cfg.get("num_classes", 2))
        self.use_aux = bool(cfg.get("use_aux", True))
        self.aux_dim = int(cfg.get("aux_dim", 2))  # delay, emotion by default
        self.node_trees = int(cfg.get("node_trees", 6))
        self.node_depth = int(cfg.get("node_depth", 4))
        self.node_tau = float(cfg.get("node_tau", 10.0))
        self.temperature = nn.Parameter(torch.tensor(float(cfg.get("temperature", 1.0))), requires_grad=True)

        in_dim = int(cfg.get("input_dim", self.hidden))  # must match the output of fusion head
        eff_in = in_dim + (self.aux_dim if self.use_aux else 0)

        # Pre-NODE feature conditioner (light MLP)
        self.pre = nn.Sequential(
            nn.Linear(eff_in, self.hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        for m in self.pre:
            if isinstance(m, nn.Linear): _init_lin(m)

        # NODE ensemble head
        self.node = NODEEnsemble(in_dim=self.hidden, num_classes=self.num_classes,
                                 num_trees=self.node_trees, depth=self.node_depth, tau=self.node_tau, dropout=0.3)

        # Optional linear bypass (residual logit)
        self.bypass = nn.Linear(self.hidden, self.num_classes)
        _init_lin(self.bypass)

        self.to(_device())

    def _concat_inputs(self, fused: torch.Tensor, aux: Optional[torch.Tensor]) -> torch.Tensor:
        x = fused
        if self.use_aux and (aux is not None):
            x = torch.cat([x, aux], dim=-1)
        return x

    def forward(self, fused: torch.Tensor, aux: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Returns:
          {
            "logits":      (B,2),
            "probs":       (B,2),
            "temperature": scalar
          }
        """
        dev = _device()
        fused = fused.to(dev, dtype=torch.float32)
        aux = aux.to(dev, dtype=torch.float32) if (aux is not None) else None

        x = self._concat_inputs(fused, aux)  # (B, F [+A])
        h = self.pre(x)                      # (B, H)

        logits_node = self.node(h)           # (B, 2)
        logits_bypass = self.bypass(h)       # (B, 2)
        logits = logits_node + logits_bypass

        # temperature scaling (learnable)
        t = torch.clamp(self.temperature, min=0.5, max=5.0)
        probs = F.softmax(logits / t, dim=-1)
        return {"logits": logits, "probs": probs, "temperature": t}

    # -----------------------------
    # Inference helpers
    # -----------------------------
    @torch.no_grad()
    def predict_proba(self, fused: torch.Tensor, aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.forward(fused, aux)
        return out["probs"]

    @torch.no_grad()
    def predict(self, fused: torch.Tensor, aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = self.predict_proba(fused, aux)
        return probs.argmax(dim=-1)

    # -----------------------------
    # Interpretability
    # -----------------------------
    def feature_importance(self, fused: torch.Tensor, aux: Optional[torch.Tensor] = None, class_idx: int = 1,
                           aggregate: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Gradient×Input importance.
        Returns:
          per_input_importance: (B, F[+A])
          agg_mean (optional):  (F[+A],) if aggregate=True
        """
        dev = _device()
        fused = fused.to(dev, dtype=torch.float32).requires_grad_(True)
        aux = aux.to(dev, dtype=torch.float32).requires_grad_(True) if (aux is not None) else None

        x = self._concat_inputs(fused, aux)
        h = self.pre(x)
        logits = self.node(h) + self.bypass(h)
        target = logits[:, class_idx].sum()
        target.backward()

        grad = x.grad.detach()  # (B, F[+A])
        imp = (grad * x).abs()  # Grad×Input
        if aggregate:
            return imp, imp.mean(dim=0)
        return imp, None

    def explain_shap(self, fused: torch.Tensor, aux: Optional[torch.Tensor] = None, max_samples: int = 256):
        """
        Uses SHAP if available; otherwise falls back to a smoothed gradient explanation.
        Returns dict:
          {
            "method": "shap" | "smooth-grad",
            "values": np.ndarray[B, F[+A]]
          }
        """
        try:
            import shap  # type: ignore
            self.eval()
            dev = _device()

            # Wrap a callable that outputs probabilities for class 1
            def f(X):
                X = torch.from_numpy(X).to(dev, dtype=torch.float32)
                Fdim = fused.shape[-1]
                if self.use_aux:
                    X_f = X[:, :Fdim]
                    X_a = X[:, Fdim:]
                else:
                    X_f, X_a = X, None
                with torch.no_grad():
                    out = self.forward(X_f, X_a)["probs"][:, 1]
                return out.detach().cpu().numpy()

            X = fused.detach().cpu()
            if self.use_aux and aux is not None:
                X = torch.cat([X, aux.detach().cpu()], dim=-1)
            X_shap = X[:max_samples].numpy()

            explainer = shap.KernelExplainer(f, X_shap[:32])  # tiny background for speed
            vals = explainer.shap_values(X_shap, nsamples="auto")
            vals = vals[1] if isinstance(vals, list) else vals  # class-1 at index 1
            return {"method": "shap", "values": vals}

        except Exception:
            # SmoothGrad fallback
            self.eval()
            dev = _device()
            X = fused.detach().cpu()
            if self.use_aux and aux is not None:
                X = torch.cat([X, aux.detach().cpu()], dim=-1)

            X = X[:max_samples].to(dev, dtype=torch.float32)
            X.requires_grad_(True)
            total = torch.zeros_like(X)
            N = 16
            sigma = 0.1 * X.std(dim=0, keepdim=True).clamp_min(1e-6)
            for _ in range(N):
                noise = torch.randn_like(X) * sigma
                out = self.forward(X[:, :fused.shape[-1]],
                                   X[:, fused.shape[-1]:] if self.use_aux else None)["probs"][:, 1]
                out.sum().backward(retain_graph=True)
                total += X.grad.detach().abs()
                X.grad.zero_()
                X = (X + noise).detach().requires_grad_(True)
            vals = (total / N).detach().cpu().numpy()
            return {"method": "smooth-grad", "values": vals}
