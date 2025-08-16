# src/models/gnn/numpy_fallback.py
import numpy as np

def normalize_adj(A: np.ndarray) -> np.ndarray:
    """
    Normalize adjacency matrix A to make it stable for GNN calculations.
    A: (N, N) adjacency matrix (sparse or dense)
    Returns normalized adjacency matrix.
    """
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(D) + 1e-9))
    return D_inv_sqrt @ A @ D_inv_sqrt

class GNNSimple:
    """
    Tiny NumPy GCN: good for quick tests anywhere (no heavy deps).
    """
    def __init__(self, in_dim, hid=256, layers=2, num_classes=2, seed=42):
        rng = np.random.default_rng(seed)
        self.W = []
        dims = [in_dim] + [hid] * (layers - 1) + [num_classes]
        for a, b in zip(dims[:-1], dims[1:]):
            self.W.append((rng.standard_normal((a, b)).astype(np.float32) * 0.05))
        self.cache = {}

    def forward(self, X, A_norm, training=False):
        """
        Forward pass through the GCN layers.
        X: Node features (N, F)
        A_norm: Normalized adjacency matrix (N, N)
        training: Whether the model is in training mode for dropout.
        """
        H = X
        self.cache["H"] = [H]
        for W in self.W[:-1]:
            H = A_norm @ H @ W
            H = np.maximum(0, H)  # ReLU activation
            self.cache["H"].append(H)
        Z = A_norm @ H @ self.W[-1]
        ex = np.exp(Z - Z.max(axis=1, keepdims=True))
        P = ex / (ex.sum(axis=1, keepdims=True) + 1e-9)
        self.cache["P"] = P
        return P

    def backward(self, A_norm, y_true, mask, lr=0.02, wd=1e-4):
        """
        Backpropagation: Gradient update for the GCN model.
        A_norm: Normalized adjacency matrix (N, N)
        y_true: Ground truth labels (N,)
        mask: Mask for valid data
        lr: Learning rate
        wd: Weight decay (L2 regularization)
        """
        P = self.cache["P"]
        N, C = P.shape
        Y = np.zeros_like(P)
        Y[np.arange(N), y_true] = 1.0
        G = (P - Y) / (mask.sum() + 1e-9)
        G *= mask[:, None].astype(np.float32)

        grads = [None] * len(self.W)
        Hs = self.cache["H"]
        H_last = Hs[-1]

        grads[-1] = (A_norm @ H_last).T @ G + wd * self.W[-1]
        Ghid = (G @ self.W[-1].T) * (H_last > 0).astype(np.float32)

        for li in reversed(range(len(self.W) - 1)):
            H_prev = Hs[li]
            grads[li] = (A_norm @ H_prev).T @ Ghid + wd * self.W[li]
            if li > 0:
                Ghid = (Ghid @ self.W[li].T) * (Hs[li] > 0).astype(np.float32)

        for i in range(len(self.W)):
            self.W[i] -= lr * grads[i]

    def predict(self, X, A_norm):
        """
        Predict the class for each node.
        X: Node features (N, F)
        A_norm: Normalized adjacency matrix (N, N)
        Returns predicted classes.
        """
        return np.argmax(self.forward(X, A_norm, training=False), axis=1)
