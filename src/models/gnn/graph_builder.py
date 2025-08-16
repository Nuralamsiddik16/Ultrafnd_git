# src/models/gnn/graph_builder.py
import numpy as np

def cosine_knn(X: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Build a symmetric kNN graph (cosine) with self-loops.
    X: [N, D] feature matrix
    returns A: [N, N] dense adjacency (float32)
    """
    X = X.astype(np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -1)  # exclude self when picking neighbors
    N = X.shape[0]
    edges = set()
    for i in range(N):
        nbrs = np.argpartition(-S[i], k)[:k]
        for j in nbrs:
            if i == j: 
                continue
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b))
    A = np.zeros((N, N), dtype=np.float32)
    for a, b in edges:
        A[a, b] = 1.0
        A[b, a] = 1.0
    A += np.eye(N, dtype=np.float32)  # Adding self-loops to the adjacency matrix
    return A

def add_ocr_overlap_weights(A: np.ndarray, ocr_sets, alpha: float = 0.4) -> np.ndarray:
    """
    Increase edge weights when OCR phrase sets overlap.
    ocr_sets: List[Set[str]] aligned with node order.
    """
    N = A.shape[0]
    for i in range(N):
        si = ocr_sets[i]
        for j in range(i + 1, N):
            sj = ocr_sets[j]
            ov = len(si & sj)
            if ov > 0:
                w = alpha * np.log1p(ov)
                A[i, j] += w
                A[j, i] += w
    return A

def add_temporal_inconsistency(A: np.ndarray, delay_scores: np.ndarray, beta: float = 0.25) -> np.ndarray:
    """
    Re-weight edges by temporal inconsistency differences (e.g., lip-sync lag).
    delay_scores: [N] in [0,1]
    """
    N = A.shape[0]
    for i in range(N):
        di = delay_scores[i]
        for j in range(i + 1, N):
            w = 1.0 + beta * abs(di - delay_scores[j])
            A[i, j] *= w
            A[j, i] *= w
    return A

def build_dense_adj(X, ocr_sets, delay_scores, k=8, alpha=0.4, beta=0.25) -> np.ndarray:
    """
    Convenience: kNN + OCR overlap + temporal inconsistency; returns dense A.
    """
    A = cosine_knn(X, k=k)
    A = add_ocr_overlap_weights(A, ocr_sets, alpha=alpha)
    A = add_temporal_inconsistency(A, delay_scores, beta=beta)
    return A
