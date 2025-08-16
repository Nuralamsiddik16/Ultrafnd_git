# src/models/chronos_guard.py
from typing import Optional, List, Union, Dict
import numpy as np

# Optional deps
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _as_numpy_frame(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-6:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr

def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if _HAS_CV2:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # manual
    r, g, b = img[...,0], img[...,1], img[...,2]
    return (0.2989*r + 0.587*g + 0.114*b).astype(np.uint8)

def _resize(img: np.ndarray, size=(256,256)) -> np.ndarray:
    if _HAS_CV2:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    # naive
    from math import floor
    H,W = img.shape[:2]
    out = np.zeros((size[1], size[0]) + (() if img.ndim==2 else (img.shape[2],)), dtype=img.dtype)
    for y in range(size[1]):
        for x in range(size[0]):
            out[y,x] = img[floor(y*H/size[1]), floor(x*W/size[0])]
    return out

def _hist_diff(g0: np.ndarray, g1: np.ndarray) -> float:
    """Scene cut proxy: histogram difference (L1 between normalized 32-bin histograms)."""
    h0,_ = np.histogram(g0, bins=32, range=(0,255), density=True)
    h1,_ = np.histogram(g1, bins=32, range=(0,255), density=True)
    return float(np.abs(h0 - h1).sum())

def _flow_mag(g0: np.ndarray, g1: np.ndarray) -> float:
    """Mean optical-flow magnitude (TV-L1 if available else Farneback else |Δ|)."""
    if _HAS_CV2 and hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
        try:
            flow = cv2.optflow.DualTVL1OpticalFlow_create().calc(g0, g1, None)
            fx, fy = flow[...,0], flow[...,1]
            return float(np.sqrt(fx*fx + fy*fy).mean())
        except Exception:
            pass
    if _HAS_CV2:
        try:
            flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            fx, fy = flow[...,0], flow[...,1]
            return float(np.sqrt(fx*fx + fy*fy).mean())
        except Exception:
            pass
    return float(np.abs(g1.astype(np.float32) - g0.astype(np.float32)).mean())

def _norm01(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))


class ChronosGuard:
    """
    Temporal tampering detector (cuts/pacing anomalies/motion inconsistencies).
    Exposes:
      - extract_features(frames) -> np.ndarray[feat_dim]  (default 128)
      - temporal_tamper_score(frames, audio=None) -> float ∈ [0,1]
      - estimate_av_lag(audio_env, mouth_open, sr, fps) -> seconds (optional, for reporting)
    """

    def __init__(self, feat_dim: int = 128):
        self.feat_dim = int(feat_dim)

    def extract_features(self, frames_or_tensor: Union[List, "np.ndarray", "torch.Tensor"]) -> np.ndarray:
        frames = []
        if isinstance(frames_or_tensor, (list, tuple)):
            frames = [ _as_numpy_frame(f) for f in frames_or_tensor if _as_numpy_frame(f) is not None ]
        elif _HAS_TORCH and isinstance(frames_or_tensor, torch.Tensor) and frames_or_tensor.ndim == 4:
            frames = [ _as_numpy_frame(frames_or_tensor[i]) for i in range(frames_or_tensor.shape[0]) ]
        elif isinstance(frames_or_tensor, np.ndarray) and frames_or_tensor.ndim == 4:
            frames = [ _as_numpy_frame(frames_or_tensor[i]) for i in range(frames_or_tensor.shape[0]) ]

        if len(frames) < 2:
            return np.zeros(self.feat_dim, dtype=np.float32)

        # compute per-step cues
        cut_scores, flows = [], []
        for i in range(len(frames)-1):
            f0 = _resize(frames[i], (256,256))
            f1 = _resize(frames[i+1], (256,256))
            g0, g1 = _gray(f0), _gray(f1)
            cut_scores.append(_hist_diff(g0, g1))
            flows.append(_flow_mag(g0, g1))

        cut_scores = np.asarray(cut_scores, dtype=np.float32)  # [T-1]
        flows = np.asarray(flows, dtype=np.float32)

        # Aggregate stats
        feat = [
            cut_scores.mean(), cut_scores.std(), cut_scores.max(),
            flows.mean(), flows.std(), flows.max(),
            float(np.corrcoef(cut_scores, flows)[0,1]) if cut_scores.size>3 else 0.0,
        ]
        v = np.asarray(feat, dtype=np.float32)

        # expand/project to fixed dim
        if v.shape[0] < self.feat_dim:
            reps = int(np.ceil(self.feat_dim / v.shape[0]))
            v = np.tile(v, reps)[: self.feat_dim]
        else:
            v = v[: self.feat_dim]
        n = np.linalg.norm(v) + 1e-9
        return (v / n).astype(np.float32)

    def temporal_tamper_score(self, frames_or_tensor: Union[List, "np.ndarray", "torch.Tensor"], audio: Optional[np.ndarray] = None) -> float:
        """
        Heuristic score in [0,1].
        High when many hard cuts with inconsistent motion, or very low motion yet frequent cuts.
        """
        frames = []
        if isinstance(frames_or_tensor, (list, tuple)):
            frames = [ _as_numpy_frame(f) for f in frames_or_tensor if _as_numpy_frame(f) is not None ]
        elif _HAS_TORCH and isinstance(frames_or_tensor, torch.Tensor) and frames_or_tensor.ndim == 4:
            frames = [ _as_numpy_frame(frames_or_tensor[i]) for i in range(frames_or_tensor.shape[0]) ]
        elif isinstance(frames_or_tensor, np.ndarray) and frames_or_tensor.ndim == 4:
            frames = [ _as_numpy_frame(frames_or_tensor[i]) for i in range(frames_or_tensor.shape[0]) ]

        if len(frames) < 2:
            return 0.0

        cut_scores, flows = [], []
        for i in range(len(frames)-1):
            f0 = _resize(frames[i], (256,256))
            f1 = _resize(frames[i+1], (256,256))
            g0, g1 = _gray(f0), _gray(f1)
            c = _hist_diff(g0, g1)
            m = _flow_mag(g0, g1)
            cut_scores.append(c); flows.append(m)

        cut_scores = np.asarray(cut_scores, dtype=np.float32)
        flows = np.asarray(flows, dtype=np.float32)

        # rule: many cuts + low/irregular motion → suspicious
        c_mean = float(cut_scores.mean())
        f_mean = float(flows.mean())
        f_std = float(flows.std())

        score = 0.6 * _norm01(c_mean, 0.05, 0.5) + 0.4 * _norm01(abs(f_std - f_mean), 0.0, 0.5)
        return float(np.clip(score, 0.0, 1.0))

    # Optional: cross-correlation based A/V lag (seconds)
    @staticmethod
    def estimate_av_lag(audio_env: np.ndarray, mouth_open: np.ndarray, sr: float = 16000.0, fps: float = 25.0, max_lag_s: float = 0.5) -> float:
        L = min(len(audio_env), len(mouth_open))
        if L < 4:
            return 0.0
        a = (audio_env[:L] - np.mean(audio_env[:L])) / (np.std(audio_env[:L]) + 1e-9)
        m = (mouth_open[:L] - np.mean(mouth_open[:L])) / (np.std(mouth_open[:L]) + 1e-9)
        n = 1
        while n < 2 * L:
            n <<= 1
        A = np.fft.rfft(a, n)
        M = np.fft.rfft(m, n)
        xc = np.fft.irfft(A * np.conj(M), n)
        xc = np.concatenate([xc[-(L-1):], xc[:L]])
        max_lag = int(max_lag_s * sr)
        center = len(xc) // 2
        lo = max(0, center - max_lag)
        hi = min(len(xc), center + max_lag + 1)
        window = xc[lo:hi]
        lag_idx = np.argmax(window)
        lag_samples = (lo + lag_idx) - center
        return float(lag_samples / sr)
