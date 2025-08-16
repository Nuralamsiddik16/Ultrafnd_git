# src/core_blocks/visual_blocks.py
from typing import List, Union, Optional
import numpy as np
import io

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

try:
    from skimage.feature import local_binary_pattern
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


# =========================
# Utilities
# =========================

def _as_numpy_frame(x) -> Optional[np.ndarray]:
    """
    Convert a single frame to numpy uint8 RGB [H,W,3] in [0,255].
    Accepts: np.ndarray (BGR/RGB/float), torch.Tensor (HWC/CHW), gracefully handles None.
    """
    if x is None:
        return None

    # Torch -> NumPy
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    # If CHW, make HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and (arr.shape[2] != 3):
        # assume CHW
        arr = np.transpose(arr, (1, 2, 0))

    # Normalize floats to 0..255
    if arr.dtype != np.uint8:
        # try to interpret as 0..1
        if arr.max() <= 1.0 + 1e-6:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    # If BGR (common from cv2), convert to RGB
    if _HAS_CV2 and arr.ndim == 3 and arr.shape[2] == 3:
        # Heuristic: assume input already RGB unless explicitly flagged; we’ll leave as-is to avoid double-conversion.
        # If you *know* your frames are BGR, uncomment:
        # arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        pass

    return arr


def _ensure_gray(img_rgb: np.ndarray) -> np.ndarray:
    if img_rgb.ndim == 2:
        return img_rgb
    if _HAS_CV2:
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # manual rgb->gray
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
    return gray.astype(np.uint8)


def _resize(img: np.ndarray, size=(256, 256)) -> np.ndarray:
    if _HAS_CV2:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    # naive nearest neighbor
    from math import floor
    H, W = img.shape[:2]
    out = np.zeros((size[1], size[0]) + (() if img.ndim == 2 else (img.shape[2],)), dtype=img.dtype)
    for y in range(size[1]):
        for x in range(size[0]):
            out[y, x] = img[floor(y * H / size[1]), floor(x * W / size[0])]
    return out


def _frames_from_input(frames_or_text) -> Optional[List[np.ndarray]]:
    """
    Normalize input to a list of RGB uint8 frames.
    Accepts: list/tuple of frames, torch tensor [T,H,W,C], numpy [T,H,W,C], or a text (returns None).
    """
    if isinstance(frames_or_text, (list, tuple)):
        frames = []
        for f in frames_or_text:
            fr = _as_numpy_frame(f)
            if fr is not None:
                frames.append(fr)
        return frames if frames else None

    arr = None
    if _HAS_TORCH and isinstance(frames_or_text, torch.Tensor):
        arr = frames_or_text.detach().cpu().numpy()
    elif isinstance(frames_or_text, np.ndarray):
        arr = frames_or_text

    if arr is not None and arr.ndim == 4 and arr.shape[-1] == 3:
        return [ _as_numpy_frame(arr[i]) for i in range(arr.shape[0]) ]

    # Not frames → probably text
    return None


def _hash_vec(text: str, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for tok in (text or "").split()[: dim]:
        v[hash(tok) % dim] += 1.0
    n = np.linalg.norm(v) + 1e-9
    return v / n


# =========================
# Optical Flow 3D CNN (pooled)
# =========================

class OpticalFlow3DCNN:
    """
    Compute motion features from consecutive frames and pool them to a fixed vector.
    Priorities:
      1) TV-L1 optical flow (best for tamper/low-texture motion)
      2) Farneback fallback
      3) Simple frame-diff fallback

    Output: fixed-size vector (dim, default 256)
    """

    def __init__(self, dim: int = 256, n_pyramid_levels: int = 3, use_tvl1: bool = True):
        self.dim = int(dim)
        self.n_pyr = int(n_pyramid_levels)
        self.use_tvl1 = bool(use_tvl1 and _HAS_CV2 and hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"))

        # Prepare flow estimator(s)
        self._tvl1 = None
        if self.use_tvl1:
            try:
                self._tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
            except Exception:
                self.use_tvl1 = False

    def _flow_pair(self, g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
        if self.use_tvl1 and self._tvl1 is not None:
            flow = self._tvl1.calc(g0, g1, None)  # [H,W,2], float32
            return flow

        # Farneback fallback
        if _HAS_CV2:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    g0, g1, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                return flow
            except Exception:
                pass

        # Simple fallback: pseudo-flow from difference (dx=0, dy=0; magnitude=|Δ|)
        flow = np.zeros((*g0.shape, 2), dtype=np.float32)
        flow[..., 0] = 0.0
        flow[..., 1] = (g1.astype(np.float32) - g0.astype(np.float32))
        return flow

    def _pool_flow(self, flows: List[np.ndarray]) -> np.ndarray:
        """
        Compute histograms of magnitude & orientation with temporal pyramid,
        then flatten and project to fixed dim.
        """
        mags = []
        angs = []
        for F in flows:
            fx, fy = F[..., 0], F[..., 1]
            mag = np.sqrt(fx * fx + fy * fy)
            ang = (np.arctan2(fy, fx) + np.pi) / (2 * np.pi)  # [0,1]
            mags.append(mag)
            angs.append(ang)

        if not mags:
            # empty → zero vec
            return np.zeros(self.dim, dtype=np.float32)

        mags = np.stack(mags, axis=0)  # [T-1, H, W]
        angs = np.stack(angs, axis=0)  # [T-1, H, W]

        # Temporal pyramid: split along time into 1, 2, 4 chunks (sum=7)
        def temporal_chunks(arr, levels=3):
            T = arr.shape[0]
            chunks = []
            n_parts_total = sum(2 ** i for i in range(levels))
            start = 0
            for i in range(levels):
                parts = 2 ** i
                seg_len = max(1, T // parts)
                s = 0
                for p in range(parts):
                    a = p * seg_len
                    b = (p + 1) * seg_len if (p < parts - 1) else T
                    chunks.append(arr[a:b])
            return chunks

        chunks_m = temporal_chunks(mags, self.n_pyr)
        chunks_a = temporal_chunks(angs, self.n_pyr)

        # Histogram features per chunk
        feat = []
        for cm, ca in zip(chunks_m, chunks_a):
            # mean over time window
            m = cm.mean(axis=0)
            a = ca.mean(axis=0)
            # global stats
            feat += [m.mean(), m.std(), m.max()]
            # coarse orientation hist (8 bins)
            hist, _ = np.histogram(a, bins=8, range=(0.0, 1.0))
            feat += list(hist.astype(np.float32))

        v = np.asarray(feat, dtype=np.float32)
        # Project/expand to fixed dim
        if v.shape[0] < self.dim:
            reps = int(np.ceil(self.dim / v.shape[0]))
            v = np.tile(v, reps)[: self.dim]
        else:
            v = v[: self.dim]
        # Normalize
        n = np.linalg.norm(v) + 1e-9
        return (v / n).astype(np.float32)

    def extract(self, frames_or_text: Union[str, List, np.ndarray, "torch.Tensor"]) -> np.ndarray:
        # Text fallback
        if isinstance(frames_or_text, str):
            return _hash_vec(frames_or_text, self.dim)

        frames = _frames_from_input(frames_or_text)
        if not frames or len(frames) < 2:
            # Not enough frames → zero vec
            return np.zeros(self.dim, dtype=np.float32)

        flows = []
        # Standardize size & gray
        for i in range(len(frames) - 1):
            f0 = _resize(_as_numpy_frame(frames[i]), (256, 256))
            f1 = _resize(_as_numpy_frame(frames[i + 1]), (256, 256))
            g0 = _ensure_gray(f0)
            g1 = _ensure_gray(f1)
            flows.append(self._flow_pair(g0, g1))

        return self._pool_flow(flows)


# =========================
# DeepForgeryDetector (ELA + LBP)
# =========================

class DeepForgeryDetector:
    """
    Compute ELA (Error Level Analysis) + LBP histogram and pool into a fixed vector.
    Works with a single image or any frame stack (uses the middle frame).
    """

    def __init__(self, dim: int = 256, ela_quality: int = 85, ela_scale: float = 1.0,
                 lbp_radius: int = 1, lbp_points: int = 8):
        self.dim = int(dim)
        self.ela_quality = int(ela_quality)
        self.ela_scale = float(ela_scale)
        self.lbp_radius = int(lbp_radius)
        self.lbp_points = int(lbp_points)

    def _jpeg_reencode(self, rgb: np.ndarray) -> np.ndarray:
        if not _HAS_CV2:
            return rgb.copy()
        # Encode to JPEG and decode back
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), self.ela_quality])
        if not ok:
            return rgb.copy()
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if dec is None:
            return rgb.copy()
        dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        return dec

    def _ela_map(self, rgb: np.ndarray) -> np.ndarray:
        rec = self._jpeg_reencode(rgb)
        diff = cv2.absdiff(rgb, rec) if _HAS_CV2 else np.abs(rgb.astype(np.float32) - rec.astype(np.float32)).astype(np.uint8)
        # amplify ELA magnitude
        ela = np.clip(diff.astype(np.float32) * self.ela_scale, 0, 255).astype(np.uint8)
        return ela

    def _lbp_hist(self, gray: np.ndarray) -> np.ndarray:
        if _HAS_SKIMAGE:
            lbp = local_binary_pattern(gray, P=self.lbp_points, R=self.lbp_radius, method="uniform")
            n_bins = self.lbp_points + 2
            hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
            return hist.astype(np.float32)
        # fallback: simple 3x3 sign pattern
        H, W = gray.shape
        out = []
        for y in range(1, H - 1, 2):
            for x in range(1, W - 1, 2):
                patch = gray[y - 1:y + 2, x - 1:x + 2].astype(np.float32)
                c = patch[1, 1]
                code = (patch > c).astype(np.int32)
                out.append(int(code.sum()))
        hist, _ = np.histogram(np.array(out), bins=10, range=(0, 10), density=True)
        return hist.astype(np.float32)

    def ela_lbp(self, image_or_frames_or_text: Union[str, np.ndarray, List, "torch.Tensor"]) -> np.ndarray:
        # Text fallback
        if isinstance(image_or_frames_or_text, str):
            return _hash_vec(image_or_frames_or_text, self.dim)

        frames = _frames_from_input(image_or_frames_or_text)
        if frames:
            img = frames[len(frames) // 2]  # middle frame
        else:
            # single image array?
            img = _as_numpy_frame(image_or_frames_or_text)

        if img is None:
            return np.zeros(self.dim, dtype=np.float32)

        img = _resize(img, (256, 256))
        ela = self._ela_map(img)
        gray = _ensure_gray(ela)

        # Stats on ELA magnitude
        stats = np.array([ela.mean(), ela.std(), ela.max(), ela.min()], dtype=np.float32)

        # LBP histogram on ELA gray
        lbp_h = self._lbp_hist(gray)

        v = np.concatenate([stats, lbp_h], axis=0)
        # Project/expand to fixed dim
        if v.shape[0] < self.dim:
            reps = int(np.ceil(self.dim / v.shape[0]))
            v = np.tile(v, reps)[: self.dim]
        else:
            v = v[: self.dim]
        # Normalize
        n = np.linalg.norm(v) + 1e-9
        return (v / n).astype(np.float32)


# =========================
# FaceWarpAnalyzer (quick score)
# =========================

class FaceWarpAnalyzer:
    """
    Quick anomaly score in [0,1] using gradient (Sobel) & ELA magnitude.
    This is intentionally lightweight and robust.
    """
    def __init__(self):
        pass

    def score(self, image_or_frames_or_text: Union[str, np.ndarray, List, "torch.Tensor"]) -> float:
        # Text fallback
        if isinstance(image_or_frames_or_text, str):
            h = abs(hash(image_or_frames_or_text)) % 1000
            return float((h % 100) / 100.0)

        frames = _frames_from_input(image_or_frames_or_text)
        if frames:
            img = frames[len(frames) // 2]
        else:
            img = _as_numpy_frame(image_or_frames_or_text)

        if img is None:
            return 0.0

        img = _resize(img, (256, 256))
        gray = _ensure_gray(img)

        # Gradient-based sharpness/warping cue
        if _HAS_CV2:
            sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sx * sx + sy * sy)
        else:
            # simple finite differences
            gx = np.zeros_like(gray, dtype=np.float32)
            gy = np.zeros_like(gray, dtype=np.float32)
            gx[:, 1:] = gray[:, 1:].astype(np.float32) - gray[:, :-1].astype(np.float32)
            gy[1:, :] = gray[1:, :].astype(np.float32) - gray[:-1, :].astype(np.float32)
            grad_mag = np.sqrt(gx * gx + gy * gy)

        g_mean = float(grad_mag.mean())
        g_std = float(grad_mag.std())

        # ELA mean as artifact cue
        ela = DeepForgeryDetector(dim=16). _ela_map(img)  # small dim instance just to reuse the method
        ela_mean = float(ela.mean()) / 255.0

        # Heuristic fusion: more artifacts → higher score
        score = 0.5 * np.tanh((g_std / (g_mean + 1e-6))) + 0.5 * ela_mean
        return float(np.clip(score, 0.0, 1.0))
