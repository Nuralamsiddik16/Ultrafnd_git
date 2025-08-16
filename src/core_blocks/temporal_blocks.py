# src/core_blocks/temporal_blocks.py
from typing import Optional, Union, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    an = a / (a.norm(dim=-1, keepdim=True) + eps)
    bn = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (an * bn).sum(dim=-1, keepdim=True)  # [B,1]


class _TinyTCN(nn.Module):
    """
    Minimal Temporal Convolutional block (Dilated 1D convs) for optional sequence inputs.
    Not used by default in align(), but available if you pass sequences to forward().
    """
    def __init__(self, in_ch: int, hid: int = 128, layers: int = 2, k: int = 3, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        ch = in_ch
        for i in range(layers):
            self.convs.append(nn.Conv1d(ch, hid, kernel_size=k, padding='same', dilation=2**i))
            self.norms.append(nn.BatchNorm1d(hid))
            ch = hid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        h = x
        for conv, bn in zip(self.convs, self.norms):
            z = conv(h)
            z = F.gelu(bn(z))
            z = self.drop(z)
            # residual if channel sizes match
            if z.shape == h.shape:
                h = h + z
            else:
                h = z
        return h  # [B, C, T]


class TemporalSyncNet(nn.Module):
    """
    Temporal synchronization and cross-modal alignment.

    Primary APIs used elsewhere:
      - align(text_vec, visual_vec) -> np.ndarray[out_dim]        # vector-level fusion for temporal cues
      - delay_score(audio_len, video_len) -> float in [0,1]       # lightweight proxy if only lengths known

    Optional advanced (available for later use):
      - forward(text_seq, vis_seq) -> torch.Tensor[B, out_dim]    # sequence-aware encoding via tiny TCN+pool
      - estimate_av_lag(audio_env, mouth_open, sr, fps) -> seconds (can be mapped to 0..1 if desired)
    """

    def __init__(
        self,
        in_dim: int = 768,         # dimensionality of input single-modality vectors (e.g., text)
        out_dim: int = 256,        # final temporal-alignment feature size
        use_tcn: bool = False,     # enable if you pass sequences to forward()
        tcn_hid: int = 128,
        tcn_layers: int = 2,
        tcn_kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        # Device: MPS on Apple Silicon if available
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.dtype = torch.float32

        # Vector-level projector for align()
        # Input to projector is [t, v, t-v, t*v, cos]  ->  4*D + 1
        proj_in = 4 * self.in_dim + 1
        self.proj = nn.Sequential(
            nn.Linear(proj_in, 2 * self.out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.out_dim, self.out_dim),
        )

        # Optional sequence encoder (TCN + pooling)
        self.use_tcn = bool(use_tcn)
        if self.use_tcn:
            self.tcn = _TinyTCN(in_ch=self.in_dim, hid=tcn_hid, layers=tcn_layers, k=tcn_kernel, dropout=dropout)
            self.head = nn.Linear(tcn_hid * 2, self.out_dim)  # global mean & max pool concat
        else:
            self.tcn = None
            self.head = None

        self.to(self.device, dtype=self.dtype)

    # ---------------------------
    # Vector-only path (used now)
    # ---------------------------
    @torch.inference_mode()
    def align(self, text_vec: Union[np.ndarray, torch.Tensor], visual_vec: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Align single text & visual vectors into a temporal-consistency embedding.
        Uses cross-modal interactions (t, v, t-v, t*v, cosine) fed to an MLP.
        Returns: np.ndarray[out_dim]
        """
        # Convert to torch on device
        if isinstance(text_vec, np.ndarray):
            t = torch.from_numpy(text_vec).to(self.device, dtype=self.dtype).unsqueeze(0)  # [1,D]
        elif isinstance(text_vec, torch.Tensor):
            t = text_vec.to(self.device, dtype=self.dtype).unsqueeze(0) if text_vec.ndim == 1 else text_vec.to(self.device, dtype=self.dtype)
        else:
            raise TypeError("text_vec must be np.ndarray or torch.Tensor")

        if isinstance(visual_vec, np.ndarray):
            v = torch.from_numpy(visual_vec).to(self.device, dtype=self.dtype).unsqueeze(0)
        elif isinstance(visual_vec, torch.Tensor):
            v = visual_vec.to(self.device, dtype=self.dtype).unsqueeze(0) if visual_vec.ndim == 1 else visual_vec.to(self.device, dtype=self.dtype)
        else:
            raise TypeError("visual_vec must be np.ndarray or torch.Tensor")

        # Ensure same dim (pad/trunc if needed)
        D = t.shape[-1]
        if v.shape[-1] != D:
            if v.shape[-1] < D:
                pad = torch.zeros((v.shape[0], D - v.shape[-1]), device=v.device, dtype=v.dtype)
                v = torch.cat([v, pad], dim=-1)
            else:
                v = v[..., :D]

        # Cross-modal interactions
        tv_cos = _cosine(t, v)             # [B,1]
        diff = t - v                       # [B,D]
        prod = t * v                       # [B,D]
        feat = torch.cat([t, v, diff, prod, tv_cos], dim=-1)  # [B, 4D+1]

        out = self.proj(feat)              # [B, out_dim]
        return out.detach().cpu().numpy()[0].astype(np.float32)

    # -------------------------------------------
    # Optional: sequence path if you have sequences
    # -------------------------------------------
    def forward(self, text_seq: torch.Tensor, vis_seq: torch.Tensor) -> torch.Tensor:
        """
        Sequence-aware path (optional):
          text_seq: [B, T, D]
          vis_seq:  [B, T, D]
        We concat then pass through a small TCN, then global pool.
        """
        assert self.use_tcn, "Enable use_tcn=True to use the sequence path."
        # concat along channels: [B,T,2D] -> [B,2D,T] for Conv1d
        x = torch.cat([text_seq, vis_seq], dim=-1).to(self.device, dtype=self.dtype)  # [B,T,2D]
        x = x.transpose(1, 2)  # [B,2D,T]
        h = self.tcn(x)        # [B,H,T]
        g_mean = h.mean(dim=-1)                # [B,H]
        g_max, _ = h.max(dim=-1)               # [B,H]
        out = self.head(torch.cat([g_mean, g_max], dim=-1))  # [B,out_dim]
        return out

    # -------------------------------------------
    # Delay estimators
    # -------------------------------------------
    @staticmethod
    def delay_score(audio_len: int, video_len: int) -> float:
        """
        Lightweight proxy when only lengths are known; returns 0..1.
        0 = perfectly matched; 1 = highly mismatched.
        """
        a = float(max(0, audio_len))
        v = float(max(0, video_len))
        m = max(1.0, max(a, v))
        return float(abs(a - v) / m)

    @staticmethod
    def estimate_av_lag(
        audio_envelope: Union[np.ndarray, torch.Tensor],
        mouth_open:     Union[np.ndarray, torch.Tensor],
        sr: float = 16000.0,
        fps: float = 25.0,
        max_lag_s: float = 0.5,
    ) -> float:
        """
        Cross-correlation based A/V lag estimate (seconds).
        Inputs should be 1D envelopes aligned in time. If they differ in length, shorter is used.
        Positive result ~ audio leads; negative ~ video leads.
        """
        # to numpy 1D float32
        def to1d(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().float().numpy()
            x = np.asarray(x).astype(np.float32).ravel()
            return x

        a = to1d(audio_envelope)
        m = to1d(mouth_open)
        L = min(len(a), len(m))
        if L < 4:
            return 0.0
        a = (a[:L] - a[:L].mean()) / (a[:L].std() + 1e-9)
        m = (m[:L] - m[:L].mean()) / (m[:L].std() + 1e-9)

        # xcorr via FFT (fast)
        n = 1
        while n < 2 * L:
            n <<= 1
        A = np.fft.rfft(a, n)
        M = np.fft.rfft(m, n)
        xc = np.fft.irfft(A * np.conj(M), n)  # circular xcorr
        # shift to have lags centered
        xc = np.concatenate([xc[-(L-1):], xc[:L]])

        # convert max lag in samples (use audio rate for safety)
        max_lag = int(max_lag_s * sr)
        center = len(xc) // 2
        lo = max(0, center - max_lag)
        hi = min(len(xc), center + max_lag + 1)
        window = xc[lo:hi]
        lag_idx = np.argmax(window)
        lag_samples = (lo + lag_idx) - center
        lag_seconds = float(lag_samples / sr)
        return lag_seconds
