# src/core_blocks/audio_blocks.py
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn

# Optional deps
try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    _HAS_W2V2 = True
except Exception:
    _HAS_W2V2 = False

from src.utils.config_utils import ConfigManager


# -----------------------------
# Small utilities
# -----------------------------

def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)

def _ensure_mono_16k(wave: Union[np.ndarray, torch.Tensor], sr: int) -> (np.ndarray, int):
    """
    Ensure mono waveform at 16k. Accepts torch or numpy; returns numpy float32.
    """
    wav = _to_numpy(wave).astype(np.float32)
    if wav.ndim == 2:  # [C, T] -> mono
        wav = wav.mean(axis=0)
    # resample to 16k if librosa is available and sr != 16000
    if sr != 16000 and _HAS_LIBROSA:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000
    return wav, sr

def _hash_embed(text: str, dim: int) -> np.ndarray:
    """Deterministic light embedding for text proxies (no external model)."""
    v = np.zeros(dim, dtype=np.float32)
    for tok in (text or "").split()[: dim]:
        v[hash(tok) % dim] += 1.0
    n = np.linalg.norm(v) + 1e-9
    return v / n


# -----------------------------
# Mel Spectrogram (utility)
# -----------------------------

class MelSpectrogramGenerator:
    """
    Lightweight mel-spectrogram generator for analysis/visualization.
    Returns either a 2D mel matrix or a flattened vector depending on use.
    """
    def __init__(self, sr: int = 16000, n_mels: int = 64, n_fft: int = 400, hop_length: int = 160):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop = hop_length

    def generate(self, wave: Union[np.ndarray, torch.Tensor], sr: int = 16000, flatten: bool = True) -> np.ndarray:
        wav, sr = _ensure_mono_16k(wave, sr)
        if _HAS_LIBROSA:
            S = librosa.feature.melspectrogram(
                y=wav, sr=sr, n_fft=self.n_fft, hop_length=self.hop, n_mels=self.n_mels, power=2.0
            )
            S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)  # [n_mels, frames]
            return S_db.flatten().astype(np.float32) if flatten else S_db
        # Torch fallback: coarse STFT magnitude + mel-like pooling
        t = torch.from_numpy(wav).float()
        spec = torch.stft(t, n_fft=self.n_fft, hop_length=self.hop, return_complex=True)
        mag = spec.abs().numpy()  # [freq, time]
        # simple downsample to n_mels bands
        freq_bins = mag.shape[0]
        if freq_bins <= self.n_mels:
            mel_like = mag
        else:
            factor = freq_bins // self.n_mels
            mel_like = mag[: self.n_mels * factor].reshape(self.n_mels, factor, -1).mean(axis=1)
        mel_like = mel_like.astype(np.float32)
        return mel_like.flatten().astype(np.float32) if flatten else mel_like


# -----------------------------
# Spectral Forensics
# -----------------------------

class SpectralForensics(nn.Module):
    """
    Audio feature extractor for fake/tamper cues.
    Priorities:
      1) Use Wav2Vec2 embeddings if available.
      2) Else compute rich spectral stats via librosa.
      3) Else fallback to hashed proxy for text input or simple STFT stats.

    Output: fixed-size vector (dim, default 128)
    Accepts:
      - wave: Union[np.ndarray, torch.Tensor] (mono or stereo), sr: int
      - OR a text proxy str (e.g., title/comments) when raw audio isn’t accessible.
    """
    def __init__(self, dim: int = 128, w2v2_name: str = "facebook/wav2vec2-base-960h"):
        super().__init__()
        self.dim = int(dim)
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        self.dtype = torch.float32

        self.use_w2v2 = _HAS_W2V2
        if self.use_w2v2:
            try:
                self.processor = Wav2Vec2Processor.from_pretrained(w2v2_name)
                self.backbone = Wav2Vec2Model.from_pretrained(w2v2_name)
                self.backbone.to(self.device, dtype=self.dtype)
                self.hidden = int(self.backbone.config.hidden_size)
                # project to fixed dim if needed
                self.proj = nn.Identity() if self.hidden == self.dim else nn.Linear(self.hidden, self.dim)
                self.proj.to(self.device, dtype=self.dtype)
            except Exception:
                self.use_w2v2 = False
                self.processor, self.backbone, self.proj = None, None, None

    @torch.inference_mode()
    def _w2v2_features(self, wav: np.ndarray) -> np.ndarray:
        inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.backbone(**inputs).last_hidden_state  # [1, T, H]
        # mean pool over time
        pooled = out.mean(dim=1)  # [1, H]
        pooled = self.proj(pooled)  # [1, dim]
        return pooled.detach().cpu().to(torch.float32).numpy()[0]

    def _librosa_spectral(self, wav: np.ndarray) -> np.ndarray:
        # Core descriptors
        feats = []
        S = np.abs(librosa.stft(wav, n_fft=400, hop_length=160))  # [freq, time]
        # STFT magnitude stats
        feats += [S.mean(), S.std(), S.max(), S.min()]
        # Spectral contrast
        try:
            sc = librosa.feature.spectral_contrast(S=S, sr=16000)
            feats += [sc.mean(), sc.std()]
        except Exception:
            feats += [0.0, 0.0]
        # Spectral flatness
        try:
            sf = librosa.feature.spectral_flatness(S=S)
            feats += [sf.mean(), sf.std()]
        except Exception:
            feats += [0.0, 0.0]
        # Centroid / rolloff / ZCR
        try:
            cent = librosa.feature.spectral_centroid(y=wav, sr=16000).mean()
            roll = librosa.feature.spectral_rolloff(y=wav, sr=16000).mean()
            zcr = librosa.feature.zero_crossing_rate(y=wav).mean()
            feats += [cent, roll, zcr]
        except Exception:
            feats += [0.0, 0.0, 0.0]
        v = np.asarray(feats, dtype=np.float32)
        # Project/expand to fixed dim
        if v.shape[0] < self.dim:
            reps = int(np.ceil(self.dim / v.shape[0]))
            v = np.tile(v, reps)[: self.dim]
        else:
            v = v[: self.dim]
        # Normalize
        n = np.linalg.norm(v) + 1e-9
        return (v / n).astype(np.float32)

    def _torch_stft_fallback(self, wav: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(wav).float()
        spec = torch.stft(t, n_fft=400, hop_length=160, return_complex=True).abs().numpy()
        feats = np.array([spec.mean(), spec.std(), spec.max(), spec.min()], dtype=np.float32)
        if feats.shape[0] < self.dim:
            reps = int(np.ceil(self.dim / feats.shape[0]))
            feats = np.tile(feats, reps)[: self.dim]
        else:
            feats = feats[: self.dim]
        n = np.linalg.norm(feats) + 1e-9
        return (feats / n).astype(np.float32)

    def extract(
        self,
        audio_or_text: Union[str, np.ndarray, torch.Tensor],
        sr: int = 16000
    ) -> np.ndarray:
        """
        Main entry:
          - If a string is provided, returns a deterministic hashed embedding (dim).
          - If a waveform is provided, returns Wav2Vec2 features (if available),
            else librosa/tTorch spectral stats (dim).
        """
        # Case 1: text proxy
        if isinstance(audio_or_text, str):
            return _hash_embed(audio_or_text, self.dim)

        # Case 2: waveform
        wav, sr = _ensure_mono_16k(audio_or_text, sr)

        if self.use_w2v2:
            try:
                return self._w2v2_features(wav)
            except Exception:
                pass

        if _HAS_LIBROSA:
            try:
                return self._librosa_spectral(wav)
            except Exception:
                pass

        return self._torch_stft_fallback(wav)


# -----------------------------
# Voice cloning/tamper scoring
# -----------------------------

class VoiceCloneDetector:
    """
    Lightweight tamper likelihood proxy. Returns a float in [0,1].
    If librosa is available, uses a mix of simple descriptors; else a string proxy hash or waveform energy proxy.
    """
    def __init__(self):
        pass

    def score(self, audio_or_text: Union[str, np.ndarray, torch.Tensor], sr: int = 16000) -> float:
        # Text proxy path
        if isinstance(audio_or_text, str):
            # stable hash → pseudo-probability
            h = abs(hash(audio_or_text)) % 1000
            return float((h % 100) / 100.0)

        # Waveform path
        wav, sr = _ensure_mono_16k(audio_or_text, sr)
        if _HAS_LIBROSA:
            try:
                zcr = librosa.feature.zero_crossing_rate(wav).mean()
                flat = librosa.feature.spectral_flatness(y=wav).mean()
                cent = librosa.feature.spectral_centroid(y=wav, sr=sr).mean()
                # heuristic mapping to 0..1
                # higher flatness + certain centroid bands + zcr → more suspicious
                score = 0.4 * float(flat) + 0.3 * float(zcr) + 0.3 * float(np.tanh(cent / 3000.0))
                return float(np.clip(score, 0.0, 1.0))
            except Exception:
                pass
        # Fallback: energy proxy
        e = float(np.mean(np.square(wav)))
        return float(np.clip(e / (e + 1.0), 0.0, 1.0))
