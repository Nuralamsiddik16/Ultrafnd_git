# src/core_blocks/text_blocks.py
from __future__ import annotations
from typing import List, Optional
import numpy as np

# Optional: HuggingFace (kept optional so the project still runs without it)
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    _HAS_TX = True
except Exception:
    _HAS_TX = False
    torch = None  # type: ignore

_DEFAULT_MODEL = "bert-base-uncased"  # does NOT require sentencepiece
_DIM = 768


def _hash_embed(text: str, dim: int = _DIM) -> np.ndarray:
    """Deterministic, fast fallback embedding so we never crash without HF models."""
    v = np.zeros(dim, dtype=np.float32)
    if not text:
        return v
    for tok in text.split():
        v[hash(tok) % dim] += 1.0
    n = float(np.linalg.norm(v) + 1e-9)
    return (v / n).astype(np.float32)


class BERTContextEncoder:
    """
    Robust 768‑D text encoder with graceful fallback.

    Methods
    -------
    encode(text: str) -> np.ndarray[768]
    encode_fields(title: str, ocr: str, comments: List[str]) -> np.ndarray[768]
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, dim: int = _DIM, max_length: int = 256):
        self.dim = int(dim)
        self.max_length = int(max_length)
        self.use_hf = False
        self.device = None

        if _HAS_TX:
            try:
                self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
                self.tok = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                self.use_hf = True
            except Exception:
                # fall back cleanly if anything fails (offline, cache missing, etc.)
                self.tok = None
                self.model = None
                self.use_hf = False
        else:
            self.tok = None
            self.model = None

    @torch.inference_mode() if _HAS_TX else (lambda f: f)
    def encode(self, text: Optional[str]) -> np.ndarray:
        """Encode a single string into a 768‑D vector. Falls back to hashing if HF unavailable."""
        if not text:
            return np.zeros(self.dim, dtype=np.float32)

        if self.use_hf and self.tok is not None and self.model is not None:
            try:
                enc = self.tok(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc).last_hidden_state  # [1, L, H]

                # Mean‑pool with attention mask where available
                if "attention_mask" in enc:
                    mask = enc["attention_mask"].unsqueeze(-1).float()  # [1, L, 1]
                    summed = (out * mask).sum(dim=1)                     # [1, H]
                    denom = mask.sum(dim=1).clamp_min(1e-6)              # [1, 1]
                    rep = (summed / denom)[0]
                else:
                    rep = out.mean(dim=1)[0]

                vec = rep.detach().cpu().to(torch.float32).numpy()
                # Ensure exact dimension
                if vec.shape[0] != self.dim:
                    if vec.shape[0] > self.dim:
                        vec = vec[: self.dim]
                    else:
                        z = np.zeros(self.dim, dtype=np.float32)
                        z[: vec.shape[0]] = vec
                        vec = z
                # L2 normalize
                vec = vec / (np.linalg.norm(vec) + 1e-9)
                return vec.astype(np.float32)
            except Exception:
                # Graceful fallback if tokenization/model forward fails
                pass

        return _hash_embed(text, dim=self.dim)

    def encode_fields(self, title: Optional[str], ocr: Optional[str], comments: Optional[List[str]]) -> np.ndarray:
        """
        Aggregate title + OCR + up to 10 comments with simple average pooling.
        Returns a stable 768‑D vector.
        """
        parts: List[np.ndarray] = []
        if title:
            parts.append(self.encode(title))
        if ocr:
            parts.append(self.encode(ocr))
        if comments:
            for c in comments[:10]:
                if c:
                    parts.append(self.encode(c))

        if not parts:
            return np.zeros(self.dim, dtype=np.float32)

        M = np.stack(parts, axis=0).astype(np.float32)  # [m, dim]
        v = M.mean(axis=0)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


# ------- Optional helpers (kept for compatibility with earlier drafts) -------

class SemanticAlignmentLayer:
    """Light semantic alignment between two text vectors (placeholder; safe no‑op baseline)."""
    def __init__(self, dim: int = _DIM):
        self.dim = dim
    def align(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a is None or b is None or a.size == 0 or b.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        v = 0.5 * (a.astype(np.float32) + b.astype(np.float32))
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


class MultilingualEmbedding(BERTContextEncoder):
    """
    Alias for multilingual models that do use SentencePiece (e.g., XLM‑R).
    Only switch to this if you install sentencepiece successfully.
    """
    def __init__(self, model_name: str = "xlm-roberta-base", dim: int = _DIM, max_length: int = 256):
        super().__init__(model_name=model_name, dim=dim, max_length=max_length)
