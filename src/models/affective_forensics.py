# src/models/affective_forensics.py
from typing import Dict, Optional, Union
import numpy as np

# Optional deps
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _HAS_TXT = True
except Exception:
    _HAS_TXT = False

try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False


# -----------------------------
# Light lexicon (fallback)
# -----------------------------
_EMO_LEX = {
    "fear": {"恐惧", "警告", "危险", "外星", "消失", "危机", "害怕", "恐怖"},
    "anger": {"愤怒", "欺骗", "骗局", "谣言", "假", "讨厌", "生气"},
    "joy": {"真相", "辟谣", "科学", "证据", "研究", "发现", "开心", "高兴"},
}

def _lexicon_probs(text: str) -> Dict[str, float]:
    if not text:
        return {"fear": 0.0, "anger": 0.0, "joy": 0.0}
    c = {"fear": 0.0, "anger": 0.0, "joy": 0.0}
    for k, ws in _EMO_LEX.items():
        for w in ws:
            if w in text:
                c[k] += 1.0
    s = sum(c.values()) + 1e-9
    if s == 0:
        return {"fear": 0.0, "anger": 0.0, "joy": 0.0}
    return {k: v / s for k, v in c.items()}


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


# -----------------------------
# AffectiveForensics
# -----------------------------

class AffectiveForensics:
    """
    Unified emotion estimation from text (+ optional audio prosody).
    Exposes:
      - analyze(text, audio) -> {
            "probs": {"fear":p, "anger":p, "joy":p},
            "intensity": float in [0,1],
            "arousal": float in [0,1],
            "valence": float in [0,1]
        }
      - get_emotion_intensity(text, audio) -> float ∈ [0,1]
    """

    def __init__(self, text_model: str = "j-hartmann/emotion-english-distilroberta-base"):
        self.text_model_name = text_model
        self.use_text_model = False
        self.device = None

        if _HAS_TXT:
            try:
                self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
                self.tok = AutoTokenizer.from_pretrained(text_model)
                self.txt_model = AutoModelForSequenceClassification.from_pretrained(text_model)
                self.txt_model.to(self.device)
                self.txt_model.eval()
                self.labels = getattr(self.txt_model.config, "id2label", {i: str(i) for i in range(self.txt_model.config.num_labels)})
                self.use_text_model = True
            except Exception:
                self.use_text_model = False
                self.tok = None
                self.txt_model = None
                self.labels = {"0": "neutral"}

    def _text_probs(self, text: str) -> Dict[str, float]:
        if not text:
            return {"fear": 0.0, "anger": 0.0, "joy": 0.0}
        if self.use_text_model and self.tok is not None:
            try:
                with torch.inference_mode():
                    inp = self.tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(self.device)
                    out = self.txt_model(**inp).logits  # [1, C]
                    p = torch.softmax(out, dim=-1)[0].detach().cpu().numpy()
                    # Map generic labels to our 3 heads (fear/anger/joy) conservatively
                    names = [self.labels.get(i, str(i)).lower() for i in range(len(p))]
                    fear = float(p[[i for i, n in enumerate(names) if any(k in n for k in ["fear", "anx", "worr", "scare"])]].sum())
                    anger = float(p[[i for i, n in enumerate(names) if any(k in n for k in ["anger", "annoy", "mad", "rage"])]].sum())
                    joy = float(p[[i for i, n in enumerate(names) if any(k in n for k in ["joy", "happi", "delight", "amuse"])]].sum())
                    s = fear + anger + joy + 1e-9
                    if s == 0:
                        return {"fear": 0.0, "anger": 0.0, "joy": 0.0}
                    return {"fear": fear / s, "anger": anger / s, "joy": joy / s}
            except Exception:
                pass
        # fallback to lexicon
        return _lexicon_probs(text)

    def _audio_arousal(self, audio: Optional[np.ndarray], sr: int = 16000) -> float:
        if audio is None:
            return 0.5
        if _HAS_LIBROSA:
            try:
                # crude arousal proxy: energy + pitch stability (more energy → higher arousal)
                en = float(np.mean(audio ** 2))
                try:
                    f0, _, _ = librosa.pyin(audio, fmin=65, fmax=300, sr=sr)
                    f0 = f0[np.isfinite(f0)]
                    pit = float(np.mean(f0)) if f0.size else 0.0
                    pit_std = float(np.std(f0)) if f0.size else 0.0
                except Exception:
                    pit = float(librosa.feature.spectral_centroid(y=audio, sr=sr).mean())
                    pit_std = float(0.0)
                a = _sigmoid(np.tanh(5.0 * en) + np.tanh((pit / 300.0)) - 0.5 * np.tanh(pit_std / 50.0))
                return float(np.clip(a, 0.0, 1.0))
            except Exception:
                pass
        # Fallback: RMS energy only
        en = float(np.mean(audio ** 2)) if audio is not None else 0.0
        return float(np.clip(_sigmoid(5.0 * en), 0.0, 1.0))

    def analyze(self, text: Optional[str] = None, audio: Optional[np.ndarray] = None, sr: int = 16000) -> Dict[str, Union[float, Dict[str, float]]]:
        probs = self._text_probs(text or "")
        # intensity: penalize joy, reward fear/anger (we’re detecting manipulation/rumor)
        raw = (probs["fear"] + probs["anger"] - 0.5 * probs["joy"])
        text_intensity = float(np.clip(_sigmoid(2.5 * raw), 0.0, 1.0))

        arousal = self._audio_arousal(audio, sr=sr)
        # fuse: simple average
        intensity = float(np.clip(0.6 * text_intensity + 0.4 * arousal, 0.0, 1.0))

        # rough valence proxy (joy high → positive)
        valence = float(np.clip(0.5 + 0.5 * (probs["joy"] - 0.5 * (probs["fear"] + probs["anger"])), 0.0, 1.0))

        return {
            "probs": probs,
            "intensity": intensity,
            "arousal": arousal,
            "valence": valence,
        }

    def get_emotion_intensity(self, text: Optional[str] = None, audio: Optional[np.ndarray] = None, sr: int = 16000) -> float:
        return float(self.analyze(text=text, audio=audio, sr=sr)["intensity"])
