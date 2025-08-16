from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

# Core blocks
from src.core_blocks.text_blocks import BERTContextEncoder
from src.core_blocks.audio_blocks import SpectralForensics
from src.core_blocks.visual_blocks import OpticalFlow3DCNN, DeepForgeryDetector
from src.core_blocks.temporal_blocks import TemporalSyncNet


class FakeSVRawDataset:
    """
    Minimal raw dataset wrapper for the FakeSV layout.

    Expects data_root containing:
      - data_complete.json
      - video_comment/   (optional for this cache)
      - videos/          (optional for this cache)

    We use JSON textual fields (title, ocr, comments) for cache building,
    plus very light image/text proxies (so it works even without frames).
    """
    def __init__(self, data_root: str):
        self.root = Path(data_root)
        self.json_path = self.root / "data_complete.json"
        if not self.json_path.exists():
            raise FileNotFoundError(f"data_complete.json not found at {self.json_path}")

        self.records: List[Dict[str, Any]] = []
        with open(self.json_path, "r", encoding="utf-8") as f:
            # Support one big JSON array or JSONL
            first = f.read(1)
            f.seek(0)
            if first == "[":
                self.records = json.load(f)
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))

        # Label normalization: 辟谣->0 (real), 假->1 (fake). Fallback 0.
        def label_of(rec: Dict[str, Any]) -> int:
            ann = (rec.get("annotation") or "").strip()
            if ann in ("假", "fake"):
                return 1
            if ann in ("辟谣", "true", "real"):
                return 0
            return 0

        self.labels = np.array([label_of(r) for r in self.records], dtype=np.int64)

    def __len__(self):
        return len(self.records)

    def get_item(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        title = r.get("title") or ""
        ocr = r.get("ocr") or ""
        comments = r.get("comments") or []
        if isinstance(comments, str):
            comments = [comments]
        return {
            "id": r.get("video_id") or f"rec_{idx}",
            "title": title,
            "ocr": ocr,
            "comments": comments,
            "label": int(self.labels[idx]),
        }

    def augment_audio(self, audio):
        """ Inject random noise or apply transformations to audio """
        noise_factor = np.random.uniform(0.005, 0.05)
        noise = np.random.randn(len(audio))
        audio = audio + noise_factor * noise
        return audio

    def augment_video(self, video):
        """ Apply transformations like random flip or rotation to video frames """
        if np.random.rand() < 0.5:
            video = np.flip(video, axis=1)  # Horizontal flip
        if np.random.rand() < 0.5:
            video = np.rot90(video, k=np.random.choice([1, 2, 3]))  # Random rotation
        return video

    def augment_text(self, text):
        """ Apply simple text paraphrasing or random word substitution for augmentation """
        words = text.split()
        if len(words) > 2:
            rand_index = np.random.randint(0, len(words)-1)
            words[rand_index] = "random"  # Just a simple example
        return " ".join(words)

def build_gnn_cache_from_raw_dataset(
    raw: FakeSVRawDataset,
    ocr_phrase_pkl: Optional[str] = None,
    text_dim: int = 768,
    audio_dim: int = 128,
    visual_dim: int = 512,
    temporal_dim: int = 256,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Build a tensor-friendly cache + OCR phrase sets for the trainer & GNN.

    Returns dict with:
      ids:        (N,)   object array of str
      labels:     (N,)   int64
      text:       (N,768)
      audio:      (N,128)
      visual:     (N,512)
      temporal:   (N,256)
      aux:        (N,2)  [temporal_delay_proxy, emotion_intensity_proxy]
      ocr_sets:   list[set[str]] length N
      split:      (train_idx, val_idx, test_idx)
    """
    rng = np.random.default_rng(seed)

    # Instantiate blocks once
    text_enc = BERTContextEncoder(dim=text_dim)
    aud_enc = SpectralForensics(dim=audio_dim)
    vis_flow = OpticalFlow3DCNN(dim=visual_dim)      # we'll fuse with ELA then resize safely
    vis_ela  = DeepForgeryDetector(dim=visual_dim)
    tsync    = TemporalSyncNet(in_dim=text_dim, out_dim=temporal_dim)

    N = len(raw)
    ids: List[str] = []
    labels = np.zeros((N,), dtype=np.int64)
    T = np.zeros((N, text_dim), dtype=np.float32)
    A = np.zeros((N, audio_dim), dtype=np.float32)
    V = np.zeros((N, visual_dim), dtype=np.float32)
    U = np.zeros((N, temporal_dim), dtype=np.float32)
    AUX = np.zeros((N, 2), dtype=np.float32)  # [delay_proxy, emotion_intensity]
    ocr_sets: List[set] = []

    def _l2n(x: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(x) + 1e-9)
        return (x / n).astype(np.float32)

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        an = float(np.linalg.norm(a) + 1e-9)
        bn = float(np.linalg.norm(b) + 1e-9)
        return float(np.dot(a, b) / (an * bn))

    for i in range(N):
        rec = raw.get_item(i)
        ids.append(rec["id"])
        labels[i] = rec["label"]

        # --- Text features
        t_vec = text_enc.encode_fields(rec["title"], rec["ocr"], rec["comments"])  # (768,)
        T[i] = t_vec

        # --- Audio features (text proxy OK for cache)
        audio_proxy_text = (rec["title"] or "") + " " + (" ".join(rec["comments"][:1]) if rec["comments"] else "")
        A[i] = aud_enc.extract(audio_proxy_text)

        # --- Visual features: flow + ELA → concat → size-safe to visual_dim, then L2-norm
        # (We use text/ocr proxies so this works even without frames)
        v_flow = vis_flow.extract(rec["ocr"] or rec["title"] or "")
        v_ela  = vis_ela.ela_lbp(rec["ocr"] or rec["title"] or "")
        v_comb = np.concatenate([v_flow.astype(np.float32), v_ela.astype(np.float32)], axis=0)
        if v_comb.shape[0] >= visual_dim:
            V[i] = v_comb[:visual_dim]
        else:
            tmp = np.zeros((visual_dim,), dtype=np.float32)
            tmp[: v_comb.shape[0]] = v_comb
            V[i] = tmp
        V[i] = _l2n(V[i])

        # --- Temporal features from text↔visual alignment (256-D)
        U[i] = tsync.align(T[i], V[i])

        # --- Aux scalars (compare 256‑D to 256‑D)
        u_tv = U[i]                         # text vs visual
        u_tt = tsync.align(T[i], T[i])      # text vs itself (reference)
        delay_proxy = float(np.clip(1.0 - _cos(u_tt, u_tv), 0.0, 1.0))

        # Emotion proxy: heuristic count of “sensational” terms (title+ocr)
        emo_terms = ["恐惧", "警告", "危险", "外星", "消失", "危机", "谣言", "假"]
        emo_hits = sum(w in ((rec["title"] or "") + (rec["ocr"] or "")) for w in emo_terms)
        emo_intensity = float(min(1.0, 0.1 * emo_hits))

        AUX[i, 0] = delay_proxy
        AUX[i, 1] = emo_intensity

        # --- OCR phrase set for GNN edges (very lightweight tokenizer)
        phrases = set()
        for tok in (rec["ocr"] or "").replace("\t", " ").replace("\n", " ").split():
            tok = tok.strip()
            if len(tok) >= 2:
                phrases.add(tok)
        ocr_sets.append(phrases)

    ids = np.array(ids, dtype=object)

    # ---------- Stratified split (70/15/15) with safe fallbacks ----------
    def stratified_indices(y: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
        """Pick approx fraction per class; at least 1 if class exists."""
        take = []
        classes = np.unique(y)
        for c in classes:
            cls_idx = np.where(y == c)[0]
            if cls_idx.size == 0:
                continue
            rng.shuffle(cls_idx)
            k = max(1, int(round(frac * cls_idx.size)))
            take.append(cls_idx[:k])
        return np.concatenate(take) if len(take) else np.array([], dtype=int)

    all_idx = np.arange(N)
    rng.shuffle(all_idx)

    # train ~70%
    tr_idx = stratified_indices(labels, 0.70, rng)
    # remaining pool
    rem = np.setdiff1d(all_idx, tr_idx, assume_unique=False)
    rem_labels = labels[rem]

    # val ~15% of total (so 15/30 ≈ 0.5 of the remainder)
    val_frac_of_rem = 0.0
    if rem.size > 0:
        val_frac_of_rem = min(1.0, 0.15 / (rem.size / float(N)))
    va_take = stratified_indices(rem_labels, val_frac_of_rem, rng)
    va_idx = rem[va_take]

    # test = remainder
    te_idx = np.setdiff1d(rem, va_idx, assume_unique=False)

    # final safety: ensure no split is empty
    if tr_idx.size == 0 and N > 0:
        tr_idx = all_idx[: max(1, int(0.7 * N))]
    if va_idx.size == 0 and N > 1:
        va_idx = all_idx[max(1, int(0.7 * N)) : max(1, int(0.85 * N))]
    if te_idx.size == 0 and N > 2:
        te_idx = np.setdiff1d(all_idx, np.concatenate([tr_idx, va_idx]), assume_unique=False)

    cache = {
        "ids": ids,
        "labels": labels,
        "text": T,
        "audio": A,
        "visual": V,
        "temporal": U,
        "aux": AUX,
        "ocr_sets": ocr_sets,
        "split": (tr_idx, va_idx, te_idx),
    }
    return cache
