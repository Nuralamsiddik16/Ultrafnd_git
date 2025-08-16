# src/models/semantic_forgery.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from transformers import CLIPProcessor, CLIPModel, AutoProcessor
    _HAS_TX = True
except Exception:
    _HAS_TX = False


def l2n(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@dataclass
class SemanticConfig:
    model_name: str = "openai/clip-vit-base-patch32"
    proj_dim: int = 512
    dropout: float = 0.3  # Increased dropout for better regularization
    use_fast: bool = True  # force fast processor to avoid warning
    max_length: int = 64


class SemanticForgeryAnalyzer(nn.Module):
    """
    Text–visual semantic consistency module.
    Returns a fixed-dim vector (proj_dim) used by the fusion block.

    Inputs (dict):
      - text_features: (B, 768) precomputed BERT/encoder features  [optional for mixing]
      - title: Optional[List[str]]
      - ocr:   Optional[List[str]]

    You can feed raw strings (title/ocr) OR pre-extracted embeddings. If no CLIP is
    available (offline/transformers missing), we return zero vectors with correct shapes.
    """
    def __init__(self, cfg: Optional[SemanticConfig] = None, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg or SemanticConfig()
        self.device = device or (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))

        self.model = None
        self.processor = None
        self.use_clip = False

        if _HAS_TX:
            try:
                # Prefer AutoProcessor to honor use_fast; fallback to CLIPProcessor if needed
                if self.cfg.use_fast:
                    try:
                        self.processor = AutoProcessor.from_pretrained(self.cfg.model_name, use_fast=True)
                    except Exception:
                        self.processor = CLIPProcessor.from_pretrained(self.cfg.model_name)
                else:
                    self.processor = CLIPProcessor.from_pretrained(self.cfg.model_name)

                self.model = CLIPModel.from_pretrained(self.cfg.model_name)
                self.model.to(self.device)
                self.model.eval()
                self.use_clip = True
            except Exception:
                self.model = None
                self.processor = None
                self.use_clip = False

        # Projection to a stable fusion dim with added dropout for regularization
        self.text_proj = nn.Sequential(
            nn.Linear(512, self.cfg.proj_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),  # Increased dropout for regularization
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(512, self.cfg.proj_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),  # Increased dropout for regularization
        )

        # If CLIP isn’t available we’ll route zeros through projs; still fine.
        self.out_dim = self.cfg.proj_dim

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """
        Encode a batch of strings with CLIP text tower → (B, 512).
        If CLIP unavailable, return zeros.
        """
        B = len(texts)
        if self.use_clip and self.processor is not None and self.model is not None:
            try:
                toks = self.processor.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.cfg.max_length,
                    return_tensors="pt",
                )
                toks = {k: v.to(self.device) for k, v in toks.items()}
                out = self.model.get_text_features(**toks)  # (B, 512)
                return l2n(out)
            except Exception:
                pass
        return torch.zeros(B, 512, device=self.device, dtype=torch.float32)

    @torch.inference_mode()
    def encode_image_like(self, texts: list[str]) -> torch.Tensor:
        """
        Stand‑in for image features when we only have OCR/title text (no frames here).
        We use CLIP’s text tower again as a proxy; many works do this for pseudo‑vision
        when frames aren’t present at this stage. Output → (B, 512).
        """
        return self.encode_text(texts)

    def forward(self, batch: Dict[str, torch.Tensor | list[str]]) -> Dict[str, torch.Tensor]:
        """
        batch can contain:
          - "title": list[str] or None
          - "ocr":   list[str] or None

        Returns:
          {
            "semantic_text":  (B, proj_dim),
            "semantic_image": (B, proj_dim),
            "semantic_gap":   (B, proj_dim)  # simple difference signal
          }
        """
        titles = batch.get("title") or []
        ocrs   = batch.get("ocr") or []

        # Align lengths
        B = max(len(titles), len(ocrs))
        if len(titles) < B:
            titles = titles + [""] * (B - len(titles))
        if len(ocrs) < B:
            ocrs = ocrs + [""] * (B - len(ocrs))

        # Encode via CLIP text tower for both modalities (proxy for visuals in this stage)
        txt_feat = self.encode_text(titles)              # (B, 512)
        img_feat = self.encode_image_like(ocrs)          # (B, 512)

        # Project to fusion dim
        txt_proj = self.text_proj(txt_feat)              # (B, D)
        img_proj = self.vision_proj(img_feat)            # (B, D)

        # Simple semantic discrepancy signal (directional)
        gap = l2n(txt_proj - img_proj)

        return {
            "semantic_text":  l2n(txt_proj),
            "semantic_image": l2n(img_proj),
            "semantic_gap":   gap,
        }
