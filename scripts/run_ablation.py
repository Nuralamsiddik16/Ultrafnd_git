"""Utility to run simple ablation studies on modality usage."""

import dataclasses

from src.train import TrainConfig, main as train_main


def run_ablation_studies(base_cfg: TrainConfig):
    results = {}

    # Text only
    cfg = dataclasses.replace(base_cfg)
    cfg.use_visual = False
    cfg.use_meta = False
    results["text_only"] = train_main(cfg)

    # Visual only
    cfg = dataclasses.replace(base_cfg)
    cfg.use_visual = True
    cfg.use_text = False
    cfg.use_meta = False
    results["visual_only"] = train_main(cfg)

    # Metadata only
    cfg = dataclasses.replace(base_cfg)
    cfg.use_visual = False
    cfg.use_text = False
    cfg.use_meta = True
    results["meta_only"] = train_main(cfg)

    # Full model
    cfg = dataclasses.replace(base_cfg)
    cfg.use_visual = True
    cfg.use_text = True
    cfg.use_meta = True
    results["full_model"] = train_main(cfg)

    return results
