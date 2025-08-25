"""Minimal training script using the new dataset and model.

This is intended as a starting point; real experiments should expand on
this to include proper logging, checkpointing and hyper-parameter tuning.
"""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from data_pipeline.dataloader import FakeNewsDataset, get_video_transform
from models.model import get_model
from utils.metrics import MetricsCalculator


@dataclass
class TrainConfig:
    data_root: str
    model_name: str = "hierarchical_fusion"
    text_model_name: str = "bert-base-uncased"
    num_labels: int = 2
    use_text: bool = True
    use_visual: bool = True
    use_meta: bool = True
    batch_size: int = 2
    epochs: int = 1
    lr: float = 1e-4


def train_one_epoch(model, loader, optim, device, metrics, cfg):
    model.train()
    for text_inputs, frames, meta, labels in loader:
        text_inputs = {k: v.squeeze(1).to(device) for k, v in text_inputs.items()}
        frames = frames.to(device)
        meta = meta.to(device)
        labels = labels.to(device)

        if not cfg.use_text:
            text_inputs = {k: torch.zeros_like(v) for k, v in text_inputs.items()}
        if not cfg.use_visual:
            frames = torch.zeros_like(frames)
        if not cfg.use_meta:
            meta = torch.zeros_like(meta)

        optim.zero_grad()
        outputs = model(text_inputs, frames, meta)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optim.step()

        metrics.update(outputs.detach(), labels.detach())


@torch.no_grad()
def evaluate(model, loader, device, metrics, cfg):
    model.eval()
    for text_inputs, frames, meta, labels in loader:
        text_inputs = {k: v.squeeze(1).to(device) for k, v in text_inputs.items()}
        frames = frames.to(device)
        meta = meta.to(device)
        labels = labels.to(device)

        if not cfg.use_text:
            text_inputs = {k: torch.zeros_like(v) for k, v in text_inputs.items()}
        if not cfg.use_visual:
            frames = torch.zeros_like(frames)
        if not cfg.use_meta:
            meta = torch.zeros_like(meta)

        outputs = model(text_inputs, frames, meta)
        metrics.update(outputs, labels)


def main(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_video_transform()
    train_ds = FakeNewsDataset(cfg.data_root, split="train", transform=transform, text_model_name=cfg.text_model_name)
    val_ds = FakeNewsDataset(cfg.data_root, split="val", transform=transform, text_model_name=cfg.text_model_name, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = get_model(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    train_metrics = MetricsCalculator()
    val_metrics = MetricsCalculator()

    for epoch in range(cfg.epochs):
        train_metrics.reset()
        train_one_epoch(model, train_loader, optim, device, train_metrics, cfg)
        train_results = train_metrics.compute()

        val_metrics.reset()
        evaluate(model, val_loader, device, val_metrics, cfg)
        val_results = val_metrics.compute()

        print(f"Epoch {epoch}: Train F1={train_results['f1_score']:.4f} | Val F1={val_results['f1_score']:.4f}")
        last_f1 = val_results["f1_score"]


    return last_f1

if __name__ == "__main__":
    # Example usage; adjust paths and hyper-parameters as needed.
    cfg = TrainConfig(data_root="./fakesv")
    main(cfg)
