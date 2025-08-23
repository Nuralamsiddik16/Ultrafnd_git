#!/usr/bin/env python3
# run_train_eval.py
"""
End-to-end training & evaluation entrypoint for Ultrafnd_try v2.

Examples (M1 paths you used):
  python run_train_eval.py \
    --data_root /Volumes/SR_disk/FakeSV \
    --ocr_phrase_pkl /Users/siddikmdnuralam/Ultrafnd_try/fakesv/preprocess_ocr/ocr_phrase_fea.pkl \
    --out_dir outputs_v2 --epochs 12 --batch_size 16

Eval only (load best.ckpt and test):
  python run_train_eval.py --eval_only --out_dir outputs_v2
"""

import os
import argparse
from pathlib import Path
import torch

# Prefer MPS on Apple Silicon, but keep CPU safety
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Local imports
from src.training.forensic_trainer import TrainConfig, ForensicTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Ultrafnd_try v2 — train/test")
    p.add_argument("--data_root", type=str, default="/Volumes/SR_disk/FakeSV",
                   help="Root with videos/, video_comment/, data_complete.json")
    p.add_argument("--ocr_phrase_pkl", type=str,
                   default="/Users/siddikmdnuralam/Ultrafnd_try/fakesv/preprocess_ocr/ocr_phrase_fea.pkl",
                   help="OCR phrase cache produced in Step 0 (optional; trainer falls back if missing).")
    p.add_argument("--out_dir", type=str, default="outputs_v2", help="Where to save checkpoints & logs")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gnn_dim", type=int, default=128)
    p.add_argument("--gnn_overlap_thresh", type=float, default=0.12,
                   help="OCR Jaccard threshold for graph edges")
    p.add_argument("--focal_gamma", type=float, default=0.0,
                   help="Use focal loss with the given gamma; 0 disables focal loss")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="Force CPU even if MPS is available")
    p.add_argument("--no_gnn", action="store_true", help="Disable GNN features")
    p.add_argument("--eval_only", action="store_true", help="Skip training; load best and test")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    data_root = Path(args.data_root).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    ocr_pkl = Path(args.ocr_phrase_pkl).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Torch niceties
    if not args.cpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed)

    print("==== Ultrafnd_try v2 ====")
    print(f"Device:          {device}")
    print(f"Data root:       {data_root}")
    print(f"OCR phrase pkl:  {ocr_pkl}  (exists: {ocr_pkl.exists()})")
    print(f"Output dir:      {out_dir}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Use GNN:         {not args.no_gnn}")
    print(f"GNN overlap thr: {args.gnn_overlap_thresh}")
    print(f"Focal gamma:     {args.focal_gamma}")
    print("==========================")

    cfg = TrainConfig(
        data_root=str(data_root),
        ocr_phrase_pkl=str(ocr_pkl) if ocr_pkl.exists() else None,
        out_dir=str(out_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gnn_dim=args.gnn_dim,
        gnn_overlap_thresh=args.gnn_overlap_thresh,
        focal_gamma=args.focal_gamma,
        seed=args.seed,
        use_mps=(device.type == "mps"),
        use_gnn=(not args.no_gnn),
        save_best=True,
    )

    trainer = ForensicTrainer(cfg)

    if not args.eval_only:
        print("\n>>> Training...")
        trainer.fit()

    print("\n>>> Testing best checkpoint...")
    results = trainer.test()

    print("\n==== Final Results ====")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Acc : {results['test_acc']:.4f}")
    print(f"Test AUC : {results['test_auc']:.4f}")
    # If trainer returns extended keys, show them:
    for k in ("test_precision", "test_recall", "test_f1", "test_cmcs", "test_dfdr"):
        if k in results:
            print(f"{k.replace('test_', 'Test ').title()}: {results[k]:.4f}")


if __name__ == "__main__":
    main()
