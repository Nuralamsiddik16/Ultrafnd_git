import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import torch
from torch.utils.data import DataLoader
from src.data_pipeline.fakesv_dataset import FakeSVRawDataset, build_gnn_cache_from_raw_dataset
from src.training.metrics.forensic_metrics import compute_classification_metrics, pretty_print
from src.training.forensic_trainer import ForensicTrainer, TrainConfig
from src.models.fusion.cross_modal_transformer import CrossModalTransformer
from src.models.fusion.deep_truth_classifier import DeepTruthClassifier


def test_model_initialization():
    """
    Ensure the model initializes without errors.
    """
    # Dummy config
    config = TrainConfig(
        data_root="/Volumes/SR_disk/FakeSV", 
        ocr_phrase_pkl="/Users/siddikmdnuralam/Ultrafnd_try/fakesv/preprocess_ocr/ocr_phrase_fea.pkl", 
        out_dir="outputs_v2", epochs=2, batch_size=8, lr=2e-4, weight_decay=1e-4
    )
    
    trainer = ForensicTrainer(config)
    print("Model initialized successfully!")
    
    # Perform a forward pass (dummy batch)
    batch = next(iter(trainer.train_loader))
    outputs = trainer._forward_batch(batch, "train")
    print("Forward pass successful:", outputs.keys())


def test_metrics():
    """
    Run a test for computing metrics without errors.
    """
    # Dummy true and predicted values
    y_true = [0, 1, 1, 0]
    y_pred = [0.1, 0.9, 0.8, 0.2]  # Probabilities, not labels
    
    metrics = compute_classification_metrics(y_true, y_pred, threshold=0.5)
    pretty_print("test", metrics)


def test_data_loading():
    """
    Ensure data loading is working properly.
    """
    # Check if the dataset loads
    data_root = "/Volumes/SR_disk/FakeSV"
    ocr_phrase_pkl = "/Users/siddikmdnuralam/Ultrafnd_try/fakesv/preprocess_ocr/ocr_phrase_fea.pkl"
    raw = FakeSVRawDataset(data_root)
    cache = build_gnn_cache_from_raw_dataset(raw, ocr_phrase_pkl=ocr_phrase_pkl)
    
    train_loader, val_loader, test_loader = trainer._build_dataloaders()
    
    print(f"Train loader: {len(train_loader.dataset)} samples")
    print(f"Validation loader: {len(val_loader.dataset)} samples")
    print(f"Test loader: {len(test_loader.dataset)} samples")


if __name__ == "__main__":
    # Run all tests
    test_model_initialization()
    test_metrics()
    test_data_loading()
    print("Sanity check complete!")
