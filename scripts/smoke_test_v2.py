#!/usr/bin/env python3
import os, sys, unittest
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from src.models.gnn import GNNModel
    from src.models.fusion.cross_modal_transformer import CrossModalTransformer
    from src.models.fusion.deep_truth_classifier import DeepTruthClassifier
    from src.training.forensic_trainer import TrainConfig, ForensicTrainer
    IMPORTS_OK = True
except Exception as e:
    print(f"❌ Import error: {e}")
    IMPORTS_OK = False

class SmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not IMPORTS_OK:
            raise unittest.SkipTest("Imports failed; aborting tests.")
        cls.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        print("✅ All required imports successful")
        print("🚀 Starting Ultrafnd_try Smoke Test")
        print("=" * 50)
        print(f"\nRunning tests on device: {cls.device.type}")

    def test_model_initialization(self):
        try:
            # GNNModel takes in_dim (NOT input_dim)
            gnn_model = GNNModel(in_dim=416, hid=256, out_dim=128, dropout=0.1).to(self.device)
            self.assertTrue(hasattr(gnn_model, "forward"))

            fusion = CrossModalTransformer("configs/model_configs/fusion.yaml")
            clf = DeepTruthClassifier("configs/model_configs/classifier.yaml")

            B = 2
            import torch
            T = torch.randn(B, 768); A = torch.randn(B, 128)
            V = torch.randn(B, 512); U = torch.randn(B, 256)
            G = torch.randn(B, 128)

            fused_out = fusion({
                "text_features": T,
                "audio_features": A,
                "visual_features": V,
                "temporal_features": U,
                "gnn_feat": G
            })
            self.assertEqual(tuple(fused_out["fused"].shape), (B, 512))
            res = clf(fused_out["fused"], torch.rand(B, 2))
            self.assertEqual(tuple(res["probs"].shape), (B, 2))
        except Exception as e:
            self.fail(f"Model initialization failed: {e}")

    def test_trainer_initialization(self):
        try:
            cfg = TrainConfig(
                data_root="/Volumes/SR_disk/FakeSV",
                ocr_phrase_pkl="/Users/siddikmdnuralam/Ultrafnd_try/fakesv/preprocess_ocr/ocr_phrase_fea.pkl",
                out_dir="outputs_smoke",
                batch_size=4,
                epochs=0,  # skip training; just build and eval once
                lr=2e-4,
                weight_decay=1e-4,
                gnn_dim=128,
                gnn_overlap_thresh=0.12,
                seed=42,
                use_mps=torch.backends.mps.is_available(),
                use_gnn=True,
                save_best=False,
            )
            trainer = ForensicTrainer(cfg)
            res = trainer.test()
            for k in ("test_loss", "test_acc", "test_auc"):
                self.assertIn(k, res)
        except Exception as e:
            self.fail(f"Trainer initialization failed: {e}")

if __name__ == "__main__":
    if not IMPORTS_OK:
        sys.exit(1)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SmokeTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All smoke tests passed.")
        sys.exit(0)
    else:
        print("❌ Some smoke tests failed. Please check the output above.")
        sys.exit(2)
