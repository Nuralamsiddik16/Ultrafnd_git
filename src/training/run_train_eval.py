#main entry point before introducing the GNN

import os
# Set MPS fallback for 3D convolutions and enable M1 GPU optimization
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Optimize memory usage

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
from torch.nn.utils.rnn import pad_sequence
import random
import torch.nn.functional as F
from torchvision import transforms

# Add src to path - fix for M1 GPU training
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root))

try:
    from src.core_blocks.audio_blocks import SpectralForensics
    from src.core_blocks.visual_blocks import OpticalFlow3DCNN
    from src.core_blocks.text_blocks import BERTContextEncoder
    from src.core_blocks.temporal_blocks import TemporalSyncNet
    from src.models.fusion.cross_modal_transformer import CrossModalTransformer
    from src.models.fusion.deep_truth_classifier import DeepTruthClassifier
    from src.models.affective_forensics import AffectiveForensics
    from src.models.chronos_guard import ChronosGuard
    from src.models.semantic_forgery import SemanticForgeryDetector
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔄 Trying alternative import paths...")
    
    # Try alternative import paths
    try:
        from core_blocks.audio_blocks import SpectralForensics
        from core_blocks.visual_blocks import OpticalFlow3DCNN
        from core_blocks.text_blocks import BERTContextEncoder
        from core_blocks.temporal_blocks import TemporalSyncNet
        from models.fusion.cross_modal_transformer import CrossModalTransformer
        from models.fusion.deep_truth_classifier import DeepTruthClassifier
        from models.affective_forensics import AffectiveForensics
        from models.chronos_guard import ChronosGuard
        from models.semantic_forgery import SemanticForgeryDetector
        print("✅ Imports successful with alternative paths")
    except ImportError as e2:
        print(f"❌ Alternative import also failed: {e2}")
        print("🔄 Using dummy models for testing...")
        # Create dummy models for testing
        class SpectralForensics(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(80000, 128)
            def get_feature_vector(self, x):
                return self.fc(x[:80000].float())
            def forward(self, x):
                return self.get_feature_vector(x)
        
        class OpticalFlow3DCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(30*256*256*3, 512)
            def forward(self, x):
                return self.fc(x.view(x.size(0), -1)), torch.randn(x.size(0), 3)
        
        class BERTContextEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(500, 768)
            def forward(self, x):
                if isinstance(x, list):
                    x = [str(item)[:500] for item in x]
                    x = torch.tensor([[ord(c) for c in s] + [0]*(500-len(s)) for s in x]).float()
                return self.fc(x)
        
        class TemporalSyncNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(256*2, 256)
            def forward(self, audio, video):
                combined = torch.cat([audio.mean(1), video.mean(1)], dim=-1)
                return self.fc(combined)
        
        class CrossModalTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(768+128+512+256, 2)  # Changed to 2 classes
            def forward(self, features):
                combined = torch.cat([
                    features['text_features'],
                    features['audio_features'],
                    features['visual_features'],
                    features['temporal_features']
                ], dim=-1)
                return {'logits': self.fc(combined)}
        
        class DeepTruthClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(768+128+512+256, 2)  # Changed to 2 classes
            def forward(self, features):
                combined = torch.cat([
                    features['text_features'],
                    features['audio_features'],
                    features['visual_features'],
                    features['temporal_features']
                ], dim=-1)
                return {'logits': self.fc(combined)}
        
        class AffectiveForensics(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(128, 128)
            def forward(self, text, audio, video):
                return {'audio_features': self.fc(audio.mean(0).unsqueeze(0))}
        
        class ChronosGuard(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(30*256*256*3, 256)
            def forward(self, video):
                return self.fc(video.view(video.size(0), -1))
        
        class SemanticForgeryDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(768+512, 256)
            def process_features(self, text, visual):
                combined = torch.cat([text, visual], dim=-1)
                return self.fc(combined)

def get_m1_optimized_device():
    """
    Get the best available device for M1 Mac with automatic fallback.
    For reliability, we'll use CPU by default and only try MPS for simple operations.
    """
    # For complex models like BERT, Wav2Vec2, etc., CPU is more reliable on M1
    print("🔧 Using CPU for maximum compatibility and reliability")
    print("💡 M1 GPU can be enabled for simple operations, but complex models work better on CPU")
    return torch.device("cpu")

def get_optimized_config(device, debug_mode=False):
    """
    Get optimized configuration based on device and mode.
    """
    if debug_mode:
        # Debug mode: minimal resources
        return {
            'max_samples': 5,
            'batch_size': 1,
            'val_batch_size': 1,
            'max_frames': 8,
            'frame_size': 128,
            'ensemble_size': 1,
            'max_epochs': 10
        }
    elif device.type == "mps":
        # M1 GPU mode: very conservative for compatibility
        return {
            'max_samples': 10,
            'batch_size': 1,
            'val_batch_size': 1,
            'max_frames': 10,
            'frame_size': 128,
            'ensemble_size': 1,
            'max_epochs': 20
        }
    else:
        # CPU mode: can handle more
        return {
            'max_samples': 40,
            'batch_size': 2,
            'val_batch_size': 2,
            'max_frames': 30,
            'frame_size': 256,
            'ensemble_size': 2,
            'max_epochs': 100
        }

def m1_memory_cleanup():
    """
    Clean up M1 GPU memory when needed.
    """
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except:
            pass

# ---- Raw Data FakeSV Dataset Class ----
class FakeSVRawDataset(Dataset):
    def __init__(self, data_dir: str, max_samples: int = None):
        """
        Initialize FakeSV dataset with raw data for full pipeline processing.
        
        Args:
            data_dir: Directory containing raw data (videos, comments, metadata)
            max_samples: Maximum number of samples to load (for testing)
        """
        self.data_dir = Path(data_dir)
        self.videos_dir = self.data_dir / 'videos'
        self.comments_dir = self.data_dir / 'video_comment'
        self.metadata_file = self.data_dir / 'data_complete.json'
        
        # Load metadata
        self.samples = self._load_metadata()
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
        
        # Initialize video/audio processors
        self._initialize_processors()
    
    def _load_metadata(self):
        """Load and validate metadata."""
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        samples = []
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        video_id = item.get('video_id')
                        
                        # Check if video file exists
                        video_file = self.videos_dir / f"{video_id}.mp4"
                        if video_file.exists():
                            samples.append({
                                'video_id': video_id,
                                'video_path': video_file,
                                'comment_dir': self.comments_dir / video_id,
                                'metadata': item
                            })
                    except json.JSONDecodeError:
                        continue
        
        return samples
    
    def _initialize_processors(self):
        """Initialize video and audio processors."""
        import cv2
        import torchaudio
        
        self.video_processor = cv2
        self.audio_processor = torchaudio
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load raw video frames
        video_frames = self._load_video_frames(sample['video_path'])
        
        # Load raw audio
        audio_waveform = self._load_audio(sample['video_path'])
        
        # Load raw text
        text_data = self._load_text(sample['comment_dir'], sample['metadata'])
        
        # Get label
        annotation = sample['metadata'].get('annotation', '真')
        label = self._annotation_to_label(annotation)
        
        return {
            'video_id': sample['video_id'],
            'video_frames': video_frames,  # Raw video frames
            'audio_waveform': audio_waveform,  # Raw audio waveform
            'text_data': text_data,  # Raw text data
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def _load_video_frames(self, video_path: Path) -> torch.Tensor:
        """Load raw video frames."""
        import cv2
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            frames = []
            frame_count = 0
            max_frames = 30  # Limit frames for memory efficiency
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB and normalize to [0, 1]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
                # Resize to standard size (256x256)
                frame_tensor = torch.nn.functional.interpolate(
                    frame_tensor.permute(2, 0, 1).unsqueeze(0), 
                    size=(256, 256), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0).permute(1, 2, 0)
                # Ensure shape is [256, 256, 3]
                if frame_tensor.shape != (256, 256, 3):
                    print(f"[DEBUG] Frame shape before correction: {frame_tensor.shape}")
                    frame_tensor = frame_tensor[:256, :256, :3]
                frames.append(frame_tensor)
                frame_count += 1
            
            cap.release()
            
            if not frames:
                # Return dummy frames if video loading fails
                dummy_frame = torch.zeros(256, 256, 3)
                frames = [dummy_frame] * max_frames
            
            # Ensure we have exactly max_frames frames
            if len(frames) < max_frames:
                # Pad with the last frame or dummy frames
                last_frame = frames[-1] if frames else torch.zeros(256, 256, 3)
                while len(frames) < max_frames:
                    frames.append(last_frame)
            elif len(frames) > max_frames:
                # Truncate to max_frames
                frames = frames[:max_frames]
            
            # Debug print for all frame shapes
            for i, f in enumerate(frames):
                if f.shape != (256, 256, 3):
                    print(f"[DEBUG] Frame {i} shape in stack: {f.shape}")
            # Stack frames: [frames, height, width, channels]
            video_tensor = torch.stack(frames)
            return video_tensor
            
        except Exception as e:
            print(f"Error loading video from {video_path}: {e}")
            # Return dummy frames if video loading fails
            dummy_frame = torch.zeros(256, 256, 3)
            frames = [dummy_frame] * 30
            return torch.stack(frames)
    
    def _load_audio(self, video_path: Path) -> torch.Tensor:
        """Load raw audio waveform with robust error handling."""
        import torchaudio
        import subprocess
        import tempfile
        
        # First try direct torchaudio loading
        try:
            # Check if file exists and is readable
            if not video_path.exists() or video_path.stat().st_size == 0:
                raise FileNotFoundError(f"Video file not found or empty: {video_path}")
            
            # Extract audio from video
            waveform, sample_rate = torchaudio.load(str(video_path))
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Limit duration to 10 seconds for memory efficiency
            max_samples = 16000 * 10  # 10 seconds at 16kHz
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            
            # Ensure we have at least 5 seconds of audio
            min_samples = 16000 * 5  # 5 seconds at 16kHz
            if waveform.shape[1] < min_samples:
                # Pad with zeros if too short
                padding = min_samples - waveform.shape[1]
                waveform = torch.cat([waveform, torch.zeros(1, padding)], dim=1)
            
            return waveform.squeeze(0)  # Remove channel dimension
            
        except Exception as e:
            # Try alternative audio extraction using ffmpeg if available
            try:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    # Use ffmpeg to extract audio
                    cmd = [
                        'ffmpeg', '-i', str(video_path), '-ac', '1', '-ar', '16000',
                        '-t', '10', '-y', tmp_file.name, '-loglevel', 'quiet'
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    # Load the extracted audio
                    waveform, _ = torchaudio.load(tmp_file.name)
                    
                    # Clean up temp file
                    Path(tmp_file.name).unlink(missing_ok=True)
                    
                    # Ensure proper length
                    min_samples = 16000 * 5
                    if waveform.shape[1] < min_samples:
                        padding = min_samples - waveform.shape[1]
                        waveform = torch.cat([waveform, torch.zeros(1, padding)], dim=1)
                    
                    return waveform.squeeze(0)
                    
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass  # ffmpeg not available or failed
            
            # Final fallback: return dummy audio
            print(f"⚠️  Using dummy audio for {video_path.name} (original error: {e})")
            return torch.zeros(16000 * 5)  # 5 seconds of silence
    
    def _load_text(self, comment_dir: Path, metadata: dict) -> dict:
        """Load raw text data with robust error handling."""
        text_data = {
            'title': metadata.get('title', ''),
            'description': metadata.get('description', ''),
            'comments': []
        }
        
        # Load comments if directory exists
        if comment_dir.exists():
            try:
                for comment_file in comment_dir.glob('*.json'):
                    # Skip macOS metadata files that start with ._
                    if comment_file.name.startswith('._'):
                        continue
                    
                    # Skip empty files
                    if comment_file.stat().st_size == 0:
                        continue
                    
                    try:
                        with open(comment_file, 'r', encoding='utf-8', errors='ignore') as f:
                            # Read content and check if it's valid
                            content = f.read().strip()
                            if not content:
                                continue
                            
                            # Try to parse JSON
                            comment_data = json.loads(content)
                            
                            # Handle both dict and list formats
                            if isinstance(comment_data, dict):
                                comment_text = comment_data.get('content', '')
                                if comment_text and isinstance(comment_text, str):
                                    text_data['comments'].append(comment_text)
                            elif isinstance(comment_data, list):
                                for comment in comment_data:
                                    if isinstance(comment, dict):
                                        comment_text = comment.get('content', '')
                                        if comment_text and isinstance(comment_text, str):
                                            text_data['comments'].append(comment_text)
                                    elif isinstance(comment, str) and comment.strip():
                                        text_data['comments'].append(comment.strip())
                                        
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        # Silently skip corrupted files instead of printing errors
                        continue
                    except Exception as e:
                        # Skip any other file-related errors
                        continue
                        
            except Exception as e:
                # Silently handle directory access errors
                pass
        
        # Ensure we have some text content even if comments fail to load
        if not text_data['title'] and not text_data['description'] and not text_data['comments']:
            text_data['title'] = 'Default video title'  # Fallback content
        
        return text_data
    
    def _annotation_to_label(self, annotation: str) -> int:
        """Convert annotation string to numeric label."""
        label_mapping = {
            '真': 0,      # Real
            '假': 1,      # Fake
            '辟谣': 1,    # Debunked (map to fake)
            'real': 0,
            'fake': 1,
            'debunked': 1  # Map to fake
        }
        return label_mapping.get(annotation, 0)

def multimodal_collate_fn(batch):
    """
    Custom collate function to pad/truncate video, audio, and handle text and label batching.
    """
    # Video: [num_frames, H, W, C] -> pad/truncate to 30 frames
    max_frames = 30
    video_batch = []
    for item in batch:
        frames = item['video_frames']
        if frames.shape[0] < max_frames:
            pad_frames = max_frames - frames.shape[0]
            last_frame = frames[-1] if frames.shape[0] > 0 else torch.zeros(256, 256, 3)
            pad = [last_frame for _ in range(pad_frames)]
            frames = torch.cat([frames, torch.stack(pad)], dim=0)
        elif frames.shape[0] > max_frames:
            frames = frames[:max_frames]
        # Debug print for each video
        print(f"[DEBUG] Video frames shape before stacking: {frames.shape}")
        video_batch.append(frames)
    # Debug print for all video shapes
    print(f"[DEBUG] All video shapes: {[v.shape for v in video_batch]}")
    video_batch = torch.stack(video_batch)  # [B, 30, 256, 256, 3]
    print(f"[DEBUG] Final video_batch shape after stacking: {video_batch.shape}")

    # Audio: pad/truncate to 5 seconds (80000 samples at 16kHz)
    max_audio_len = 16000 * 5
    audio_batch = []
    for item in batch:
        audio = item['audio_waveform']
        if audio.shape[0] < max_audio_len:
            pad = torch.zeros(max_audio_len - audio.shape[0])
            audio = torch.cat([audio, pad], dim=0)
        elif audio.shape[0] > max_audio_len:
            audio = audio[:max_audio_len]
        audio_batch.append(audio)
    audio_batch = torch.stack(audio_batch)  # [B, 80000]

    # Text: keep as list of dicts for further processing
    text_batch = [item['text_data'] for item in batch]

    # Labels: stack
    label_batch = torch.stack([item['label'] for item in batch])

    # Video IDs: keep as list
    video_ids = [item['video_id'] for item in batch]

    return {
        'video_frames': video_batch,
        'audio_waveform': audio_batch,
        'text_data': text_batch,
        'label': label_batch,
        'video_id': video_ids
    }

# ---- Training and Evaluation ----
def train_and_evaluate(data_dir: str = None, max_samples: int = 100, debug_mode: bool = False):
    """
    Train and evaluate the UltraFND3 pipeline with automatic M1/CPU optimization.
    
    Args:
        data_dir: Directory containing raw data (videos, comments, metadata)
        max_samples: Maximum number of samples to use (for testing)
        debug_mode: Use minimal resources for quick testing
    """
    # Get optimized device and configuration
    device = get_m1_optimized_device()
    config = get_optimized_config(device, debug_mode)
    
    print(f"🚀 Using device: {device}")
    print(f"⚙️ Configuration: {config}")
    
    # Set default dtype for optimization
    torch.set_default_dtype(torch.float32)
    
    # CPU optimizations for M1 Mac
    if device.type == "cpu":
        print("🔧 Applying CPU optimizations for M1 Mac...")
        # Set number of threads for optimal CPU performance
        torch.set_num_threads(min(8, torch.get_num_threads()))
        print(f"🧵 Using {torch.get_num_threads()} CPU threads")
    
    # Use provided data directory or default
    if data_dir is None:
        # Try to find raw data
        possible_paths = [
            "/Volumes/SR_disk/FakeSV",
            "data",
            "dataset"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                data_dir = path
                break
        
        if data_dir is None:
            print("No raw data found. Using dummy data for demonstration.")
            # Fall back to dummy data
            return _train_with_dummy_data(device)
    
    print(f"Loading raw data from: {data_dir}")
    
    # Load dataset with automatic fallback
    try:
        print(f"Loading FakeSV dataset from: {data_dir}")
        dataset = FakeSVRawDataset(data_dir, max_samples=config['max_samples'])
        
        print(f"Initial dataset size: {len(dataset)}")
        
        # Ensure we have enough samples
        if len(dataset) < config['max_samples']:
            print(f"Warning: Only {len(dataset)} samples available, using all available samples")
            dataset = FakeSVRawDataset(data_dir, max_samples=len(dataset))
        
        # Optimized dataset split
        total_samples = len(dataset)
        
        # Calculate split sizes ensuring they sum to total_samples
        if total_samples >= 10:
            train_size = int(0.75 * total_samples)
            val_size = total_samples - train_size
        else:
            # For smaller datasets, ensure at least 1 validation sample
            val_size = max(1, int(0.25 * total_samples))
            train_size = total_samples - val_size
        
        print(f"📊 Dataset split: {train_size} train, {val_size} validation (total: {total_samples})")
        
        # Ensure we have enough samples for both train and validation
        if train_size < 2 or val_size < 1:
            print("❌ Error: Not enough samples for proper training and validation split")
            print(f"Available: {total_samples}, Required: at least 3 (2 train + 1 validation)")
            return _train_with_dummy_data(device, config)
        
        # Create the split
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducible splits
        )
        
        # Apply data augmentation to training set
        train_dataset = AugmentedFakeSVDataset(train_dataset, augment=True)
        val_dataset = AugmentedFakeSVDataset(val_dataset, augment=False)  # No augmentation for validation
        
        # Use optimized batch sizes from config
        batch_size = min(config['batch_size'], train_size)
        val_batch_size = min(config['val_batch_size'], val_size)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=multimodal_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, collate_fn=multimodal_collate_fn)
        
        print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")
        print(f"Total samples loaded: {len(dataset)}")
        print(f"Train samples: {train_size}, Validation samples: {val_size}")
        print(f"Batch sizes: Train={batch_size}, Validation={val_batch_size}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to dummy data...")
        return _train_with_dummy_data(device, config)
    
    # Initialize complete multimodal pipeline with automatic fallback
    print("\nInitializing complete multimodal pipeline...")
    
    try:
        # Core feature extraction blocks
        audio_model = SpectralForensics().to(device)
        visual_model = OpticalFlow3DCNN().to(device)
        text_model = BERTContextEncoder().to(device)
        temporal_model = TemporalSyncNet().to(device)
        
        # Specialized forensic models
        affective_model = AffectiveForensics().to(device)
        chronos_model = ChronosGuard().to(device)
        semantic_model = SemanticForgeryDetector().to(device)
        
    except RuntimeError as e:
        if "out of memory" in str(e) and device.type == "mps":
            print(f"❌ M1 GPU out of memory: {e}")
            print("🔄 Falling back to CPU...")
            device = torch.device("cpu")
            config = get_optimized_config(device, debug_mode)
            print(f"⚙️ New configuration: {config}")
            
            # Re-initialize models on CPU
            audio_model = SpectralForensics().to(device)
            visual_model = OpticalFlow3DCNN().to(device)
            text_model = BERTContextEncoder().to(device)
            temporal_model = TemporalSyncNet().to(device)
            affective_model = AffectiveForensics().to(device)
            chronos_model = ChronosGuard().to(device)
            semantic_model = SemanticForgeryDetector().to(device)
        else:
            raise e
    
    # Create ensemble of fusion models with optimized size
    try:
        fusion_models = create_ensemble_models(device, config['ensemble_size'])
        
        # Ensure all models are on the correct device
        print(f"Moving all models to device: {device}")
        for model in fusion_models:
            model.to(device)
            
    except RuntimeError as e:
        if "out of memory" in str(e) and device.type == "mps":
            print(f"❌ M1 GPU out of memory during ensemble creation: {e}")
            print("🔄 Falling back to CPU...")
            device = torch.device("cpu")
            config = get_optimized_config(device, debug_mode)
            print(f"⚙️ New configuration: {config}")
            
            # Re-create ensemble on CPU
            fusion_models = create_ensemble_models(device, config['ensemble_size'])
            for model in fusion_models:
                model.to(device)
        else:
            raise e
    
    # Optimized loss function with focal loss for class imbalance
    criterion = FocalLoss(alpha=1, gamma=2)
    
    # M1 GPU optimized optimizers and schedulers
    optimizers = []
    schedulers = []
    for model in fusion_models:
        # M1 GPU optimized learning rate and parameters
        if device.type == "mps":
            optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999))
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
    
    # M1 GPU optimized training loop
    print("\n--- Training with M1 GPU Optimized Techniques ---")
    
    best_accuracy = 0.0
    patience = 20 if device.type == "mps" else 15  # More patience for M1 GPU
    patience_counter = 0
    
    # Optimized epochs from config
    max_epochs = config['max_epochs']
    for epoch in range(max_epochs):
        print(f"\n[INFO] Starting epoch {epoch+1}/{max_epochs}...")
        total_loss = 0
        correct = 0
        total = 0
        
        # Train each model in ensemble
        for model_idx, (fusion_model, optimizer) in enumerate(zip(fusion_models, optimizers)):
            print(f"[INFO] Training model {model_idx+1}/{len(fusion_models)} in ensemble...")
            fusion_model.train()
            
            for batch_idx, batch in enumerate(train_loader):
                print(f"[INFO]   Batch {batch_idx+1}/{len(train_loader)}...")
                optimizer.zero_grad()
                
                # Apply mixup augmentation
                use_mixup = random.random() < 0.5  # 50% chance of mixup
                
                # 1. Core feature extraction
                # Process audio batch one by one
                audio_features_list = []
                for audio_sample in batch['audio_waveform']:
                    audio_features_list.append(audio_model.get_feature_vector(audio_sample.to(device)))
                audio_features = torch.stack(audio_features_list)  # (batch, 128)
                visual_features, visual_logits = visual_model(batch['video_frames'].to(device))  # (batch, 512)
                # Process text data for BERT
                text_strings = []
                for text_data in batch['text_data']:
                    combined_text = f"{text_data['title']} {text_data['description']}"
                    if text_data['comments']:
                        combined_text += " " + " ".join(text_data['comments'][:5])  # Limit to 5 comments
                    text_strings.append(combined_text[:500])  # Limit length
                text_features = text_model(text_strings)  # (batch, 768)
                # For temporal model, create temporal features from audio and video
                audio_temporal = audio_features.unsqueeze(1).repeat(1, 30, 1)  # [B, 30, 128]
                video_temporal = visual_features.unsqueeze(1).repeat(1, 30, 1)  # [B, 30, 512]
                max_features = max(audio_temporal.shape[-1], video_temporal.shape[-1])
                if audio_temporal.shape[-1] < max_features:
                    padding = torch.zeros(audio_temporal.shape[0], audio_temporal.shape[1], max_features - audio_temporal.shape[-1], device=audio_temporal.device, dtype=audio_temporal.dtype)
                    audio_temporal = torch.cat([audio_temporal, padding], dim=-1)
                if video_temporal.shape[-1] < max_features:
                    padding = torch.zeros(video_temporal.shape[0], video_temporal.shape[1], max_features - video_temporal.shape[-1], device=video_temporal.device, dtype=video_temporal.dtype)
                    video_temporal = torch.cat([video_temporal, padding], dim=-1)
                audio_temporal = audio_temporal.to(torch.float32)
                video_temporal = video_temporal.to(torch.float32)
                audio_proj = nn.Linear(audio_temporal.shape[-1], 256, dtype=torch.float32).to(device)
                video_proj = nn.Linear(video_temporal.shape[-1], 256, dtype=torch.float32).to(device)
                audio_temporal = audio_proj(audio_temporal)  # [B, 30, 256]
                video_temporal = video_proj(video_temporal)  # [B, 30, 256]
                target_steps = 30
                if audio_temporal.shape[1] > target_steps:
                    audio_temporal = audio_temporal[:, :target_steps, :]
                elif audio_temporal.shape[1] < target_steps:
                    pad_size = target_steps - audio_temporal.shape[1]
                    audio_temporal = torch.cat([
                        audio_temporal,
                        torch.zeros(audio_temporal.shape[0], pad_size, audio_temporal.shape[2], device=audio_temporal.device, dtype=audio_temporal.dtype)
                    ], dim=1)
                if video_temporal.shape[1] > target_steps:
                    video_temporal = video_temporal[:, :target_steps, :]
                elif video_temporal.shape[1] < target_steps:
                    pad_size = target_steps - video_temporal.shape[1]
                    video_temporal = torch.cat([
                        video_temporal,
                        torch.zeros(video_temporal.shape[0], pad_size, video_temporal.shape[2], device=video_temporal.device, dtype=video_temporal.dtype)
                    ], dim=1)
                temporal_features = temporal_model(audio_temporal, video_temporal)
                if isinstance(temporal_features, tuple):
                    temporal_features = temporal_features[0]
                if temporal_features.dim() == 3:
                    temporal_features = temporal_features.mean(dim=1)  # [batch, 256]
                # Debug print for feature shapes before fusion
                print(f"[FUSION DEBUG] text_features: {text_features.shape}, audio_features: {audio_features.shape}, visual_features: {visual_features.shape}, temporal_features: {temporal_features.shape}")

                # Mixup on features and labels if enabled
                if use_mixup:
                    text_features, y_a, y_b, lam = mixup_data(text_features, batch['label'].to(device))
                    audio_features, _, _, _ = mixup_data(audio_features, batch['label'].to(device))
                    visual_features, _, _, _ = mixup_data(visual_features, batch['label'].to(device))
                    temporal_features, _, _, _ = mixup_data(temporal_features, batch['label'].to(device))
                else:
                    y_a = batch['label'].to(device)
                    y_b = batch['label'].to(device)
                    lam = 1.0

                # 2. Specialized forensic analysis (do not use for main features)
                text_features_orig = text_features
                text_strings = []
                for text_data in batch['text_data']:
                    combined_text = f"{text_data['title']} {text_data['description']}"
                    if text_data['comments']:
                        combined_text += " " + " ".join(text_data['comments'][:5])
                    text_strings.append(combined_text[:500])
                raw_audio = batch['audio_waveform'].to(device)
                raw_video = batch['video_frames'].to(device)
                affective_evidence = affective_model(text_strings, raw_audio, raw_video)
                chronos_evidence = chronos_model(raw_video)
                semantic_evidence = semantic_model.process_features(text_features_orig, visual_features)
                # 3. Cross-modal fusion
                def get_tensor(x):
                    return x[0] if isinstance(x, tuple) else x
                text_features = get_tensor(text_features)
                audio_features = get_tensor(audio_features)
                visual_features = get_tensor(visual_features)
                temporal_features = get_tensor(temporal_features)
                fusion_out = fusion_model({
                    'text_features': text_features,  # (batch, 768)
                    'audio_features': audio_features,  # (batch, 128)
                    'visual_features': visual_features,  # (batch, 512)
                    'temporal_features': temporal_features  # (batch, 256)
                })
                # 4. Calculate loss with mixup if enabled
                final_logits = fusion_out['logits']
                if use_mixup:
                    loss = mixup_criterion(criterion, final_logits, y_a, y_b, lam)
                else:
                    loss = criterion(final_logits, batch['label'].to(device))
                print(f"[INFO]     Loss calculated: {loss.item():.4f}")
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                preds = torch.argmax(final_logits, dim=1)
                correct += (preds == batch['label'].to(device)).sum().item()
                total += batch['label'].size(0)
                
                # Print progress every 10 batches
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}, Model {model_idx+1}, Batch {batch_idx}: Loss = {loss.item():.4f}, Acc = {100*correct/total:.2f}%")
            print(f"[INFO] Finished training model {model_idx+1}/{len(fusion_models)} for epoch {epoch+1}.")
        
        # Update learning rate schedulers
        for scheduler in schedulers:
            scheduler.step()
        
        # Calculate average metrics
        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total
        print(f"[INFO] Finished epoch {epoch+1}. Avg Loss = {avg_loss:.4f}, Train Acc = {train_acc:.2%}")
        
        # Memory cleanup
        if device.type == "mps":
            m1_memory_cleanup()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "cpu":
            # CPU doesn't need explicit cleanup, but we can force garbage collection
            import gc
            gc.collect()
        
        # Evaluate on validation set every 5 epochs
        if (epoch + 1) % 5 == 0:
            val_acc = _evaluate_model(fusion_models, val_loader, device, audio_model, visual_model, text_model, temporal_model, affective_model, chronos_model, semantic_model)
            print(f"Validation Accuracy: {val_acc*100:.2f}%")
            
            # Check for early stopping
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                patience_counter = 0
                print(f"New best validation accuracy: {best_accuracy*100:.2f}%")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print("Early stopping triggered. Training completed.")
                break
    
    # Final evaluation on validation set
    val_acc = _evaluate_model(fusion_models, val_loader, device, audio_model, visual_model, text_model, temporal_model, affective_model, chronos_model, semantic_model)
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
    
    return {
        'accuracy': val_acc,
        'epochs': epoch + 1,
        'best_accuracy': best_accuracy,
        'patience_counter': patience_counter,
        'patience': patience
    }

def _evaluate_model(models, data_loader, device, audio_model, visual_model, text_model, temporal_model, affective_model, chronos_model, semantic_model):
    """Evaluate ensemble of models."""
    for model in models:
        model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            # Get ensemble prediction
            ensemble_pred = ensemble_predict_advanced(models, batch, device, audio_model, visual_model, text_model, temporal_model, affective_model, chronos_model, semantic_model)
            
            # Calculate accuracy
            preds = torch.argmax(ensemble_pred, dim=1)
            correct += (preds == batch['label'].to(device)).sum().item()
            total += batch['label'].size(0)
    
    accuracy = correct / total
    return accuracy

def ensemble_predict_advanced(models, batch, device, audio_model, visual_model, text_model, temporal_model, affective_model, chronos_model, semantic_model):
    """Advanced ensemble prediction with full pipeline processing."""
    predictions = []
    
    for model in models:
        # Process the batch through the full pipeline
        audio_features_list = []
        for audio_sample in batch['audio_waveform']:
            audio_features_list.append(audio_model.get_feature_vector(audio_sample.to(device)))
        audio_features = torch.stack(audio_features_list)
        
        visual_features, visual_logits = visual_model(batch['video_frames'].to(device))
        
        # Process text data for BERT
        text_strings = []
        for text_data in batch['text_data']:
            combined_text = f"{text_data['title']} {text_data['description']}"
            if text_data['comments']:
                combined_text += " " + " ".join(text_data['comments'][:5])
            text_strings.append(combined_text[:500])
        
        text_features = text_model(text_strings)
        if isinstance(text_features, tuple):
            text_features = text_features[0]
        else:
            pass
        
        # Temporal features processing
        audio_temporal = audio_features.unsqueeze(1).repeat(1, 30, 1)
        video_temporal = visual_features.unsqueeze(1).repeat(1, 30, 1)
        
        max_features = max(audio_temporal.shape[-1], video_temporal.shape[-1])
        if audio_temporal.shape[-1] < max_features:
            padding = torch.zeros(audio_temporal.shape[0], audio_temporal.shape[1], max_features - audio_temporal.shape[-1], device=audio_temporal.device, dtype=audio_temporal.dtype)
            audio_temporal = torch.cat([audio_temporal, padding], dim=-1)
        if video_temporal.shape[-1] < max_features:
            padding = torch.zeros(video_temporal.shape[0], video_temporal.shape[1], max_features - video_temporal.shape[-1], device=video_temporal.device, dtype=video_temporal.dtype)
            video_temporal = torch.cat([video_temporal, padding], dim=-1)
        
        audio_temporal = audio_temporal.to(torch.float32)
        video_temporal = video_temporal.to(torch.float32)
        
        audio_proj = nn.Linear(audio_temporal.shape[-1], 256, dtype=torch.float32).to(device)
        video_proj = nn.Linear(video_temporal.shape[-1], 256, dtype=torch.float32).to(device)
        
        audio_temporal = audio_proj(audio_temporal)
        video_temporal = video_proj(video_temporal)
        
        # Ensure temporal alignment
        target_steps = 30
        if audio_temporal.shape[1] > target_steps:
            audio_temporal = audio_temporal[:, :target_steps, :]
        elif audio_temporal.shape[1] < target_steps:
            pad_size = target_steps - audio_temporal.shape[1]
            audio_temporal = torch.cat([
                audio_temporal,
                torch.zeros(audio_temporal.shape[0], pad_size, audio_temporal.shape[2], device=audio_temporal.device, dtype=audio_temporal.dtype)
            ], dim=1)
        if video_temporal.shape[1] > target_steps:
            video_temporal = video_temporal[:, :target_steps, :]
        elif video_temporal.shape[1] < target_steps:
            pad_size = target_steps - video_temporal.shape[1]
            video_temporal = torch.cat([
                video_temporal,
                torch.zeros(video_temporal.shape[0], pad_size, video_temporal.shape[2], device=video_temporal.device, dtype=video_temporal.dtype)
            ], dim=1)
        
        temporal_features = temporal_model(audio_temporal, video_temporal)
        if isinstance(temporal_features, tuple):
            temporal_features = temporal_features[0]
        if temporal_features.dim() == 3:
            temporal_features = temporal_features.mean(dim=1)
        
        # Specialized forensic analysis
        text_strings = []
        for text_data in batch['text_data']:
            combined_text = f"{text_data['title']} {text_data['description']}"
            if text_data['comments']:
                combined_text += " " + " ".join(text_data['comments'][:5])
            text_strings.append(combined_text[:500])
        
        raw_audio = batch['audio_waveform'].to(device)
        raw_video = batch['video_frames'].to(device)
        
        affective_evidence = affective_model(text_strings, raw_audio, raw_video)
        audio_features = affective_evidence["audio_features"]
        
        chronos_evidence = chronos_model(raw_video)
        semantic_evidence = semantic_model.process_features(text_features, visual_features)
        
        # Get tensor representations
        def get_tensor(x):
            return x[0] if isinstance(x, tuple) else x
        
        text_features = get_tensor(text_features)
        audio_features = get_tensor(audio_features)
        visual_features = get_tensor(visual_features)
        temporal_features = get_tensor(temporal_features)
        
        # Get model prediction
        fusion_out = model({
            'text_features': text_features,  # (batch, 768)
            'audio_features': audio_features,  # (batch, 128)
            'visual_features': visual_features,  # (batch, 512)
            'temporal_features': temporal_features  # (batch, 256)
        })
        
        predictions.append(fusion_out['logits'])
    
    # Average predictions from all models
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred

def _train_with_dummy_data(device, config):
    """Fallback training with dummy data."""
    print("Training with dummy data...")
    
    # Create dummy dataset with optimized size
    class DummyDataset(Dataset):
        def __init__(self, num_samples=100):
            self.text_features = torch.randn(num_samples, 768)
            self.audio_features = torch.randn(num_samples, 128)
            self.visual_features = torch.randn(num_samples, 512)
            self.temporal_features = torch.randn(num_samples, 256)
            self.labels = torch.randint(0, 2, (num_samples,))  # 2 classes: 0 and 1
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return {
                'text_features': self.text_features[idx],
                'audio_features': self.audio_features[idx],
                'visual_features': self.visual_features[idx],
                'temporal_features': self.temporal_features[idx],
                'label': self.labels[idx]
            }
    
    train_dataset = DummyDataset(50)
    test_dataset = DummyDataset(10)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['val_batch_size'])
    
    # Initialize complete pipeline for dummy data
    audio_model = SpectralForensics().to(device)
    visual_model = OpticalFlow3DCNN().to(device)
    text_model = BERTContextEncoder().to(device)
    temporal_model = TemporalSyncNet().to(device)
    affective_model = AffectiveForensics().to(device)
    chronos_model = ChronosGuard().to(device)
    semantic_model = SemanticForgeryDetector().to(device)
    fusion_model = CrossModalTransformer().to(device)
    
    # Training with optimized epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=1e-4)
    
    for epoch in range(min(3, config['max_epochs'])):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Complete pipeline - dummy data already has correct format
            audio_features = audio_model(batch['audio_features'].to(device))
            visual_features = visual_model(batch['visual_features'].to(device))
            text_features = text_model(batch['text_features'].to(device))
            temporal_features = temporal_model(batch['temporal_features'].to(device))
            
            fusion_out = fusion_model({
                'text_features': text_features,
                'audio_features': audio_features,
                'visual_features': visual_features,
                'temporal_features': temporal_features
            })
            
            final_logits = fusion_out['logits']  # Use fusion model's logits directly
            loss = criterion(final_logits, batch['label'].to(device))
            loss.backward()
            
            # M1 GPU memory cleanup
            if device.type == 'mps':
                m1_memory_cleanup()
            
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
    
    # Evaluation
    fusion_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            audio_features = audio_model(batch['audio_features'].to(device))
            visual_features = visual_model(batch['visual_features'].to(device))
            text_features = text_model(batch['text_features'].to(device))
            temporal_features = temporal_model(batch['temporal_features'].to(device))
            
            fusion_out = fusion_model({
                'text_features': text_features,
                'audio_features': audio_features,
                'visual_features': visual_features,
                'temporal_features': temporal_features
            })
            
            final_logits = fusion_out['logits']  # Use fusion model's logits directly
            preds = torch.argmax(final_logits, dim=1)
            correct += (preds == batch['label'].to(device)).sum().item()
            total += batch['label'].size(0)
    
    accuracy = correct / total
    print(f"Dummy Data Test Accuracy: {accuracy*100:.2f}%")
    return {'accuracy': accuracy}

# Add data augmentation imports
import random
import torch.nn.functional as F
from torchvision import transforms

class AugmentedFakeSVDataset(Dataset):
    """Enhanced dataset with data augmentation for better generalization."""
    
    def __init__(self, base_dataset, augment=True):
        self.base_dataset = base_dataset
        self.augment = augment
        
        # Define augmentation transforms
        self.video_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        if self.augment:
            # Augment video frames
            video_frames = sample['video_frames']
            if video_frames.dim() == 4:  # [frames, H, W, C]
                augmented_frames = []
                for frame in video_frames:
                    # Convert to PIL-like format for transforms
                    frame_tensor = frame.permute(2, 0, 1)  # [C, H, W]
                    frame_aug = self.video_transforms(frame_tensor)
                    frame_aug = frame_aug.permute(1, 2, 0)  # [H, W, C]
                    augmented_frames.append(frame_aug)
                sample['video_frames'] = torch.stack(augmented_frames)
            
            # Augment audio with noise and pitch shift
            audio_waveform = sample['audio_waveform']
            if random.random() < 0.3:
                # Add small amount of noise
                noise = torch.randn_like(audio_waveform) * 0.01
                sample['audio_waveform'] = audio_waveform + noise
            
            # Text augmentation (synonym replacement, etc.)
            if random.random() < 0.2:
                text_data = sample['text_data']
                if isinstance(text_data, dict) and 'title' in text_data:
                    # Simple text augmentation - add random words
                    augmentation_words = ['video', 'content', 'media', 'clip', 'footage']
                    if random.random() < 0.5:
                        text_data['title'] += f" {random.choice(augmentation_words)}"
                    sample['text_data'] = text_data
        
        return sample

def create_ensemble_models(device, ensemble_size=2):
    """Create an ensemble of models for better accuracy."""
    models = []
    
    # Create multiple fusion models with different architectures
    for i in range(ensemble_size):
        model = CrossModalTransformer().to(device)
        models.append(model)
    
    return models

def ensemble_predict(models, batch, device):
    """Get ensemble prediction from multiple models."""
    predictions = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            # Process the batch through each model
            # (This is a simplified version - you'd need to implement the full forward pass)
            pred = model(batch)  # This would be the actual prediction
            predictions.append(pred)
    
    # Average predictions
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred

# Add advanced training techniques
def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

if __name__ == "__main__":
    # M1 GPU optimized training execution with automatic fallback
    print("🚀 UltraFND3 M1 GPU Optimized Training and Evaluation")
    print("=" * 60)
    
    # Check M1 GPU availability
    if torch.backends.mps.is_available():
        print("✅ M1 GPU (MPS) detected and available!")
        print("🔧 Optimizing for Apple Silicon...")
    else:
        print("⚠️ M1 GPU not available, using CPU fallback")
    
    # You can specify the data directory here
    data_dir = "/Volumes/SR_disk/FakeSV"  # Raw data directory
    
    # Enable debug mode for quick testing (set to False for full training)
    debug_mode = False  # Set to False for full training
    
    print(f"📁 Data directory: {data_dir}")
    print(f"🐛 Debug mode: {debug_mode}")
    
    try:
        results = train_and_evaluate(data_dir=data_dir, debug_mode=debug_mode)
        
        print("\n🎉 M1 GPU Optimized Training and evaluation completed!")
        print(f"📈 Final Results: {results}")
        
        # M1 GPU specific cleanup
        if torch.backends.mps.is_available():
            m1_memory_cleanup()
            print("🧹 M1 GPU memory cleaned up")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("🔄 Attempting CPU fallback...")
        
        # Force CPU fallback
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        device = torch.device("cpu")
        config = get_optimized_config(device, debug_mode=True)  # Force debug mode for fallback
        results = _train_with_dummy_data(device, config)
        print(f"📈 CPU Fallback Results: {results}") 