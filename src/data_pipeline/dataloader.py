import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import torchvision.transforms as T


def get_video_transform():
    """Default transform for video frames."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class FakeNewsDataset(Dataset):
    """Dataset wrapper around the FakeSV metadata layout.

    Each sample returns tokenised text inputs, a stack of video frames, a
    metadata feature vector and the binary label (0 real, 1 fake).
    """

    def __init__(self,
                 root_dir: str,
                 split: str = "train",
                 transform: Optional[T.Compose] = None,
                 text_model_name: str = "bert-base-uncased",
                 max_length: int = 128,
                 scaler: Optional[object] = None):
        self.root_dir = root_dir
        self.transform = transform or get_video_transform()
        self.split = split
        self.scaler = scaler

        # --- Load main metadata
        meta_path = os.path.join(root_dir, "metadata.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"metadata.csv not found in {root_dir}")
        self.meta_df = pd.read_csv(meta_path)

        # --- Apply official split
        split_file = os.path.join(root_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file {split}.txt missing in {root_dir}")
        with open(split_file, "r") as f:
            video_ids_in_split = [line.strip() for line in f.readlines()]
        self.meta_df = self.meta_df[self.meta_df["video_id"].isin(video_ids_in_split)].reset_index(drop=True)

        # --- Tokeniser
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.max_length = max_length

        # --- Preprocess metadata
        self._preprocess_metadata()

    def _preprocess_metadata(self):
        """Normalise and scale numerical metadata columns."""
        from sklearn.preprocessing import StandardScaler

        numerical_features = ["like_count", "share_count", "comment_count"]
        for feat in numerical_features:
            self.meta_df[feat] = self.meta_df[feat].fillna(0)

        if self.split == "train":
            self.scaler = StandardScaler()
            self.meta_df[numerical_features] = self.scaler.fit_transform(self.meta_df[numerical_features])
        else:
            if self.scaler is None:
                raise RuntimeError("Scaler must be provided for non-training split")
            self.meta_df[numerical_features] = self.scaler.transform(self.meta_df[numerical_features])

        # Convert verified flag to int
        self.meta_df["user_verified"] = self.meta_df["user_verified"].astype(int)

    def _sample_video_frames(self, video_path: str, num_frames: int = 16):
        """Uniformly sample frames from a video file."""
        import cv2

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        cap.release()
        return frames

    def __getitem__(self, idx: int):
        row = self.meta_df.iloc[idx]
        video_id = row["video_id"]
        label = row["label"]

        # Text modality
        text = f"{row['title']} [SEP] {row['description']}"
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Visual modality
        video_path = os.path.join(self.root_dir, "videos", f"{video_id}.mp4")
        frames = self._sample_video_frames(video_path, num_frames=16)
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])

        # Metadata modality
        meta_features = torch.tensor(
            [
                row["like_count"],
                row["share_count"],
                row["comment_count"],
                row["user_verified"],
            ],
            dtype=torch.float32,
        )

        return text_inputs, frames, meta_features, int(label)

    def __len__(self) -> int:
        return len(self.meta_df)
