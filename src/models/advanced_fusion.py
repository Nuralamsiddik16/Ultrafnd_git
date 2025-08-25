import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class HierarchicalFusionModel(nn.Module):
    """Text, visual and metadata fusion with cross-modal attention."""

    def __init__(self, config):
        super().__init__()

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(config.text_model_name)
        text_dim = self.text_encoder.config.hidden_size

        # Visual encoder - simple 3D CNN
        self.visual_encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        visual_dim = 128

        # Metadata encoder
        self.meta_encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        meta_dim = 32

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=8,
            batch_first=True,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(text_dim + visual_dim + meta_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, config.num_labels),
        )

    def forward(self, text_input, video_frames, meta_features):
        # Text features
        text_outputs = self.text_encoder(**text_input)
        text_features = text_outputs.last_hidden_state  # (B, L, H)

        # Visual features
        b, f, c, h, w = video_frames.shape
        video_input = video_frames.permute(0, 2, 1, 3, 4)
        visual_features = self.visual_encoder(video_input).view(b, -1)

        # Metadata features
        meta_features = self.meta_encoder(meta_features)

        # Cross-modal attention: attend text tokens to visual global feature
        cls_token = text_features[:, 0:1, :]
        visual_expanded = visual_features.unsqueeze(1).repeat(1, text_features.size(1), 1)
        attended, _ = self.cross_attention(
            query=text_features,
            key=visual_expanded,
            value=visual_expanded,
        )
        attended_pooled = attended.mean(dim=1)

        combined = torch.cat([attended_pooled, visual_features, meta_features], dim=1)
        return self.classifier(combined)


class SimpleFusionModel(nn.Module):
    """Very small baseline using only metadata features."""

    def __init__(self, config):
        super().__init__()
        self.classifier = nn.Linear(4, config.num_labels)

    def forward(self, text_input, video_frames, meta_features):
        return self.classifier(meta_features)
