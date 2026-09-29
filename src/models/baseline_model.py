"""Model B: Standalone Lightweight CNN / Google Teachable Machine (GTM) Audio Baseline Model.
Runs in total isolation from Model A, enforcing strict Anti-Shortcut compliance.
"""
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import MANDATORY_CLASSES


class BaselineGTMModel(nn.Module):
    """Google Teachable Machine (GTM) Audio compatible standalone neural classifier.
    Features a compact, edge-optimized multi-layer convolutional topology
    operating directly on raw Mel-spectrogram patches.
    """

    def __init__(self, classes: Optional[List[str]] = None):
        super().__init__()
        self.classes = list(classes) if classes else list(MANDATORY_CLASSES)

        # GTM compact convolutional architecture with BatchNorm stabilization
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(96)

        self.global_pool = nn.AdaptiveAvgPool2d((2, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(96 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, len(self.classes)),
        )

    def update_classes(self, new_classes: List[str]):
        """Update projection layer for newly registered categories."""
        old_classes = self.classes
        old_linear: nn.Linear = self.classifier[-1]
        new_linear = nn.Linear(old_linear.in_features, len(new_classes))

        with torch.no_grad():
            nn.init.xavier_uniform_(new_linear.weight)
            nn.init.zeros_(new_linear.bias)
            for idx, c in enumerate(new_classes):
                if c in old_classes:
                    old_idx = old_classes.index(c)
                    new_linear.weight[idx] = old_linear.weight[old_idx]
                    new_linear.bias[idx] = old_linear.bias[old_idx]

        self.classifier[-1] = new_linear
        self.classes = list(new_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for baseline model."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute softmax probabilities and class predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return probs, logits
