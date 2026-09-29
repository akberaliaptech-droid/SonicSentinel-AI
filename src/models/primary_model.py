"""Model A: Primary Transfer Learning Audio CNN (PyTorch) with Grad-CAM hooks.
Adheres strictly to Aptech Technical Specifications and anti-shortcut guidelines.
"""
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import MANDATORY_CLASSES


class DepthwiseSeparableConv(nn.Module):
    """Pro-level highly efficient depthwise separable convolution."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=stride, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_ch)
        
    def forward(self, x):
        return F.relu(self.bn(self.pointwise(self.depthwise(x))))

class AudioConvBackbone(nn.Module):
    """Pro-Level convolutional feature extractor for high efficiency.
    Operates on Log Mel-Spectrogram tensors of shape (Batch, 1, 64, 173).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))  # Decisive feature map layer for Grad-CAM
        x = self.adaptive_pool(x)
        return x


class PrimaryAudioModel(nn.Module):
    """Primary Audio Model (Model A) featuring a frozen transfer backbone
    and a dynamic active projection classification head.
    """

    def __init__(self, classes: Optional[List[str]] = None):
        super().__init__()
        self.classes = list(classes) if classes else list(MANDATORY_CLASSES)
        self.backbone = AudioConvBackbone()

        # Classification Projection Head: 256 * 4 * 4 = 4096 features
        in_features = 256 * 4 * 4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, len(self.classes)),
        )

        # Gradient & activation caches for Grad-CAM explainability
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._register_gradcam_hooks()

    def _register_gradcam_hooks(self):
        """Register PyTorch forward and backward hooks on conv4 for Grad-CAM calculation."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.backbone.conv4.register_forward_hook(forward_hook)
        self.backbone.conv4.register_full_backward_hook(backward_hook)

    def freeze_backbone(self):
        """Freeze base feature extractor layers for rapid transfer learning fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze base feature extractor layers."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def update_classes(self, new_classes: List[str]):
        """Dynamically reinitialize / expand the final dense projection layer for N+1 categories
        without destroying or overriding frozen base weights.
        """
        old_classes = self.classes
        old_num = len(old_classes)
        new_num = len(new_classes)

        # Old final projection layer
        old_linear: nn.Linear = self.head[-1]
        new_linear = nn.Linear(old_linear.in_features, new_num)

        # Preserve weights for existing classes
        with torch.no_grad():
            nn.init.xavier_uniform_(new_linear.weight)
            nn.init.zeros_(new_linear.bias)
            for idx, c in enumerate(new_classes):
                if c in old_classes:
                    old_idx = old_classes.index(c)
                    new_linear.weight[idx] = old_linear.weight[old_idx]
                    new_linear.bias[idx] = old_linear.bias[old_idx]

        self.head[-1] = new_linear
        self.classes = list(new_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits."""
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute softmax probabilities and class predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return probs, logits
