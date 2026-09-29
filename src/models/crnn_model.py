"""Algorithm 2: Audio CRNN (Convolutional Recurrent Neural Network).
Combines 2D spectral convolutions with a Bidirectional GRU temporal sequence encoder
and temporal self-attention pooling for time-series acoustic event detection.
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import MANDATORY_CLASSES


class TemporalAttention(nn.Module):
    """Self-attention mechanism across temporal frames."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, rnn_out: torch.Tensor) -> torch.Tensor:
        # rnn_out: (batch, time_steps, hidden_dim)
        scores = self.projection(rnn_out)  # (batch, time_steps, 1)
        weights = F.softmax(scores, dim=1)
        context = torch.sum(rnn_out * weights, dim=1)  # (batch, hidden_dim)
        return context


class AudioCRNNModel(nn.Module):
    """Convolutional Recurrent Neural Network (CRNN) Architecture.
    Stage 1: Conv2D feature maps from Log-Mel Spectrogram.
    Stage 2: Bidirectional GRU temporal modeling.
    Stage 3: Temporal attention pooling and classification head.
    """
    def __init__(self, classes: Optional[List[str]] = None, hidden_dim: int = 128):
        super().__init__()
        self.classes = list(classes) if classes else list(MANDATORY_CLASSES)
        self.hidden_dim = hidden_dim

        # Conv Block 1: (B, 1, 64, 173) -> (B, 32, 32, 86)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 2: (B, 32, 32, 86) -> (B, 64, 16, 43)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Block 3: (B, 64, 16, 43) -> (B, 128, 8, 43) (pool only frequency axis)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        # GRU input: 128 channels * 8 freq bins = 1024 features per temporal frame
        rnn_input_dim = 128 * 8
        self.gru = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.25,
        )

        # Bidirectional GRU produces 2 * hidden_dim
        self.attention = TemporalAttention(hidden_dim * 2)

        # Classification Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, len(self.classes))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 64, 173)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Reshape for RNN: (batch, channels, freq, time) -> (batch, time, channels * freq)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Bidirectional GRU
        gru_out, _ = self.gru(x)  # (batch, time, 2 * hidden_dim)

        # Attention pooling across time
        context = self.attention(gru_out)  # (batch, 2 * hidden_dim)

        logits = self.fc(context)
        return logits

    def update_classes(self, new_classes: List[str]):
        """Dynamically expand projection layer."""
        self.classes = list(new_classes)
        in_dim = self.fc[-1].in_features
        self.fc[-1] = nn.Linear(in_dim, len(self.classes))
