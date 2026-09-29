"""Explainable AI (XAI) Forensics Engine.
Computes genuine Grad-CAM activation heatmaps overlaying Mel-spectrograms
to prove feature extraction validity on time-frequency regions.
"""
from typing import Tuple, Optional
import io
import base64
from pathlib import Path
import numpy as np
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from src.models.primary_model import PrimaryAudioModel
from src.config import SAMPLE_RATE, DURATION_SECONDS, F_MIN, F_MAX


def compute_gradcam_heatmap(
    model: PrimaryAudioModel,
    mel_tensor: torch.Tensor,
    target_class_idx: int,
) -> np.ndarray:
    """Compute mathematical Grad-CAM class activation map on the last convolutional layer.
    
    Args:
        model: PrimaryAudioModel with registered hooks on conv4
        mel_tensor: torch.Tensor shape (1, 1, 64, 173)
        target_class_idx: Index of target class for gradient backpropagation
        
    Returns:
        cam_resized: np.ndarray shape (64, 173) normalized in range [0, 1]
    """
    model.eval()
    # Ensure gradients can be computed
    mel_input = mel_tensor.clone().detach().requires_grad_(True)

    # Forward pass
    logits = model(mel_input)
    score = logits[0, target_class_idx]

    # Backward pass for target class score
    model.zero_grad()
    score.backward(retain_graph=True)

    # Retrieve hooked activations and gradients
    activations = model.activations.detach().cpu().numpy()[0]  # Shape: (256, H, W)
    gradients = model.gradients.detach().cpu().numpy()[0]      # Shape: (256, H, W)

    # Global Average Pooling of gradients to calculate channel importance weights
    # alpha_k = (1 / Z) * sum_{i,j} (dY_c / dA_k_{i,j})
    weights = np.mean(gradients, axis=(1, 2))  # Shape: (256,)

    # Weighted linear combination of forward activation maps
    cam = np.zeros(activations.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations[i]

    # Apply ReLU: only positive contributions to class score are retained
    cam = np.maximum(cam, 0.0)

    # Min-Max Normalization
    cam_max = np.max(cam)
    if cam_max > 1e-9:
        cam = cam / cam_max
    else:
        cam = np.zeros_like(cam)

    # Bilinear/spline resize to exact Mel-spectrogram dimensions (64, 173)
    target_shape = mel_tensor.shape[2:]  # (64, 173)
    zoom_factors = (target_shape[0] / cam.shape[0], target_shape[1] / cam.shape[1])
    cam_resized = scipy.ndimage.zoom(cam, zoom_factors, order=1)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)

    return cam_resized


def render_spectrogram_gradcam_overlay(
    mel_spectrogram: np.ndarray,
    cam_heatmap: np.ndarray,
    class_name: str,
    confidence: float,
    output_filepath: Optional[str] = None,
) -> Tuple[str, bytes]:
    """Generates a forensic figure overlaying Grad-CAM heatmaps onto the Mel-spectrogram.
    
    Returns:
        base64_data_uri: str for frontend rendering
        img_bytes: raw PNG bytes for database / PDF embedding
    """
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    fig.patch.set_facecolor("#0a0e17")
    ax.set_facecolor("#0a0e17")

    # Time axis (0 to DURATION_SECONDS) and Mel frequency axis
    n_mels, time_frames = mel_spectrogram.shape
    extent = [0, DURATION_SECONDS, F_MIN, F_MAX]

    # Plot base Mel-spectrogram in dark grayscale
    ax.imshow(
        mel_spectrogram,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="gray",
        alpha=0.65,
    )

    # Overlay Grad-CAM activation heatmap with 'jet'/'inferno' colormap
    cax = ax.imshow(
        cam_heatmap,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="jet",
        alpha=0.55,
    )

    # Styling for cyber-security aesthetic
    ax.set_title(
        f"SonicSentinel AI - Explainable AI (Grad-CAM)\nClass: {class_name} | Confidence: {confidence*100:.1f}% | Decisive Spectral Region",
        color="#00f3ff",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Time (seconds)", color="#8b9bb4", fontsize=9)
    ax.set_ylabel("Acoustic Frequency (Hz)", color="#8b9bb4", fontsize=9)
    ax.tick_params(colors="#8b9bb4", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#1f293d")

    # Cyberpunk colorbar
    cb = fig.colorbar(cax, ax=ax, fraction=0.03, pad=0.03)
    cb.set_label("Activation Magnitude (Grad-CAM)", color="#8b9bb4", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="#8b9bb4", labelsize=8)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#8b9bb4")

    plt.tight_layout()

    # Save to memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    img_bytes = buf.getvalue()

    # Optionally persist to disk
    if output_filepath:
        Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(output_filepath, "wb") as f:
            f.write(img_bytes)

    base64_encoded = base64.b64encode(img_bytes).decode("utf-8")
    data_uri = f"data:image/png;base64,{base64_encoded}"

    return data_uri, img_bytes


def render_waveform_plot(
    stereo_audio: np.ndarray,
    output_filepath: str,
    title: str = "Acoustic Dual-Channel Waveform",
) -> bytes:
    """Renders raw stereo waveform for forensic incident records."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 3.5), sharex=True, dpi=150)
    fig.patch.set_facecolor("#0a0e17")

    t = np.linspace(0, DURATION_SECONDS, stereo_audio.shape[1])

    # Left Channel
    ax1.set_facecolor("#0a0e17")
    ax1.plot(t, stereo_audio[0], color="#00e5ff", linewidth=0.8, label="CH1 (Left)")
    ax1.set_ylabel("CH1 Amp", color="#8b9bb4", fontsize=8)
    ax1.set_ylim([-1.05, 1.05])
    ax1.tick_params(colors="#8b9bb4", labelsize=7)
    ax1.grid(color="#1b253b", linestyle="--", linewidth=0.5)
    for spine in ax1.spines.values():
        spine.set_color("#1f293d")

    # Right Channel
    ax2.set_facecolor("#0a0e17")
    ax2.plot(t, stereo_audio[1], color="#ff9100", linewidth=0.8, label="CH2 (Right)")
    ax2.set_xlabel("Time (seconds)", color="#8b9bb4", fontsize=8)
    ax2.set_ylabel("CH2 Amp", color="#8b9bb4", fontsize=8)
    ax2.set_ylim([-1.05, 1.05])
    ax2.tick_params(colors="#8b9bb4", labelsize=7)
    ax2.grid(color="#1b253b", linestyle="--", linewidth=0.5)
    for spine in ax2.spines.values():
        spine.set_color("#1f293d")

    fig.suptitle(title, color="#00f3ff", fontsize=10, fontweight="bold", y=0.98)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    img_bytes = buf.getvalue()

    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(output_filepath, "wb") as f:
        f.write(img_bytes)

    return img_bytes
