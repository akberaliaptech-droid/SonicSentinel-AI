"""Models package exports.
"""
from src.models.primary_model import PrimaryAudioModel
from src.models.baseline_model import BaselineGTMModel
from src.models.explainability import (
    compute_gradcam_heatmap,
    render_spectrogram_gradcam_overlay,
    render_waveform_plot,
)
from src.models.inference import DualIndependentInferenceEngine
from src.models.fine_tuner import ContinuousLearningFineTuner, fine_tuner

__all__ = [
    "PrimaryAudioModel",
    "BaselineGTMModel",
    "compute_gradcam_heatmap",
    "render_spectrogram_gradcam_overlay",
    "render_waveform_plot",
    "DualIndependentInferenceEngine",
    "ContinuousLearningFineTuner",
    "fine_tuner",
]
