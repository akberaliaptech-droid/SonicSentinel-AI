"""Audio processing package exports.
"""
from src.audio_processing.preprocessor import (
    AudioProcessingError,
    AudioSilentError,
    AudioClippedError,
    AudioCorruptError,
    compute_sha256,
    compute_signal_metrics,
    load_audio_buffer,
    validate_and_preprocess,
    extract_mel_spectrogram,
)
from src.audio_processing.doa import AcousticRadarEngine
from src.audio_processing.generator import generate_synthetic_acoustic_profile, apply_spatial_tdoa

__all__ = [
    "AudioProcessingError",
    "AudioSilentError",
    "AudioClippedError",
    "AudioCorruptError",
    "compute_sha256",
    "compute_signal_metrics",
    "load_audio_buffer",
    "validate_and_preprocess",
    "extract_mel_spectrogram",
    "AcousticRadarEngine",
    "generate_synthetic_acoustic_profile",
    "apply_spatial_tdoa",
]
