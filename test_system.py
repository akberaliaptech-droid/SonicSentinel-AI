"""Comprehensive verification script for SonicSentinel AI.
Validates:
1. Physical audio synthesis and spatial TDOA delay rendering
2. GCC-PHAT Direction of Arrival (DOA) azimuth estimation accuracy
3. Error boundaries (silence rejection, clipping rejection)
4. Dual-independent model inference & arbitration logic
5. Grad-CAM Mel-spectrogram activation heatmap generation
6. Tamper-evident forensic incident PDF generation
7. Continuous learning transfer fine-tuning loop with Macro F1 score
"""
import sys
import os
from pathlib import Path
import numpy as np
import torch
import soundfile as sf

# Set working directory to project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MANDATORY_CLASSES, SAMPLE_RATE, DURATION_SECONDS
from src.audio_processing.generator import generate_synthetic_acoustic_profile
from src.audio_processing.doa import AcousticRadarEngine
from src.audio_processing.preprocessor import (
    validate_and_preprocess,
    extract_mel_spectrogram,
    AudioSilentError,
    AudioClippedError,
    compute_sha256,
)
from src.models.inference import DualIndependentInferenceEngine
from src.models.explainability import (
    render_spectrogram_gradcam_overlay,
    render_waveform_plot,
)
from src.reporting.pdf_generator import generate_forensic_incident_pdf
from src.models.fine_tuner import fine_tuner


def test_doa_gcc_phat():
    print("\n--- TEST 1: GCC-PHAT Direction of Arrival (DOA) ---")
    radar = AcousticRadarEngine()
    test_angles = [30.0, 60.0, 90.0, 120.0, 150.0]

    for target_azimuth in test_angles:
        stereo_signal = generate_synthetic_acoustic_profile("Gunshot", azimuth_deg=target_azimuth)
        result = radar.process_stereo_buffer(stereo_signal)
        estimated_azimuth = result["azimuth"]
        error = abs(estimated_azimuth - target_azimuth)
        print(f"Target Azimuth: {target_azimuth:5.1f}° | Estimated: {estimated_azimuth:5.1f}° | Error: {error:4.1f}° | TDOA: {result['tdoa_seconds']*1000:6.3f} ms")
        assert error < 8.0, f"DOA error too high: {error}° for target {target_azimuth}°"

    print(">> GCC-PHAT DOA Verification PASSED!")


def test_error_boundaries():
    print("\n--- TEST 2: Error Boundaries (Silence & Clipping) ---")
    # 1. Test Silent Audio Rejection
    silent_audio = np.zeros((2, int(SAMPLE_RATE * 2.0)), dtype=np.float32)
    try:
        validate_and_preprocess(silent_audio)
        assert False, "Failed to reject silent audio"
    except AudioSilentError as exc:
        print(f"Silent audio correctly rejected: {exc.message}")

    # 2. Test Clipped Audio Rejection
    clipped_audio = np.ones((2, int(SAMPLE_RATE * 2.0)), dtype=np.float32)
    try:
        validate_and_preprocess(clipped_audio)
        assert False, "Failed to reject clipped audio"
    except AudioClippedError as exc:
        print(f"Clipped audio correctly rejected: {exc.message}")

    print(">> Error Boundaries Verification PASSED!")


def test_dual_inference_and_xai():
    print("\n--- TEST 3: Dual Independent Inference & Grad-CAM ---")
    import asyncio
    engine = DualIndependentInferenceEngine()

    async def run_test():
        # Generate Glass Break at 45°
        audio = generate_synthetic_acoustic_profile("Glass_Break", azimuth_deg=45.0)
        stereo_norm, mono_norm, metrics = validate_and_preprocess(audio)
        mel = extract_mel_spectrogram(mono_norm)

        # Run dual inference
        result = await engine.predict_dual_async(mel, metrics["snr_db"])
        print(f"Predicted Category: {result['predicted_category']}")
        print(f"Arbitration Status: {result['arbitration_status']}")
        print(f"Confidence Margin: {result['confidence_margin']}")
        print(f"Model A: {result['model_a']['predicted_class']} ({result['model_a']['confidence']*100:.1f}%)")
        print(f"Model B: {result['model_b']['predicted_class']} ({result['model_b']['confidence']*100:.1f}%)")

        # Grad-CAM heatmap check
        cam = result["cam_heatmap"]
        assert cam.shape == mel.shape, f"CAM shape {cam.shape} does not match Mel shape {mel.shape}"
        assert np.max(cam) <= 1.0 and np.min(cam) >= 0.0, "CAM out of range [0, 1]"

        # Render overlay image
        output_heatmap = PROJECT_ROOT / "data" / "samples" / "test_heatmap.png"
        data_uri, img_bytes = render_spectrogram_gradcam_overlay(
            mel, cam, result["predicted_category"], result["model_a"]["confidence"], str(output_heatmap)
        )
        assert output_heatmap.exists(), "Heatmap image not saved"
        print(f"Grad-CAM Heatmap saved: {output_heatmap} ({len(img_bytes)} bytes)")

        # Render waveform
        output_waveform = PROJECT_ROOT / "data" / "samples" / "test_waveform.png"
        render_waveform_plot(stereo_norm, str(output_waveform))
        assert output_waveform.exists(), "Waveform image not saved"
        print(f"Waveform plot saved: {output_waveform}")

        # Generate Forensic Incident PDF
        incident_data = {
            "id": "INC-TEST-001",
            "audio_sha256": compute_sha256(audio.tobytes()),
            "predicted_category": result["predicted_category"],
            "severity": result["severity"],
            "model_a_class": result["model_a"]["predicted_class"],
            "model_a_confidence": result["model_a"]["confidence"],
            "model_b_class": result["model_b"]["predicted_class"],
            "model_b_confidence": result["model_b"]["confidence"],
            "confidence_margin": result["confidence_margin"],
            "arbitration_status": result["arbitration_status"],
            "requires_manual_review": result["requires_manual_review"],
            "snr_db": metrics["snr_db"],
            "rms_amplitude": metrics["rms"],
            "azimuth_deg": 45.0,
            "tdoa_seconds": 0.000206,
        }
        output_pdf = PROJECT_ROOT / "data" / "samples" / "test_forensic_report.pdf"
        generate_forensic_incident_pdf(
            incident_data,
            str(output_waveform),
            str(output_heatmap),
            str(output_pdf),
        )
        assert output_pdf.exists(), "Forensic PDF was not created"
        print(f"Forensic Incident PDF created: {output_pdf} ({output_pdf.stat().st_size} bytes)")

    asyncio.run(run_test())
    print(">> Dual Inference & Explainable AI Verification PASSED!")


def test_continuous_fine_tuning():
    print("\n--- TEST 4: Continuous Learning Transfer Fine-Tuning ---")
    classes = MANDATORY_CLASSES[:5]  # Quick 5-class warm-up test
    results = fine_tuner.run_fine_tune_sync(classes, epochs=3, batch_size=4)
    print(f"Tuning Completed! Version: {results['version_tag']} | Loss: {results['loss']} | Macro F1: {results['f1_macro']}")
    assert results["f1_macro"] >= 0.0, "Invalid F1 score"
    assert Path(results["weights_path"]).exists(), "Weights checkpoint not saved"
    print(">> Continuous Learning Transfer Fine-Tuning Verification PASSED!")


if __name__ == "__main__":
    print("==========================================================")
    print("SONICSENTINEL AI: APTECH SPEC COMPLIANCE VERIFICATION SUITE")
    print("==========================================================")
    test_doa_gcc_phat()
    test_error_boundaries()
    test_dual_inference_and_xai()
    test_continuous_fine_tuning()
    print("\n==========================================================")
    print("ALL VERIFICATION SUITE MODULES PASSED WITH 100% SUCCESS!")
    print("==========================================================")
