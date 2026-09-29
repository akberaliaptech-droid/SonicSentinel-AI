"""Acoustic Test Signal Generator for SonicSentinel AI.
Generates authentic physical audio profiles for the 10 mandatory classes with
calibrated spatial time-delays (TDOA) for Direction of Arrival (DOA) radar validation.
"""
from typing import Tuple, List, Dict
import numpy as np
import scipy.signal
from src.config import (
    SAMPLE_RATE,
    DURATION_SECONDS,
    SPEED_OF_SOUND,
    MIC_DISTANCE,
    MANDATORY_CLASSES,
)


def apply_spatial_tdoa(
    mono_signal: np.ndarray,
    target_azimuth_deg: float,
    sample_rate: int = SAMPLE_RATE,
    mic_distance: float = MIC_DISTANCE,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> np.ndarray:
    """Applies inter-channel time delay of arrival (TDOA) corresponding to target azimuth angle.
    
    Returns:
        stereo_signal: np.ndarray shape (2, N)
    """
    theta_rad = np.radians(target_azimuth_deg)
    # tau = (d / c) * cos(theta)
    tau_seconds = (mic_distance / speed_of_sound) * np.cos(theta_rad)
    tau_samples = tau_seconds * sample_rate

    n = len(mono_signal)
    fft_signal = np.fft.rfft(mono_signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Apply phase shifts: +tau/2 for channel 1, -tau/2 for channel 2
    # In time domain, delay = exp(-2j * pi * f * delay)
    phase_left = np.exp(-1j * 2.0 * np.pi * freqs * (-tau_seconds / 2.0))
    phase_right = np.exp(-1j * 2.0 * np.pi * freqs * (tau_seconds / 2.0))

    ch_left = np.fft.irfft(fft_signal * phase_left, n=n)
    ch_right = np.fft.irfft(fft_signal * phase_right, n=n)

    # Slight acoustic head shadow attenuation (higher frequency slight drop off on contralateral side)
    if tau_seconds > 0:
        # Arriving closer to left microphone
        ch_right *= 0.95
    elif tau_seconds < 0:
        ch_left *= 0.95

    return np.stack([ch_left.astype(np.float32), ch_right.astype(np.float32)], axis=0)


def generate_synthetic_acoustic_profile(
    class_name: str,
    azimuth_deg: float = 90.0,
    duration_sec: float = DURATION_SECONDS,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Generates an acoustic waveform exhibiting characteristic spectral signatures
    of one of the mandatory acoustic classes, positioned at azimuth_deg.
    """
    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    mono = np.zeros(total_samples, dtype=np.float32)

    rng = np.random.RandomState(abs(hash(class_name) + int(azimuth_deg * 100)) % (2**31 - 1))

    c_lower = class_name.lower().replace(" ", "_")

    if "gunshot" in c_lower or "firework" in c_lower:
        # Sharp high-energy acoustic impulse followed by explosive reverberation
        impulse_time = 0.25
        impulse_idx = int(impulse_time * sample_rate)
        decay_samples = int(0.7 * sample_rate)
        if impulse_idx + decay_samples < total_samples:
            env = np.exp(-np.linspace(0, 18, decay_samples))
            noise = rng.normal(0, 1, decay_samples)
            sos = scipy.signal.butter(4, [750, 4800], btype="bandpass", fs=sample_rate, output="sos")
            shot = scipy.signal.sosfilt(sos, noise) * env * 1.6
            mono[impulse_idx : impulse_idx + decay_samples] += shot.astype(np.float32)

    elif "glass" in c_lower:
        # Cascading high-frequency crystalline resonant bursts (2500Hz - 7500Hz)
        for offset in [0.2, 0.28, 0.35, 0.45]:
            idx = int(offset * sample_rate)
            dur = int(0.32 * sample_rate)
            if idx + dur < total_samples:
                resonances = (
                    0.5 * np.sin(2 * np.pi * 3200 * t[:dur]) +
                    0.4 * np.sin(2 * np.pi * 4800 * t[:dur]) +
                    0.3 * np.sin(2 * np.pi * 6500 * t[:dur]) +
                    0.6 * rng.normal(0, 1, dur)
                )
                env = np.exp(-np.linspace(0, 12, dur))
                mono[idx : idx + dur] += (resonances * env * 0.85).astype(np.float32)

    elif "siren" in c_lower or "alarm" in c_lower:
        # Frequency-modulated dual-tone warble (650Hz to 1250Hz)
        modulator = 0.5 * (1.0 + np.sin(2 * np.pi * 1.6 * t))
        carrier = 650.0 + 580.0 * modulator
        phase = 2 * np.pi * np.cumsum(carrier) / sample_rate
        siren_tone = np.sin(phase) + 0.35 * np.sin(2 * phase)
        mono += (siren_tone * 0.75).astype(np.float32)

    elif "horn" in c_lower:
        # Dual-tone vehicle horn (340Hz and 440Hz klaxon chord)
        horn_chord = np.sin(2 * np.pi * 340.0 * t) + 0.8 * np.sin(2 * np.pi * 440.0 * t) + 0.3 * np.sin(2 * np.pi * 880.0 * t)
        gate = ((t % 0.6) < 0.45).astype(float)
        mono += (horn_chord * gate * 0.8).astype(np.float32)

    elif "animal" in c_lower or "dog" in c_lower or "bark" in c_lower:
        # Rhythmic burst barks with pitch descent (600Hz to 350Hz)
        for offset in [0.2, 0.7, 1.3]:
            idx = int(offset * sample_rate)
            dur = int(0.26 * sample_rate)
            if idx + dur < total_samples:
                tb = np.linspace(0, 0.26, dur)
                f_desc = np.linspace(620, 340, dur)
                phase = 2 * np.pi * np.cumsum(f_desc) / sample_rate
                bark = (np.sin(phase) + 0.4 * rng.normal(0, 1, dur)) * np.sin(np.pi * tb / 0.26)
                mono[idx : idx + dur] += (bark * 0.9).astype(np.float32)

    elif "machinery" in c_lower or "fault" in c_lower or "drill" in c_lower or "engine" in c_lower:
        # High RPM motor harmonic whirr + abrasive friction noise + metallic grinding
        motor_harmonics = (
            np.sin(2 * np.pi * 240 * t) +
            0.8 * np.sin(2 * np.pi * 720 * t) +
            0.6 * np.sin(2 * np.pi * 1440 * t) +
            0.5 * np.sin(2 * np.pi * 2880 * t)
        )
        friction = rng.normal(0, 0.45, total_samples)
        mono += ((motor_harmonics + friction) * 0.65).astype(np.float32)

    elif "panic" in c_lower or "scream" in c_lower:
        # Harmonic formant peaks in distress range (850Hz, 1700Hz, 2900Hz) with rapid vibrato
        f0 = 850.0 + 75.0 * np.sin(2 * np.pi * 6.5 * t)
        phase = 2 * np.pi * np.cumsum(f0) / sample_rate
        harmonics = (
            np.sin(phase) +
            0.75 * np.sin(2 * phase) +
            0.55 * np.sin(3 * phase) +
            0.25 * rng.normal(0, 1, total_samples)
        )
        gate = (np.sin(np.pi * t / duration_sec) ** 2)
        mono += (harmonics * gate * 0.85).astype(np.float32)

    elif "aggression" in c_lower:
        # Shouted hostile vocal formants (500Hz, 1200Hz, 2400Hz) with aggressive grit & clipping distortion
        f0 = 420.0 + 80.0 * np.sin(2 * np.pi * 3.5 * t) + rng.normal(0, 15, total_samples)
        phase = 2 * np.pi * np.cumsum(f0) / sample_rate
        shout = (
            np.sin(phase) +
            0.8 * np.sin(2 * phase) +
            0.6 * np.sin(3 * phase) +
            0.4 * rng.normal(0, 1, total_samples)
        )
        # Apply saturation overdrive for aggressive vocal strain
        shout_overdrive = np.tanh(shout * 2.2)
        gate = (np.sin(np.pi * t / duration_sec) ** 1.5)
        mono += (shout_overdrive * gate * 0.85).astype(np.float32)

    elif "help" in c_lower or "person" in c_lower:
        # Vocal formant distress synthesis ("Help me", human harmonic cadence)
        f0 = 260.0 + (45.0 * np.sin(2 * np.pi * 4.5 * t))
        phase = 2 * np.pi * np.cumsum(f0) / sample_rate
        vocal = (
            np.sin(phase) +
            0.65 * np.sin(2 * phase) +
            0.45 * np.sin(3 * phase) +
            0.25 * np.sin(4 * phase)
        )
        pulse = (np.sin(2 * np.pi * 1.8 * t) ** 2)
        mono += (vocal * pulse * 0.85).astype(np.float32)

    elif "explosion" in c_lower:
        # Low-frequency rumble shockwave (30Hz - 250Hz) with long acoustic tail
        idx = int(0.2 * sample_rate)
        dur = int(1.4 * sample_rate)
        if idx + dur < total_samples:
            env = np.exp(-np.linspace(0, 4.5, dur))
            rumble_noise = rng.normal(0, 1, dur)
            sos = scipy.signal.butter(4, [30, 260], btype="bandpass", fs=sample_rate, output="sos")
            shockwave = scipy.signal.sosfilt(sos, rumble_noise) * env * 2.0
            mono[idx : idx + dur] += shockwave.astype(np.float32)

    else:  # Background Noise or Ambient Floor
        # Smooth pink/diffuse ambient acoustic background
        noise = rng.normal(0, 0.18, total_samples)
        b, a = scipy.signal.butter(2, 0.18, btype="lowpass")
        mono += scipy.signal.filtfilt(b, a, noise).astype(np.float32)

    # Add gentle ambient acoustic floor (prevent absolute silence)
    mono += (rng.normal(0, 0.015, total_samples)).astype(np.float32)

    # Normalize peak amplitude to safe level ~ 0.85
    peak = np.max(np.abs(mono))
    if peak > 0:
        mono = (mono / peak) * 0.85

    # Apply spatial TDOA stereo rendering
    stereo = apply_spatial_tdoa(mono, azimuth_deg, sample_rate, MIC_DISTANCE, SPEED_OF_SOUND)
    return stereo
