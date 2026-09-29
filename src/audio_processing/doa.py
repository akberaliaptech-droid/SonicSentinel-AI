"""Acoustic Direction Finder (DOA) via Generalized Cross-Correlation with Phase Transform (GCC-PHAT).
Implements sub-sample precision TDOA estimation and angular azimuth mapping for 360° radar telemetry.
"""
from typing import Tuple, Dict, Any, Optional
from datetime import datetime
import numpy as np
import scipy.signal
from src.config import (
    SAMPLE_RATE,
    SPEED_OF_SOUND,
    MIC_DISTANCE,
)


class AcousticRadarEngine:
    """Acoustic Radar Engine for Direction of Arrival (DOA) localization.
    
    Reads stereo audio buffers, calculates inter-channel time delay of arrival (TDOA)
    via GCC-PHAT, and projects the acoustic wavefront arrival onto an azimuth angle.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        speed_of_sound: float = SPEED_OF_SOUND,
        mic_distance: float = MIC_DISTANCE,
    ):
        self.sample_rate = sample_rate
        self.c = speed_of_sound  # ~343.2 m/s
        self.d = mic_distance    # distance between stereo mics in meters (0.10m = 10cm)
        # Maximum physically possible delay in seconds: tau_max = d / c
        self.max_tau_seconds = self.d / self.c
        self.max_tau_samples = int(np.ceil(self.max_tau_seconds * self.sample_rate))

    def compute_gcc_phat(
        self,
        sig1: np.ndarray,
        sig2: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """Compute Generalized Cross-Correlation with Phase Transform (GCC-PHAT).
        
        Args:
            sig1: Channel 1 (Left) 1D audio array
            sig2: Channel 2 (Right) 1D audio array
            
        Returns:
            tau_seconds: Estimated time difference of arrival (seconds)
            cc_max: Peak cross-correlation value (coherence metric)
            cc_curve: 1D array of cross-correlation values across lag window
        """
        n1 = len(sig1)
        n2 = len(sig2)
        n = n1 + n2

        # Linear cross-correlation requires at least n1 + n2 - 1 points
        n_fft = 2 ** int(np.ceil(np.log2(n)))
        X1 = np.fft.rfft(sig1, n=n_fft)
        X2 = np.fft.rfft(sig2, n=n_fft)

        # Cross power spectral density
        R12 = X1 * np.conj(X2)

        # Phase transform (PHAT) whitening: divide by magnitude
        epsilon = 1e-12
        denominator = np.abs(R12) + epsilon
        R12_phat = R12 / denominator

        # Inverse FFT to time-domain cross-correlation
        cc = np.fft.irfft(R12_phat, n=n_fft)

        # Shift zero lag to center: index (n_fft // 2) corresponds to lag 0
        cc = np.fft.fftshift(cc)
        zero_index = n_fft // 2

        # Search window bounded physically to [-max_tau_samples - 2, +max_tau_samples + 2]
        search_radius = self.max_tau_samples + 2
        min_idx = max(0, zero_index - search_radius)
        max_idx = min(len(cc), zero_index + search_radius + 1)

        valid_window = cc[min_idx:max_idx]
        if len(valid_window) == 0:
            return 0.0, 0.0, np.zeros(10)

        # Find integer peak lag relative to window start
        peak_idx_rel = int(np.argmax(valid_window))
        cc_max = float(valid_window[peak_idx_rel])

        # Parabolic interpolation for sub-sample precision
        delta = 0.0
        if 0 < peak_idx_rel < len(valid_window) - 1:
            alpha = float(valid_window[peak_idx_rel - 1])
            beta = float(valid_window[peak_idx_rel])
            gamma = float(valid_window[peak_idx_rel + 1])
            denom = 2.0 * (alpha - 2.0 * beta + gamma)
            if abs(denom) > 1e-12:
                delta = (alpha - gamma) / denom

        # Calculate sample lag relative to zero_index
        peak_idx_absolute = min_idx + peak_idx_rel
        lag_samples = float(peak_idx_absolute - zero_index) + delta

        # Convert to time delay in seconds
        tau_seconds = lag_samples / float(self.sample_rate)

        # Physically bound delay
        tau_seconds = float(np.clip(tau_seconds, -self.max_tau_seconds, self.max_tau_seconds))

        return tau_seconds, cc_max, valid_window

    def calculate_azimuth(self, tau_seconds: float) -> float:
        """Convert TDOA (tau) into directional azimuth angle in degrees (0° to 180°).
        
        Using the plane wave acoustic propagation model:
            tau = - (d / c) * cos(theta)
            => cos(theta) = - (c * tau) / d
            => theta = arccos(- (c * tau) / d)
            
        Orientation convention:
            - 0°: Direct right (+X axis)
            - 90°: Boresight / dead ahead (normal to mic array)
            - 180°: Direct left (-X axis)
        """
        ratio = -(self.c * tau_seconds) / self.d
        ratio_clipped = float(np.clip(ratio, -1.0, 1.0))
        theta_rad = float(np.arccos(ratio_clipped))
        azimuth_deg = float(np.degrees(theta_rad))
        return round(azimuth_deg, 2)

    def process_stereo_buffer(
        self,
        stereo_buffer: np.ndarray,
        category: str = "Unknown",
        confidence: float = 0.0,
        severity: str = "NORMAL",
    ) -> Dict[str, Any]:
        """Compute full DOA pipeline on a dual-channel stereo buffer and construct radar telemetry.
        
        Args:
            stereo_buffer: np.ndarray shape (2, N)
            category: Predicted acoustic class
            confidence: Arbitration confidence
            severity: Threat severity label
            
        Returns:
            Dictionary payload ready for WebSocket broadcasting and radar visualization:
            {
                "category": str,
                "confidence": float,
                "azimuth": float,
                "severity": str,
                "timestamp": str,
                "tdoa_seconds": float,
                "coherence": float
            }
        """
        if stereo_buffer.ndim != 2 or stereo_buffer.shape[0] < 2:
            raise ValueError(f"Stereo buffer must have shape (2, N). Got {stereo_buffer.shape}")

        sig_left = stereo_buffer[0]
        sig_right = stereo_buffer[1]

        # Apply high-pass pre-emphasis filter to attenuate room rumble
        b, a = scipy.signal.butter(4, 150.0 / (self.sample_rate / 2.0), btype="highpass")
        sig_left_filt = scipy.signal.filtfilt(b, a, sig_left)
        sig_right_filt = scipy.signal.filtfilt(b, a, sig_right)

        tau_sec, coherence, _ = self.compute_gcc_phat(sig_left_filt, sig_right_filt)
        azimuth_deg = self.calculate_azimuth(tau_sec)

        payload = {
            "category": category,
            "confidence": round(confidence, 4),
            "azimuth": azimuth_deg,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tdoa_seconds": round(tau_sec, 7),
            "coherence": round(coherence, 4),
        }
        return payload
