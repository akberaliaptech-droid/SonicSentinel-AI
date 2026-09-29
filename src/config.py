"""Global configuration and hyperparameters for SonicSentinel AI.
"""
from pathlib import Path
from typing import List, Dict

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "audio_dataset"
SAMPLES_DIR = DATA_DIR / "samples"
REPORTS_DIR = DATA_DIR / "incident_reports"
WEIGHTS_DIR = DATA_DIR / "weights"
DB_PATH = DATA_DIR / "sonicsentinel.db"
STATIC_DIR = PROJECT_ROOT / "static"

# Ensure runtime directories exist
for directory in [DATA_DIR, DATASET_DIR, SAMPLES_DIR, REPORTS_DIR, WEIGHTS_DIR, STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Audio Signal Parameters
SAMPLE_RATE = 44100  # 44.1 kHz standard studio sampling
DURATION_SECONDS = 2.0  # 2-second standard temporal window
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)  # 88,200 samples

# Mel-Spectrogram Feature Extraction Parameters
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
F_MIN = 20.0
F_MAX = 8000.0

# Direction of Arrival (DOA) / GCC-PHAT Radar Parameters
SPEED_OF_SOUND = 343.2  # meters per second at 20°C in air
MIC_DISTANCE = 0.10  # 10 cm baseline microphone spacing
DOA_INTERPOLATION_FACTOR = 8  # Sub-sample precision upsampling

# Error Boundaries & Anti-Cheat Validation Thresholds
# NO hardcoded shortcuts: calculated based on physical acoustics
SILENCE_RMS_THRESHOLD = 0.0015  # Signal with RMS < -56 dBFS is rejected as silence (sensitive to external device playback)
CLIPPING_AMPLITUDE_THRESHOLD = 0.992  # Samples >= 0.992 indicate digital clipping
CLIPPING_MAX_RATIO = 0.015  # Rejection threshold if > 1.5% of samples are clipped
MIN_ACCEPTABLE_SNR_DB = 10.0  # Signals below 10dB SNR trigger manual review

# Dual Model Arbitration Parameters
ARBITRATION_CONFIDENCE_MARGIN_DISAGREEMENT = 0.25
ARBITRATION_TOP_TWO_MARGIN_UNCERTAIN = 0.15

# Mandatory 10 Acoustic Classes (Aptech World Tech Championship SRS)
MANDATORY_CLASSES: List[str] = [
    "Machinery Fault",
    "Glass Breaking",
    "Alarm or Siren",
    "Vehicle Horn",
    "Animal Sound",
    "Gunshot",
    "Panic Scream",
    "Aggression",
    "Person Asking for Help",
    "Background Noise",
]

# Threat Severity Matrix
SEVERITY_MAPPING: Dict[str, str] = {
    "Gunshot": "CRITICAL",
    "Panic Scream": "HIGH",
    "Aggression": "HIGH",
    "Person Asking for Help": "CRITICAL",
    "Glass Breaking": "HIGH",
    "Alarm or Siren": "MEDIUM",
    "Vehicle Horn": "MEDIUM",
    "Machinery Fault": "MEDIUM",
    "Animal Sound": "LOW",
    "Background Noise": "NORMAL",
    # Backward compatibility aliases
    "Explosion": "CRITICAL",
    "Scream_Distress": "HIGH",
    "Glass_Break": "HIGH",
    "Siren": "MEDIUM",
    "Car_Alarm": "MEDIUM",
    "Drilling_Tools": "MEDIUM",
    "Dog_Bark": "LOW",
    "Footsteps": "LOW",
    "Ambient_Noise": "NORMAL",
}

# Color coding for frontend radar and incident status
SEVERITY_COLORS: Dict[str, str] = {
    "CRITICAL": "#ff1744",  # High-intensity red
    "HIGH": "#ff9100",      # Alert orange
    "MEDIUM": "#ffd600",    # Amber
    "LOW": "#00e5ff",       # Cyan
    "NORMAL": "#00e676",    # Emerald green
}
