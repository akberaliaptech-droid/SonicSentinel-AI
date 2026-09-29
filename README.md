# SonicSentinel AI // Enterprise Acoustic Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![License: Aptech World Tech Spec](https://img.shields.io/badge/Spec-Aptech%20Tech%20Championship-cyan.svg)](#)

**SonicSentinel AI** is an enterprise-grade acoustic surveillance and intelligence system developed strictly complying with the Technical Specification Document (Aptech World Tech Championship).

The platform integrates dual-independent neural classification architectures, Generalized Cross-Correlation with Phase Transform (GCC-PHAT) 360-degree radar direction finding, Explainable AI (XAI) Grad-CAM feature heatmaps, tamper-evident forensic PDF incident dossiers, and continuous transfer learning with automated storage provisioning and hot-swapping weights without server downtime.

---

## 🏛️ System Architecture

```
                                  +---------------------------------------+
                                  |    Stereo Dual-Channel Audio Stream   |
                                  |      (16-bit PCM, 44.1 kHz, 2.0s)     |
                                  +-------------------+-------------------+
                                                      |
                                     [Cryptographic SHA-256 Digest]
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |        Acoustic Error Boundaries      |
                                  |   - RMS Floor: Reject if < 0.0040     |
                                  |   - Peak Clipping: Reject if > 0.99   |
                                  |   - Temporal Framing: 88,200 samples  |
                                  +-------------------+-------------------+
                                                      |
                         +----------------------------+----------------------------+
                         |                                                         |
                         v                                                         v
        +----------------------------------+                     +----------------------------------+
        |   Acoustic Direction Finder      |                     |      Log Mel-Spectrogram         |
        |   (AcousticRadarEngine)          |                     |    Feature Representation        |
        |   - GCC-PHAT TDOA Calculation    |                     |    - 64 Mel bands, N_FFT=1024    |
        |   - Parabolic Peak Refinement    |                     |    - Shape: (1, 1, 64, 173)      |
        |   - Azimuth Mapping (0° to 180°) |                     +-----------------+----------------+
        +----------------+-----------------+                                       |
                         |                                                         |
                         |                          +------------------------------+------------------------------+
                         |                          |                                                             |
                         |                          v                                                             v
                         |          +--------------------------------+                            +--------------------------------+
                         |          |            MODEL A             |                            |            MODEL B             |
                         |          |   Transfer Learning Audio CNN  |                            |   Baseline Standalone GTM CNN  |
                         |          |   (PyTorch Deep Backbone)      |                            |   (Lightweight Edge Topology)  |
                         |          |   - Cloned Tensor A            |                            |   - Cloned Tensor B (Isolated) |
                         |          |   - Grad-CAM Forward/Back Hook |                            |   - Strict Anti-Shortcut       |
                         |          +---------------+----------------+                            +---------------+----------------+
                         |                          |                                                             |
                         |                          +------------------------------+------------------------------+
                         |                                                         |
                         v                                                         v
        +---------------------------------------------------------------------------------------------------------+
        |                               Dual-Independent Inference & Arbitration Engine                            |
        |   - Confidence Margin: Delta = |Conf_A - Conf_B|                                                        |
        |   - Classification: 'Acceptable Match', 'Weak Match', 'Model Disagreement', 'Uncertain'                |
        |   - Manual Review Queue: Triggered on Disagreement, SNR < 10dB, or Top-Two Margin < 0.15                |
        +----------------------------------------------------+----------------------------------------------------+
                                                             |
                         +-----------------------------------+-----------------------------------+
                         |                                                                       |
                         v                                                                       v
        +----------------------------------+                                   +----------------------------------+
        |       Explainable AI (XAI)       |                                   |  Continuous Active Learning Loop |
        |   - Grad-CAM Mel Overlay         |                                   |  - Dynamic Category Provisioning |
        |   - Dual-Channel Oscillogram     |                                   |  - Frozen Backbone Fine-Tuning   |
        |   - Tamper-Evident Forensic PDF  |                                   |  - Macro F1 & Confusion Matrix   |
        |     Incident Dossier             |                                   |  - Zero-Downtime Hot-Swapping    |
        +----------------+-----------------+                                   +----------------+-----------------+
                         |                                                                       |
                         +-----------------------------------+-----------------------------------+
                                                             |
                                                             v
                                        +-----------------------------------------+
                                        |    FastAPI Real-Time WebSocket Server   |
                                        |   - /ws/live-audio Telemetry Broadcast  |
                                        |   - 360° Tactical Radar UI Dashboard    |
                                        +-----------------------------------------+
```

---

## 🔬 Core Module Specifications

### 1. Acoustic Direction Finder (`src/audio_processing/doa.py`)
- Reads dual-channel stereo buffers (44.1 kHz, baseline microphone spacing $d = 0.10$ m).
- Generalized Cross-Correlation with Phase Transform (GCC-PHAT):
  $$\psi_{\text{PHAT}}(f) = \frac{X_1(f) X_2^*(f)}{|X_1(f) X_2^*(f)| + \epsilon}$$
  $$R_{12}(\tau) = \mathcal{F}^{-1} \{\psi_{\text{PHAT}}(f)\}$$
- Sub-sample parabolic interpolation around cross-correlation peak:
  $$\delta = \frac{\alpha - \gamma}{2(\alpha - 2\beta + \gamma)}$$
- Planar wavefront azimuth projection ($0^\circ \text{ to } 180^\circ$):
  $$\theta = \arccos\left(-\frac{c \cdot \Delta t}{d}\right) \cdot \frac{180}{\pi}$$
- Verified accuracy: Sub-degree mean absolute error across all incident angles.

### 2. Dual-Model Architecture & Anti-Shortcut Arbitration (`src/models/inference.py`)
- **Model A (Primary Model)**: Deep transfer convolutional audio network with active classification head and registered hooks on `conv4` for Grad-CAM activation mapping.
- **Model B (Baseline Model)**: Standalone lightweight audio CNN based on Google Teachable Machine audio topology running in total memory isolation.
- **Strict Anti-Shortcut Compliance**: Model A and Model B operate on independent cloned tensors. Under no circumstances does Model A's output feed into Model B.
- **Consensus Arbitration Matrix**:
  - `Acceptable Match`: Models agree on top class, confidence margin $\le 0.25$, confidence $\ge 0.55$.
  - `Weak Match`: Top class agrees, but confidence is borderline.
  - `Model Disagreement`: Top-1 predicted classes differ between Model A and Model B.
  - `Uncertain`: Top-two confidence margin $< 0.15$ or signal SNR $< 10$ dB.
  - Automatic escalation to **Manual Review Queue** if disagreement or uncertainty is flagged.

### 3. Continuous Learning Pipeline (`src/models/fine_tuner.py`)
- Dynamic category provisioning via `/api/v1/categories/create`:
  - Name validation (alphanumeric, hyphen, underscore).
  - Automatically initializes separate staging directories:
    - `data/audio_dataset/{category}/python_model/`
    - `data/audio_dataset/{category}/gtm_slices/`
- Background transfer fine-tuning worker via `/api/v1/train/fine-tune`:
  - Freezes base feature-extractor layers.
  - Dynamically reinitializes the final dense projection head for $N+1$ categories.
  - Executes warm-up fine-tuning epoch loop.
  - Computes Confusion Matrix and Macro F1 score via scikit-learn.
  - Hot-swaps inference engine active weights without stopping the server.

### 4. Explainable AI & Tamper-Evident Forensic Dossiers (`src/models/explainability.py`, `src/reporting/pdf_generator.py`)
- **Grad-CAM Mel-Spectrogram Overlays**: Computes gradients of the target class score relative to `conv4` feature activations, calculates channel importance weights $\alpha_k$, and renders an overlaid heatmap on the time-frequency spectrogram.
- **Forensic PDF Incident Dossier**: Generates tamper-evident forensic documentation via ReportLab containing:
  - SHA-256 cryptographic hash of the raw audio buffer.
  - Dual-channel time-domain oscillogram.
  - Log Mel-spectrogram with Grad-CAM heatmap overlay.
  - Dual-model comparison matrix and confidence vectors.
  - GCC-PHAT spatial radar telemetry (Azimuth, TDOA, Coherence).
  - Operator audit trail and digital verification token.

### 5. Acoustic Error Boundaries (`src/audio_processing/preprocessor.py`)
- **Silence Boundary**: Calculates RMS amplitude; rejects audio if $\text{RMS} < 0.0040$ with HTTP 422 `ERR_AUDIO_SILENT`.
- **Digital Clipping Boundary**: Detects peak amplitude and clipped sample ratio; rejects if peak $\ge 0.992$ and clipped ratio $> 1.50\%$ with HTTP 422 `ERR_AUDIO_CLIPPED`.
- **Corrupt Decoder Boundary**: Handles corrupted headers or unsupported containers with HTTP 400 `ERR_AUDIO_CORRUPT`.

---

## 🏷️ 10 Mandatory Acoustic Classes

| Index | Category Name | Threat Severity | Description |
|:---:|:---|:---:|:---|
| 1 | `Gunshot` | **CRITICAL** | High-energy transient impulse with reverberant shockwave |
| 2 | `Explosion` | **CRITICAL** | Sub-bass rumble shockwave (30Hz–260Hz) with long decay |
| 3 | `Glass_Break` | **HIGH** | Cascading high-frequency crystalline resonant bursts |
| 4 | `Scream_Distress` | **HIGH** | Human harmonic distress formants (800Hz–2800Hz) |
| 5 | `Siren` | **MEDIUM** | Frequency-modulated dual-tone warble (650Hz–1200Hz) |
| 6 | `Car_Alarm` | **MEDIUM** | Alternating periodic dual-pitch siren pulses |
| 7 | `Drilling_Tools` | **MEDIUM** | High-RPM motor harmonics with abrasive friction noise |
| 8 | `Dog_Bark` | **LOW** | Rhythmic acoustic bursts with pitch descent |
| 9 | `Footsteps` | **LOW** | Repetitive low-mid transient heel-toe impacts |
| 10 | `Ambient_Noise` | **NORMAL** | Diffuse pink / Brownian acoustic background floor |

---

## 🚀 Quick Start & Installation

### 1. Requirements
- Python 3.10+
- OS: Windows, Linux, or macOS

### 2. Clone & Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Verification Suite
To verify GCC-PHAT radar accuracy, error boundaries, dual-model inference, Grad-CAM generation, and fine-tuning:
```bash
python test_system.py
```

### 4. Launch the Platform
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at **`http://127.0.0.1:8000`** to access the Cyber-Security Acoustic Intelligence Dashboard.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/api/v1/audio/analyze` | Upload stereo audio for DOA radar, dual inference, Grad-CAM, and PDF generation |
| `POST` | `/api/v1/audio/simulate` | Synthesize calibrated acoustic profile for any class at target azimuth (0°–180°) |
| `POST` | `/api/v1/categories/create` | Dynamically provision new category with Python & GTM staging folders |
| `GET` | `/api/v1/categories` | List all active categories in the active learning catalog |
| `POST` | `/api/v1/train/fine-tune` | Trigger continuous transfer learning worker (background task) |
| `GET` | `/api/v1/train/status` | Poll real-time fine-tuning loss, epoch, and Macro F1 score |
| `GET` | `/api/v1/train/versions` | List historical model versions and confusion matrices |
| `GET` | `/api/v1/incidents` | Audit history of acoustic incidents |
| `GET` | `/api/v1/incidents/{id}/heatmap` | Stream Grad-CAM Mel-spectrogram PNG image |
| `GET` | `/api/v1/incidents/export-pdf/{id}` | Download official tamper-evident forensic PDF incident dossier |
| `POST` | `/api/v1/incidents/{id}/review` | Operator sign-off for Manual Review Queue (Approve / Override) |
| `WS` | `/ws/live-audio` | Real-time live monitoring WebSocket for radar telemetry broadcasting |
