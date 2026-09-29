"""Data Augmentation Engine for SonicSentinel AI.
Automatically multiplies small datasets (e.g., 100 samples) into large-scale datasets (3,000+ samples)
using pitch shifting, time stretching, and ambient noise injection.
"""
import os
import numpy as np
import scipy.signal
import soundfile as sf
from pathlib import Path
from src.config import DATASET_DIR, SAMPLE_RATE

def apply_noise_injection(signal: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    """Injects random Gaussian noise."""
    noise = np.random.randn(len(signal))
    return signal + noise_factor * noise

def apply_time_shift(signal: np.ndarray, shift_max: float = 0.2, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Shifts the audio signal randomly in time."""
    shift = np.random.randint(sr * shift_max)
    direction = np.random.choice(['right', 'left'])
    if direction == 'right':
        return np.roll(signal, shift)
    else:
        return np.roll(signal, -shift)

def apply_pitch_shift_simulation(signal: np.ndarray, rate: float = 1.1) -> np.ndarray:
    """Simulates pitch shifting via fast/slow resampling."""
    num_samples = int(len(signal) / rate)
    return scipy.signal.resample(signal, num_samples)

def harvest_and_augment_category(category_name: str, target_count: int = 3000):
    """
    Takes existing samples in a category and augments them until the target_count is reached.
    If 100 samples exist, it generates 2900 new variations.
    """
    cat_dir = DATASET_DIR / "train" / category_name.replace(" ", "_")
    if not cat_dir.exists():
        cat_dir.mkdir(parents=True, exist_ok=True)
    
    existing_files = list(cat_dir.glob("*.wav"))
    current_count = len(existing_files)
    
    if current_count == 0:
        return {"status": "error", "message": f"No base samples found in {category_name} to augment."}
    
    if current_count >= target_count:
        return {"status": "success", "message": f"{category_name} already has {current_count} samples."}
        
    needed = target_count - current_count
    print(f"[*] Augmenting {category_name}: {current_count} -> {target_count} ({needed} needed)")
    
    generated = 0
    while generated < needed:
        for base_file in existing_files:
            if generated >= needed:
                break
                
            try:
                data, sr = sf.read(str(base_file))
                if data.ndim > 1:
                    data = data[:, 0]
                
                # Pick a random augmentation strategy
                strategy = np.random.choice(['noise', 'shift', 'pitch', 'combined'])
                
                if strategy == 'noise':
                    aug_data = apply_noise_injection(data, np.random.uniform(0.001, 0.01))
                elif strategy == 'shift':
                    aug_data = apply_time_shift(data)
                elif strategy == 'pitch':
                    aug_data = apply_pitch_shift_simulation(data, np.random.uniform(0.9, 1.1))
                else:
                    aug_data = apply_noise_injection(apply_time_shift(data), 0.005)
                
                out_path = cat_dir / f"aug_{strategy}_{generated}_{base_file.name}"
                
                # Ensure original length
                if len(aug_data) > len(data):
                    aug_data = aug_data[:len(data)]
                elif len(aug_data) < len(data):
                    aug_data = np.pad(aug_data, (0, len(data) - len(aug_data)))
                    
                sf.write(str(out_path), aug_data, sr)
                generated += 1
            except Exception as e:
                continue
                
    return {"status": "success", "message": f"Successfully augmented {category_name} to {target_count} samples."}
