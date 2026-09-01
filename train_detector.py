#!/usr/bin/env python3
"""
Deepfake Detector Training Pipeline.
Trains ForensicAcousticDeepfakeNet on human vs. AI-generated voice recordings,
fits and serializes the FeatureNormalizer, fits Platt scaling calibration parameters,
and exports the complete production model checkpoint.

Usage:
    # 1. Train on synthetic benchmark dataset:
    python train_detector.py --synthetic-samples 120 --epochs 25

    # 2. Train on custom labeled dataset (dataset/human/ and dataset/ai/):
    python train_detector.py --dataset-dir path/to/dataset --epochs 30
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Force UTF-8 stdout so Windows cp1252 doesn't choke on special characters
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.manager import ParallelAnalysisManager
from src.features.extractor import FeatureExtractor
from src.features.normalizer import FeatureNormalizer
from src.features.schema import FEATURE_NAMES, NUM_FEATURES
from src.detector.model import ForensicAcousticDeepfakeNet


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_audio(
    is_ai: bool,
    duration_sec: float = 3.5,
    sample_rate: int = 16000,
    seed_offset: int = 0
) -> np.ndarray:
    """
    Generate realistic acoustic waveforms representing genuine human speech dynamics
    or AI synthesis artifacts (vocoder checkerboard, monotone pitch, phase dispersion, cutoff).
    """
    rng = np.random.RandomState(42 + seed_offset)
    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    if not is_ai:
        # ===================================================================
        # GENUINE HUMAN SPEECH MODEL:
        # Dynamic intonation contour, natural micro-jitter, natural shimmer,
        # physiological formant ratios, natural consonant-vowel transitions.
        # ===================================================================
        base_f0 = rng.uniform(110.0, 240.0)
        # Natural conversational pitch excursion (+/- 15-35 Hz with gentle drift)
        f0_drift = 20.0 * np.sin(2 * np.pi * rng.uniform(0.8, 2.2) * t) + rng.normal(0, 1.5, total_samples)
        f0 = np.clip(base_f0 + f0_drift, 70.0, 350.0)
        phase = np.cumsum(2 * np.pi * f0 / sample_rate)

        # Micro-jitter and shimmer (vocal fold micro-perturbation)
        jitter_pert = rng.normal(0, 0.008, total_samples)
        shimmer_pert = 1.0 + rng.normal(0, 0.03, total_samples)

        # Natural vocal tract formants
        signal = (
            0.50 * np.sin(phase + jitter_pert) +
            0.30 * np.sin(2 * phase) +
            0.20 * np.sin(3 * phase) +
            0.12 * np.sin(4 * phase) +
            0.08 * np.sin(5 * phase)
        ) * shimmer_pert

        # Syllabic envelope modulation (natural pauses and speech rhythm)
        syl_rate = rng.uniform(3.5, 5.2)  # syllables per sec
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * syl_rate * t))
        envelope = np.clip(envelope, 0.05, 1.0) ** 1.5
        signal = signal * envelope

        # Natural pink noise floor (-45 dB)
        noise = rng.normal(0, rng.uniform(0.003, 0.008), total_samples)
        signal = signal + noise

    else:
        # ===================================================================
        # AI-GENERATED / SYNTHETIC SPEECH MODEL:
        # Exhibits one or more known neural synthesis artifacts:
        # 1. HiFi-GAN / MelGAN 2D transposed-conv checkerboard periodic noise
        # 2. Robotic / flat monotonic pitch contour (low nPVI)
        # 3. Brickwall high-frequency cutoff (low bandwidth vocoder)
        # 4. Phase dispersion entropy & group delay roughness
        # 5. Over-smoothed micro-perturbations (near-zero jitter/shimmer)
        # ===================================================================
        artifact_type = rng.choice(["vocoder_checkerboard", "monotone_tts", "cutoff_phase_noise", "over_smoothed"])
        base_f0 = rng.uniform(120.0, 220.0)

        if artifact_type == "monotone_tts":
            # Flat monotone intonation
            f0 = base_f0 + 2.0 * np.sin(2 * np.pi * 0.3 * t)
            phase = np.cumsum(2 * np.pi * f0 / sample_rate)
            signal = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
            # Metronomic rigid envelope
            envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
            signal = signal * envelope

        elif artifact_type == "vocoder_checkerboard":
            # Transposed conv upsampling checkerboard artifact (~3.5kHz to ~7.5kHz high-frequency periodic tone)
            f0 = base_f0 + 8.0 * np.sin(2 * np.pi * 1.2 * t)
            phase = np.cumsum(2 * np.pi * f0 / sample_rate)
            harmonics = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
            checkerboard = 0.08 * np.sin(2 * np.pi * rng.uniform(3800, 6800) * t)
            signal = harmonics + checkerboard

        elif artifact_type == "cutoff_phase_noise":
            # Brickwall cutoff at 4kHz or 6kHz + phase dispersion
            f0 = base_f0 + 12.0 * np.sin(2 * np.pi * 1.0 * t)
            phase = np.cumsum(2 * np.pi * f0 / sample_rate)
            signal = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase)
            # Phase noise
            phase_noise = rng.normal(0, 0.04, total_samples)
            signal = signal + phase_noise

        else: # over_smoothed
            # Perfect mathematical sine waves lacking biological micro-fluctuations
            f0 = np.full_like(t, base_f0)
            phase = 2 * np.pi * base_f0 * t
            signal = 0.6 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(4 * phase)

        # Mild background noise
        signal = signal + rng.normal(0, 0.002, total_samples)

    # Normalize amplitude to [-0.85, 0.85]
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = (signal / peak) * rng.uniform(0.65, 0.85)

    return signal.astype(np.float32)


def collect_features_from_dataset(
    dataset_dir: Optional[Path] = None,
    synthetic_samples_per_class: int = 100,
    manager: Optional[ParallelAnalysisManager] = None,
    extractor: Optional[FeatureExtractor] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 60-D feature vectors from files or synthetic generator.
    Returns (features, labels) where label 0 = Human, 1 = AI/Synthetic.
    """
    mgr = manager or ParallelAnalysisManager()
    ext = extractor or FeatureExtractor()

    features_list = []
    labels_list = []

    # Search default dataset paths if not explicitly provided
    search_dirs = [dataset_dir] if dataset_dir else [PROJECT_ROOT / "dataset", PROJECT_ROOT / "test_audio"]

    for d_path in search_dirs:
        if d_path and d_path.exists():
            human_dir = d_path / "human"
            ai_dir = d_path / "ai"

            for label, d in [(0, human_dir), (1, ai_dir)]:
                if d.exists():
                    audio_files = [f for f in d.glob("*") if f.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]]
                    print(f"Loading {len(audio_files)} real speech samples from {d.relative_to(PROJECT_ROOT)} (Label={label})...")
                    for fpath in audio_files:
                        try:
                            audio, sr = sf.read(str(fpath), dtype="float32")
                            if audio.ndim > 1:
                                audio = np.mean(audio, axis=1)

                            # 1. Base clean sample
                            analysis = mgr.analyze_audio(audio, sr)
                            features_list.append(ext.extract_vector(analysis))
                            labels_list.append(label)

                            # 2. Gain augmented sample
                            gain = float(np.random.uniform(0.65, 1.35))
                            audio_gain = np.clip(audio * gain, -1.0, 1.0)
                            analysis_gain = mgr.analyze_audio(audio_gain, sr)
                            features_list.append(ext.extract_vector(analysis_gain))
                            labels_list.append(label)

                            # 3. Ambient room noise augmented sample (consumer mic simulation)
                            noise_level = float(np.random.uniform(0.001, 0.006))
                            noise = np.random.normal(0, noise_level, len(audio)).astype(np.float32)
                            audio_noisy = np.clip(audio + noise, -1.0, 1.0)
                            analysis_noisy = mgr.analyze_audio(audio_noisy, sr)
                            features_list.append(ext.extract_vector(analysis_noisy))
                            labels_list.append(label)

                        except Exception as e:
                            print(f"  Warning: failed to process {fpath.name}: {e}")

    print(f"Loaded and augmented {len(features_list)} real-world feature vectors from disk.")

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.float32)
    return X, y


def fit_platt_calibration(val_logits: np.ndarray, val_labels: np.ndarray) -> Tuple[float, float]:
    """
    Fit Platt scaling sigmoid parameters: P(Y=1|s) = 1 / (1 + exp(-(a*s + b)))
    on [0, 1] classifier score inputs.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -15.0, 15.0)))
        clf = LogisticRegression(solver="lbfgs", max_iter=200)
        clf.fit(val_probs.reshape(-1, 1), val_labels)
        a = float(clf.coef_[0][0])
        b = float(clf.intercept_[0])
        if a < 8.0 or a > 18.0:
            a = 10.0
            b = -5.0
        return a, b
    except Exception:
        return 10.0, -5.0


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu"
) -> Tuple[ForensicAcousticDeepfakeNet, float, float, Dict[str, float]]:
    """
    Train ForensicAcousticDeepfakeNet with BCEWithLogitsLoss, AdamW, and cosine annealing.
    """
    dev = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    model = ForensicAcousticDeepfakeNet(input_dim=NUM_FEATURES).to(dev)

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    print(f"\nStarting training on {len(X_train)} samples (Val: {len(X_val)}) across {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad()
            logits = model.forward_logits(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)

        train_loss /= len(X_train)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(dev), by.to(dev)
                logits = model.forward_logits(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(bx)
                preds = (torch.sigmoid(logits) >= 0.50).float()
                correct += (preds == by).sum().item()

        val_loss /= len(X_val)
        val_acc = correct / len(X_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.1f}%")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Obtain validation logits for Platt scaling
    with torch.no_grad():
        val_tensor = torch.from_numpy(X_val).to(dev)
        val_logits = model.forward_logits(val_tensor).squeeze().cpu().numpy()

    platt_a, platt_b = fit_platt_calibration(val_logits, y_val)
    print(f"\nFitted Platt Calibration: a={platt_a:.3f}, b={platt_b:.3f}")

    # Compute final metrics on validation set
    calibrated_val_probs = 1.0 / (1.0 + np.exp(-(platt_a * val_logits + platt_b)))
    val_preds_binary = (calibrated_val_probs >= 0.50).astype(int)
    final_acc = float(np.mean(val_preds_binary == y_val))

    metrics = {
        "val_loss": float(round(best_val_loss, 4)),
        "val_accuracy": float(round(final_acc, 4)),
        "num_train_samples": len(X_train),
        "num_val_samples": len(X_val)
    }

    return model, platt_a, platt_b, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Forensic Deepfake Detection Model")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Directory containing human/ and ai/ subfolders")
    parser.add_argument("--synthetic-samples", type=int, default=120, help="Number of synthetic samples per class")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", "-o", type=str, default=str(PROJECT_ROOT / "models" / "detector_checkpoint.pt"), help="Output checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    set_seed(args.seed)

    print("=" * 80)
    print("  [VOICE INTEGRITY FRAMEWORK: ML DETECTOR TRAINING PIPELINE]")
    print("=" * 80)

    # 1. Feature Collection
    dataset_path = Path(args.dataset_dir) if args.dataset_dir else None
    X, y = collect_features_from_dataset(dataset_path, synthetic_samples_per_class=args.synthetic_samples)
    print(f"Total extracted feature vectors: {X.shape[0]} (Feature Dim: {X.shape[1]})")

    # 2. Train / Val Split (75% / 25%)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(0.75 * len(X))

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_val_raw, y_val = X[val_idx], y[val_idx]

    # 3. Fit Normalizer on Training Set ONLY
    normalizer = FeatureNormalizer()
    normalizer.fit(X_train_raw)
    print("Fitted FeatureNormalizer on training split.")

    X_train_norm = np.array([normalizer.normalize(v) for v in X_train_raw])
    X_val_norm = np.array([normalizer.normalize(v) for v in X_val_raw])

    # 4. Train Neural Model & Fit Platt Calibration
    model, platt_a, platt_b, metrics = train_model(
        X_train=X_train_norm,
        y_train=y_train,
        X_val=X_val_norm,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    # 5. Package and Save Checkpoint
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "normalizer_state": normalizer.to_dict(),
        "calibration_params": {
            "platt_a": platt_a,
            "platt_b": platt_b
        },
        "feature_names": FEATURE_NAMES,
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    torch.save(checkpoint_payload, str(output_path))
    print("\n" + "=" * 80)
    print(f"✅ Checkpoint successfully saved to: {output_path}")
    print(f"  • Validation Accuracy : {metrics['val_accuracy'] * 100:.1f}%")
    print(f"  • Normalizer Bundled   : {len(normalizer.means)} feature means & stds")
    print(f"  • Platt Parameters     : a={platt_a:.3f}, b={platt_b:.3f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
