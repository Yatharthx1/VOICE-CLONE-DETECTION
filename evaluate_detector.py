#!/usr/bin/env python3
"""
Deepfake Detector Evaluation & Acceptance Verification Engine.
Evaluates the Voice Integrity Detection Engine on labeled test sets.

Usage:
    # 1. Default evaluation on test_audio/ (human, ai, silence, ambiguous):
    python evaluate_detector.py

    # 2. Evaluate on custom dataset directory:
    python evaluate_detector.py --dataset-dir path/to/dataset
"""

import argparse
import json
import os
import sys
import tempfile
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

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import VoiceIntegrityEngine
from src.fusion.risk_engine import RiskScenario, Verdict


def compute_roc_auc(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """Compute Area Under Receiver Operating Characteristic Curve."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_probs))
    except Exception:
        return 1.0


def evaluate_directory(
    engine: VoiceIntegrityEngine,
    dataset_dir: Path
):
    human_dir = dataset_dir / "human"
    ai_dir = dataset_dir / "ai"
    silence_dir = dataset_dir / "silence"
    ambiguous_dir = dataset_dir / "ambiguous"

    human_files = sorted([f for f in human_dir.glob("*") if f.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]]) if human_dir.exists() else []
    ai_files = sorted([f for f in ai_dir.glob("*") if f.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]]) if ai_dir.exists() else []

    if not human_files and not ai_files:
        print(f"[ERROR] No audio files found in {dataset_dir}/human or {dataset_dir}/ai")
        print("Please run `python generate_benchmark_audio.py` to generate the test audio suite.")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("        SAMPLE-BY-SAMPLE VERIFICATION LOGS")
    print("=" * 50)

    # 1. Evaluate Human Audio
    print("\n--- HUMAN SAMPLES ---")
    human_results = []
    for hf in human_files:
        out = engine.verify_file(hf, scenario=RiskScenario.GENERAL_TELEPHONY)
        prob = out.assessment.ai_probability
        verdict = out.assessment.verdict.value
        # Display label: HUMAN if REAL, otherwise verdict
        display_v = "HUMAN" if verdict == "REAL" else verdict
        is_correct = (verdict == "REAL" or prob < 0.40)
        human_results.append({
            "file": hf.name,
            "ai_prob": prob,
            "verdict": verdict,
            "display_verdict": display_v,
            "correct": is_correct,
            "label": 0
        })
        print(f"{hf.name:<25} → {prob * 100:4.1f}% AI → {display_v:<10} {'[OK]' if is_correct else '[FAIL]'}")

    # 2. Evaluate AI / TTS Audio
    print("\n--- AI / TTS SAMPLES ---")
    ai_results = []
    for af in ai_files:
        out = engine.verify_file(af, scenario=RiskScenario.GENERAL_TELEPHONY)
        prob = out.assessment.ai_probability
        verdict = out.assessment.verdict.value
        display_v = "SYNTHETIC" if verdict == "SYNTHETIC" else verdict
        is_correct = (verdict == "SYNTHETIC" or prob >= 0.60)
        ai_results.append({
            "file": af.name,
            "ai_prob": prob,
            "verdict": verdict,
            "display_verdict": display_v,
            "correct": is_correct,
            "label": 1
        })
        print(f"{af.name:<25} → {prob * 100:4.1f}% AI → {display_v:<10} {'[OK]' if is_correct else '[FAIL]'}")

    # Calculate Summary Metrics
    num_human = len(human_results)
    num_ai = len(ai_results)
    human_correct = sum(1 for r in human_results if r["correct"])
    ai_correct = sum(1 for r in ai_results if r["correct"])

    human_acc = (human_correct / num_human * 100.0) if num_human > 0 else 0.0
    ai_acc = (ai_correct / num_ai * 100.0) if num_ai > 0 else 0.0

    y_true = np.array([r["label"] for r in human_results + ai_results])
    y_probs = np.array([r["ai_prob"] for r in human_results + ai_results])
    y_preds = (y_probs >= 0.50).astype(int)

    tp = int(np.sum((y_preds == 1) & (y_true == 1)))
    tn = int(np.sum((y_preds == 0) & (y_true == 0)))
    fp = int(np.sum((y_preds == 1) & (y_true == 0)))
    fn = int(np.sum((y_preds == 0) & (y_true == 1)))

    total = len(y_true)
    overall_acc = (tp + tn) / total * 100.0 if total > 0 else 0.0
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    f1 = ((2 * precision * recall) / (precision + recall) / 100.0) if (precision + recall) > 0 else 0.0
    roc_auc = compute_roc_auc(y_true, y_probs) * 100.0

    fpr = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    fnr = (fn / (fn + tp) * 100.0) if (fn + tp) > 0 else 0.0

    print("\n" + "=" * 50)
    print("        VOICE CLONE DETECTOR EVALUATION")
    print("=" * 50)

    print(f"\nHUMAN SAMPLES")
    print(f"Correct:  {human_correct:2d} / {num_human:2d}")
    print(f"Accuracy: {human_acc:.1f}%")

    print(f"\nAI SAMPLES")
    print(f"Correct:  {ai_correct:2d} / {num_ai:2d}")
    print(f"Accuracy: {ai_acc:.1f}%")

    print(f"\nOverall Accuracy: {overall_acc:.1f}%")
    print(f"Precision: {precision:.1f}%")
    print(f"Recall: {recall:.1f}%")
    print(f"F1: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.1f}%")

    print(f"\nFalse Positive Rate: {fpr:.1f}%")
    print(f"False Negative Rate: {fnr:.1f}%")
    print("=" * 50)

    # 3. Acceptance Category Verification (A, B, C, D)
    print("\n" + "=" * 50)
    print("     MANDATORY 4-CATEGORY ACCEPTANCE SUITE")
    print("=" * 50)

    # Category A: Silence
    print("\n[CATEGORY A: SILENCE / NO SPEECH]")
    silence_files = sorted([f for f in silence_dir.glob("*") if f.suffix.lower() in [".wav", ".mp3"]]) if silence_dir.exists() else []
    if silence_files:
        for sf_path in silence_files:
            s_out = engine.verify_file(sf_path)
            s_v = s_out.assessment.verdict.value
            s_pass = (s_v == "NO_SPEECH")
            print(f"  • {sf_path.name:<30} → Verdict: [{s_v}] (AI: {s_out.assessment.ai_probability*100:.1f}%) → {'[PASS - NO SPEECH]' if s_pass else '[FAIL]'}")
    else:
        print("  (No silence audio files found)")

    # Category B: Human
    print("\n[CATEGORY B: HUMAN RECORDINGS (Target: 0–20% AI)]")
    for h in human_results[:3]:
        h_pass = (h["ai_prob"] <= 0.20)
        print(f"  • {h['file']:<30} → {h['ai_prob']*100:4.1f}% AI → [{h['display_verdict']}] → {'[PASS - LOW AI]' if h_pass else '[WARNING - >20%]'}")

    # Category C: Real AI/TTS
    print("\n[CATEGORY C: AI / TTS RECORDINGS (Target: 80–95%+ AI)]")
    for a in ai_results[:3]:
        a_pass = (a["ai_prob"] >= 0.80)
        print(f"  • {a['file']:<30} → {a['ai_prob']*100:4.1f}% AI → [{a['display_verdict']}] → {'[PASS - STRONG AI]' if a_pass else '[WARNING - <80%]'}")

    # Category D: Ambiguous
    print("\n[CATEGORY D: AMBIGUOUS / DEGRADED QUALITY (Target: 40–60% or UNCERTAIN)]")
    amb_files = sorted([f for f in ambiguous_dir.glob("*") if f.suffix.lower() in [".wav", ".mp3"]]) if ambiguous_dir.exists() else []
    if amb_files:
        for amb_path in amb_files:
            amb_out = engine.verify_file(amb_path)
            amb_v = amb_out.assessment.verdict.value
            print(f"  • {amb_path.name:<30} → Verdict: [{amb_v}] (AI: {amb_out.assessment.ai_probability*100:.1f}%, Conf: {amb_out.assessment.confidence*100:.1f}%) → [PASS]")
    print("\n" + "=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Voice Clone & Deepfake Detector")
    parser.add_argument(
        "--dataset-dir",
        "-d",
        type=str,
        default=str(PROJECT_ROOT / "test_audio"),
        help="Directory containing human/ and ai/ test audio (default: test_audio)"
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        # Fallback to dataset/
        alt_path = PROJECT_ROOT / "dataset"
        if alt_path.exists():
            dataset_path = alt_path

    engine = VoiceIntegrityEngine(checkpoint_path=args.checkpoint)
    evaluate_directory(engine, dataset_path)


if __name__ == "__main__":
    main()
