"""
Tests for Feature Extraction, Normalization, and ML Deepfake Detector.
"""

import numpy as np
import pytest
import torch

from src.analysis.manager import ParallelAnalysisManager
from src.features.schema import FEATURE_NAMES, NUM_FEATURES
from src.features.extractor import FeatureExtractor
from src.features.normalizer import FeatureNormalizer
from src.detector.model import ForensicAcousticDeepfakeNet
from src.detector.classifier import DeepfakeClassifier, DeepfakePrediction


def test_feature_extractor_and_normalizer(speech_wav):
    """Verify feature vector extraction and Z-score standardization."""
    import soundfile as sf
    audio, sr = sf.read(str(speech_wav), dtype="float32")

    manager = ParallelAnalysisManager()
    analysis_out = manager.analyze_audio(audio, sr)

    extractor = FeatureExtractor()
    raw_vec = extractor.extract_vector(analysis_out)

    assert len(raw_vec) == NUM_FEATURES
    assert not np.any(np.isnan(raw_vec))

    normalizer = FeatureNormalizer()
    norm_vec = normalizer.normalize(raw_vec)

    assert len(norm_vec) == NUM_FEATURES
    assert not np.any(np.isnan(norm_vec))
    assert np.all(norm_vec >= -5.0) and np.all(norm_vec <= 5.0)


def test_ml_deepfake_model_forward():
    """Verify PyTorch neural network forward pass."""
    model = ForensicAcousticDeepfakeNet(input_dim=NUM_FEATURES)
    dummy_input = torch.randn(4, NUM_FEATURES)
    out = model(dummy_input)

    assert out.shape == (4, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_deepfake_classifier_inference(speech_wav):
    """Verify complete DeepfakeClassifier inference on extracted features."""
    import soundfile as sf
    audio, sr = sf.read(str(speech_wav), dtype="float32")

    manager = ParallelAnalysisManager()
    analysis_out = manager.analyze_audio(audio, sr)

    extractor = FeatureExtractor()
    raw_vec = extractor.extract_vector(analysis_out)

    classifier = DeepfakeClassifier()
    pred = classifier.predict(raw_vec)

    assert isinstance(pred, DeepfakePrediction)
    assert 0.0 <= pred.synthetic_probability <= 1.0
    assert 0.0 <= pred.confidence <= 1.0
    assert isinstance(pred.is_synthetic, bool)
    assert len(pred.top_contributing_features) > 0
