"""
Synthesis Artifacts Analyzer Engine.
"""

from typing import Dict, List, Optional
import numpy as np

from src.analysis.base import BaseAnalysisModule
from src.ingestion.models import IngestedAudio, AudioChunk
from .config import SynthesisArtifactsConfig
from .models import (
    SynthesisArtifactsResult,
    NeuralVocoderArtifacts,
    ConcatenationArtifacts
)
from .neural_vocoder import NeuralVocoderDetector
from .concatenation import ConcatenationDetector


class SynthesisArtifactsAnalyzer(BaseAnalysisModule):
    """
    Parallel Synthesis Artifacts module detecting neural vocoder transposed-conv
    artifacts, harmonic smearing, and splice glitches.
    """

    def __init__(self, config: Optional[SynthesisArtifactsConfig] = None):
        self.config = config or SynthesisArtifactsConfig()
        self.vocoder_detector = NeuralVocoderDetector(config=self.config)
        self.concat_detector = ConcatenationDetector(config=self.config)

    def analyze(self, audio: np.ndarray, sample_rate: int = 16000) -> SynthesisArtifactsResult:
        """Run complete synthesis artifacts analysis."""
        if audio is None or len(audio) == 0 or sample_rate <= 0:
            return SynthesisArtifactsResult(
                anomaly_score=0.0,
                confidence=0.0,
                vocoder_artifacts=NeuralVocoderArtifacts(),
                concatenation_artifacts=ConcatenationArtifacts(),
                artifact_features={},
                synthetic_indicators=["No audio signal available"]
            )

        # 1. Detect Vocoder Artifacts
        voc_res = self.vocoder_detector.analyze(audio, sample_rate)

        # 2. Detect Concatenation / Splice Artifacts
        concat_res = self.concat_detector.analyze(audio, sample_rate)

        # 3. Build ML Feature Dict
        features = self._build_feature_dict(voc_res, concat_res)

        # 4. Evaluate Artifact Anomaly Score
        anomaly_score, confidence, indicators = self._evaluate_artifact_anomalies(
            voc_res, concat_res, len(audio) / sample_rate
        )

        return SynthesisArtifactsResult(
            anomaly_score=float(round(anomaly_score, 4)),
            confidence=float(round(confidence, 4)),
            vocoder_artifacts=voc_res,
            concatenation_artifacts=concat_res,
            artifact_features=features,
            synthetic_indicators=indicators
        )

    def analyze_ingested(self, ingested: IngestedAudio) -> SynthesisArtifactsResult:
        """Analyze preprocessed audio from IngestedAudio object."""
        return self.analyze(ingested.processed_audio, ingested.target_sample_rate)

    def analyze_chunks(self, chunks: List[AudioChunk]) -> List[SynthesisArtifactsResult]:
        """Analyze sliding analysis chunks."""
        return [self.analyze(c.samples, c.sample_rate) for c in chunks]

    def _build_feature_dict(
        self,
        voc: NeuralVocoderArtifacts,
        concat: ConcatenationArtifacts
    ) -> Dict[str, float]:
        return {
            "artifact_checkerboard_ratio": voc.checkerboard_energy_ratio,
            "artifact_periodic_detected": 1.0 if voc.periodic_artifact_detected else 0.0,
            "artifact_harmonic_smearing": voc.harmonic_smearing_score,
            "artifact_splice_points": float(concat.splice_points_detected),
            "artifact_max_energy_jump_db": concat.max_energy_jump_db,
            "artifact_max_phase_jump_rad": concat.max_phase_jump_rad,
        }

    def _evaluate_artifact_anomalies(
        self,
        voc: NeuralVocoderArtifacts,
        concat: ConcatenationArtifacts,
        dur_sec: float
    ) -> tuple[float, float, List[str]]:
        indicators = []
        anomaly_points = 0.0
        total_weights = 0.0

        confidence = float(np.clip(0.6 + min(0.3, dur_sec / 3.0), 0.0, 1.0))

        # 1. Neural Vocoder 2D Periodic Pattern
        w_voc = 0.40
        total_weights += w_voc
        if voc.periodic_artifact_detected:
            anomaly_points += w_voc * 1.0
            indicators.append("2D spectral periodicity detected (transposed-convolution upsampling artifact typical of HiFi-GAN/MelGAN)")
        elif voc.checkerboard_energy_ratio > 0.04:
            anomaly_points += w_voc * (voc.checkerboard_energy_ratio / 0.08)

        # 2. Harmonic Smearing
        w_smear = 0.30
        total_weights += w_smear
        if voc.harmonic_smearing_score > 0.6:
            anomaly_points += w_smear * voc.harmonic_smearing_score
            indicators.append(f"Excessive spectral harmonic smearing ({voc.harmonic_smearing_score:.2f})")

        # 3. Splicing Discontinuities
        w_splice = 0.30
        total_weights += w_splice
        if concat.splice_points_detected > 0:
            anomaly_points += w_splice * min(1.0, concat.splice_points_detected * 0.4)
            indicators.append(f"Suspicious audio splice discontinuities ({concat.splice_points_detected} jump boundaries)")

        score = float(np.clip(anomaly_points / total_weights if total_weights > 0 else 0.0, 0.0, 1.0))
        return score, confidence, indicators
