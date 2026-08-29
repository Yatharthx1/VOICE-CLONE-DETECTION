"""
Phase Analyzer Engine.
Evaluates Instantaneous Frequency, Group Delay, and Phase Inconsistency metrics.
"""

from typing import Dict, List, Optional
import numpy as np

from src.analysis.base import BaseAnalysisModule
from src.analysis.spectral.spectrogram import SpectrogramExtractor
from src.ingestion.models import IngestedAudio, AudioChunk
from .config import PhaseAnalysisConfig
from .models import (
    PhaseAnalysisResult,
    InstantaneousFrequencyMetrics,
    GroupDelayMetrics
)
from .instantaneous_freq import InstantaneousFrequencyTracker
from .phase_consistency import PhaseConsistencyAnalyzer


class PhaseAnalyzer(BaseAnalysisModule):
    """
    Parallel Phase Analysis module measuring phase coherence, Instantaneous Frequency (IF)
    trajectories, and Modified Group Delay (MGD) representations.
    """

    def __init__(self, config: Optional[PhaseAnalysisConfig] = None):
        self.config = config or PhaseAnalysisConfig()
        self.if_tracker = InstantaneousFrequencyTracker(config=self.config)
        self.phase_analyzer = PhaseConsistencyAnalyzer(config=self.config)
        self.spec_extractor = SpectrogramExtractor(
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )

    def analyze(self, audio: np.ndarray, sample_rate: int = 16000) -> PhaseAnalysisResult:
        """Run complete phase analysis."""
        if audio is None or len(audio) == 0 or sample_rate <= 0:
            return PhaseAnalysisResult(
                anomaly_score=0.0,
                confidence=0.0,
                instantaneous_frequency=InstantaneousFrequencyMetrics(),
                group_delay=GroupDelayMetrics(),
                phase_features={},
                synthetic_indicators=["No audio signal available"]
            )

        # 1. Compute complex STFT
        _, complex_stft = self.spec_extractor.compute_stft(audio)

        # 2. IF Analysis
        if_res = self.if_tracker.analyze(complex_stft, sample_rate)

        # 3. Group Delay & Consistency
        gd_res = self.phase_analyzer.analyze(audio, sample_rate)

        # 4. Build ML Feature Dict
        features = self._build_feature_dict(if_res, gd_res)

        # 5. Evaluate Phase Anomalies
        anomaly_score, confidence, indicators = self._evaluate_phase_anomalies(
            if_res, gd_res, len(audio) / sample_rate
        )

        return PhaseAnalysisResult(
            anomaly_score=float(round(anomaly_score, 4)),
            confidence=float(round(confidence, 4)),
            instantaneous_frequency=if_res,
            group_delay=gd_res,
            phase_features=features,
            synthetic_indicators=indicators
        )

    def analyze_ingested(self, ingested: IngestedAudio) -> PhaseAnalysisResult:
        """Analyze preprocessed audio from IngestedAudio object."""
        return self.analyze(ingested.processed_audio, ingested.target_sample_rate)

    def analyze_chunks(self, chunks: List[AudioChunk]) -> List[PhaseAnalysisResult]:
        """Analyze sliding analysis chunks."""
        return [self.analyze(c.samples, c.sample_rate) for c in chunks]

    def _build_feature_dict(
        self,
        if_res: InstantaneousFrequencyMetrics,
        gd: GroupDelayMetrics
    ) -> Dict[str, float]:
        return {
            "phase_if_mean_deviation": if_res.if_mean_deviation,
            "phase_if_variance": if_res.if_variance,
            "phase_if_clustering": if_res.if_harmonic_clustering_score,
            "phase_mgd_peak_prominence": gd.mgd_peak_prominence,
            "phase_dispersion_entropy": gd.phase_dispersion_entropy,
            "phase_roughness": gd.unwrapped_phase_roughness,
        }

    def _evaluate_phase_anomalies(
        self,
        if_res: InstantaneousFrequencyMetrics,
        gd: GroupDelayMetrics,
        dur_sec: float
    ) -> tuple[float, float, List[str]]:
        indicators = []
        anomaly_points = 0.0
        total_weights = 0.0

        confidence = float(np.clip(0.6 + min(0.3, dur_sec / 3.0), 0.0, 1.0))

        # 1. Harmonic Phase Clustering Failure
        w_clust = 0.40
        total_weights += w_clust
        if if_res.if_harmonic_clustering_score < 0.25 and dur_sec > 1.0:
            anomaly_points += w_clust * (1.0 - if_res.if_harmonic_clustering_score)
            indicators.append(f"Atypical Instantaneous Frequency dispersion / poor phase coherence ({if_res.if_harmonic_clustering_score:.2f})")

        # 2. Phase Dispersion Entropy
        w_entropy = 0.30
        total_weights += w_entropy
        if gd.phase_dispersion_entropy > 0.85:
            anomaly_points += w_entropy * 0.8
            indicators.append(f"High phase derivative dispersion entropy ({gd.phase_dispersion_entropy:.2f}) typical of Griffin-Lim / vocoder phase noise")

        # 3. High Unwrapped Phase Roughness
        w_rough = 0.30
        total_weights += w_rough
        if gd.unwrapped_phase_roughness > 3.0:
            anomaly_points += w_rough * 0.7
            indicators.append(f"Irregular unwrapped phase second derivative ({gd.unwrapped_phase_roughness:.1f})")

        score = float(np.clip(anomaly_points / total_weights if total_weights > 0 else 0.0, 0.0, 1.0))
        return score, confidence, indicators
