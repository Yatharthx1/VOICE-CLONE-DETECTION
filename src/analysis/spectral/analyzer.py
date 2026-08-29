"""
Spectral Analyzer Engine.
Evaluates spectral distribution moments, temporal flux, rolloff, and high-frequency cutoff diagnostics.
"""

from typing import Dict, List, Optional
import numpy as np

from src.analysis.base import BaseAnalysisModule
from src.ingestion.models import IngestedAudio, AudioChunk
from .config import SpectralAnalysisConfig
from .models import (
    SpectralAnalysisResult,
    SpectralMoments,
    SpectralDynamics,
    HighFrequencyAnalysis
)
from .spectrogram import SpectrogramExtractor
from .spectral_flux import SpectralDynamicsExtractor
from .high_frequency import HighFrequencyCutoffDetector


class SpectralAnalyzer(BaseAnalysisModule):
    """
    Parallel Spectral Analysis module evaluating frequency domain characteristics,
    spectral flux dynamics, and vocoder cutoff boundaries.
    """

    def __init__(self, config: Optional[SpectralAnalysisConfig] = None):
        self.config = config or SpectralAnalysisConfig()
        self.spectrogram_extractor = SpectrogramExtractor(
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels
        )
        self.dynamics_extractor = SpectralDynamicsExtractor(
            rolloff_percentile=self.config.rolloff_percentile
        )
        self.hf_detector = HighFrequencyCutoffDetector(
            cutoff_threshold_db=self.config.hf_cutoff_threshold_db
        )

    def analyze(self, audio: np.ndarray, sample_rate: int = 16000) -> SpectralAnalysisResult:
        """Run complete spectral analysis on audio array."""
        if audio is None or len(audio) == 0 or sample_rate <= 0:
            return SpectralAnalysisResult(
                anomaly_score=0.0,
                confidence=0.0,
                moments=SpectralMoments(),
                dynamics=SpectralDynamics(),
                hf_analysis=HighFrequencyAnalysis(),
                spectral_features={},
                synthetic_indicators=["No audio signal available"]
            )

        # 1. Compute STFT magnitude
        mag, _ = self.spectrogram_extractor.compute_stft(audio)

        # 2. Extract moments, dynamics, and HF cutoff
        moments = self.dynamics_extractor.extract_moments(mag, sample_rate)
        dynamics = self.dynamics_extractor.extract_dynamics(mag, sample_rate)
        hf_res = self.hf_detector.analyze(mag, sample_rate)

        # 3. Build ML feature dictionary
        features = self._build_feature_dict(moments, dynamics, hf_res)

        # 4. Compute anomaly risk score and forensic findings
        anomaly_score, confidence, indicators = self._evaluate_spectral_anomalies(
            moments, dynamics, hf_res, len(audio) / sample_rate
        )

        return SpectralAnalysisResult(
            anomaly_score=float(round(anomaly_score, 4)),
            confidence=float(round(confidence, 4)),
            moments=moments,
            dynamics=dynamics,
            hf_analysis=hf_res,
            spectral_features=features,
            synthetic_indicators=indicators
        )

    def analyze_ingested(self, ingested: IngestedAudio) -> SpectralAnalysisResult:
        """Analyze preprocessed audio from IngestedAudio object."""
        return self.analyze(ingested.processed_audio, ingested.target_sample_rate)

    def analyze_chunks(self, chunks: List[AudioChunk]) -> List[SpectralAnalysisResult]:
        """Analyze sliding analysis chunks."""
        return [self.analyze(c.samples, c.sample_rate) for c in chunks]

    def _build_feature_dict(
        self,
        moments: SpectralMoments,
        dynamics: SpectralDynamics,
        hf: HighFrequencyAnalysis
    ) -> Dict[str, float]:
        return {
            "spectral_centroid_mean": moments.centroid_mean_hz,
            "spectral_centroid_std": moments.centroid_std_hz,
            "spectral_spread_mean": moments.spread_mean_hz,
            "spectral_skewness_mean": moments.skewness_mean,
            "spectral_kurtosis_mean": moments.kurtosis_mean,
            "spectral_flux_mean": dynamics.flux_mean,
            "spectral_flux_std": dynamics.flux_std,
            "spectral_rolloff_mean": dynamics.rolloff_mean_hz,
            "spectral_flatness_mean": dynamics.flatness_mean,
            "spectral_crest_factor_mean": dynamics.crest_factor_mean,
            "spectral_hf_energy_ratio": hf.hf_energy_ratio,
            "has_artificial_cutoff": 1.0 if hf.has_artificial_cutoff else 0.0,
            "cutoff_frequency_hz": hf.cutoff_frequency_hz or 0.0,
        }

    def _evaluate_spectral_anomalies(
        self,
        moments: SpectralMoments,
        dynamics: SpectralDynamics,
        hf: HighFrequencyAnalysis,
        dur_sec: float
    ) -> tuple[float, float, List[str]]:
        indicators = []
        anomaly_points = 0.0
        total_weights = 0.0

        confidence = float(np.clip(0.6 + min(0.3, dur_sec / 5.0), 0.0, 1.0))

        # 1. High-Frequency Brickwall Cutoff Check
        w_hf = 0.35
        total_weights += w_hf
        if hf.has_artificial_cutoff:
            anomaly_points += w_hf * 1.0
            indicators.append(f"Synthetic brickwall spectral cutoff detected at {hf.cutoff_frequency_hz:.0f} Hz (typical of low-bandwidth neural TTS)")

        # 2. Spectral Flatness Check (Excessive flatness = noise/vocoder distortion)
        w_flatness = 0.25
        total_weights += w_flatness
        if dynamics.flatness_mean > self.config.human_flatness_max:
            dev = (dynamics.flatness_mean - self.config.human_flatness_max) / 0.3
            anomaly_points += w_flatness * min(1.0, dev)
            indicators.append(f"Atypical spectral flatness ({dynamics.flatness_mean:.3f} > {self.config.human_flatness_max})")

        # 3. Spectral Centroid Check
        c_min, c_max = self.config.human_centroid_range
        w_centroid = 0.20
        total_weights += w_centroid
        if moments.centroid_mean_hz > 0:
            if moments.centroid_mean_hz < c_min or moments.centroid_mean_hz > c_max:
                anomaly_points += w_centroid * 0.7
                indicators.append(f"Unnatural spectral centroid distribution ({moments.centroid_mean_hz:.0f} Hz)")

        # 4. Spectral Flux Stability Check (Over-smoothed spectra in some diffusion models)
        w_flux = 0.20
        total_weights += w_flux
        if dynamics.flux_mean < 0.005 and dur_sec > 1.0:
            anomaly_points += w_flux * 0.8
            indicators.append("Over-smoothed temporal spectral flux (lacks natural consonant-vowel transitions)")

        score = float(np.clip(anomaly_points / total_weights if total_weights > 0 else 0.0, 0.0, 1.0))
        return score, confidence, indicators
