"""
Acoustic Analyzer Engine.
Integrates Pitch, Perturbation (Jitter/Shimmer), Formants (LPC), and Voice Quality (HNR/CPP)
into unified forensic anomaly scoring and ML feature representations.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.analysis.base import BaseAnalysisModule
from src.ingestion.models import IngestedAudio, AudioChunk
from .config import AcousticAnalysisConfig
from .pitch import PitchAnalyzer
from .formants import FormantAnalyzer
from .voice_quality import VoiceQualityAnalyzer
from .models import (
    AcousticAnalysisResult,
    PitchAnalysisResult,
    FormantResult,
    VoiceQualityResult
)


class AcousticAnalyzer(BaseAnalysisModule):
    """
    Parallel Acoustic Analysis module evaluating acoustic properties,
    voice micro-perturbations, and vocal tract resonances to identify synthetic voices.
    """

    def __init__(self, config: Optional[AcousticAnalysisConfig] = None):
        self.config = config or AcousticAnalysisConfig()
        self.pitch_analyzer = PitchAnalyzer(config=self.config)
        self.formant_analyzer = FormantAnalyzer(config=self.config)
        self.voice_quality_analyzer = VoiceQualityAnalyzer(config=self.config)

    def analyze(self, audio: np.ndarray, sample_rate: int = 16000) -> AcousticAnalysisResult:
        """
        Run complete acoustic analysis on a 1D preprocessed mono audio array.
        """
        if audio is None or len(audio) == 0 or sample_rate <= 0:
            return AcousticAnalysisResult(
                anomaly_score=0.0,
                confidence=0.0,
                pitch=PitchAnalysisResult(),
                formants=FormantResult(),
                voice_quality=VoiceQualityResult(),
                acoustic_features={},
                synthetic_indicators=["No valid audio signal provided"]
            )

        # 1. Sub-analyses
        pitch_res = self.pitch_analyzer.analyze(audio, sample_rate)
        formant_res = self.formant_analyzer.analyze(audio, sample_rate)
        vq_res = self.voice_quality_analyzer.analyze(audio, sample_rate)

        # 2. Extract ML Feature Vector
        features = self._build_feature_vector(pitch_res, formant_res, vq_res)

        # 3. Compute Acoustic Anomaly Score and Forensic Indicators
        anomaly_score, confidence, indicators = self._evaluate_acoustic_anomalies(
            pitch_res, formant_res, vq_res, len(audio) / sample_rate
        )

        return AcousticAnalysisResult(
            anomaly_score=float(round(anomaly_score, 4)),
            confidence=float(round(confidence, 4)),
            pitch=pitch_res,
            formants=formant_res,
            voice_quality=vq_res,
            acoustic_features=features,
            synthetic_indicators=indicators
        )

    def analyze_ingested(self, ingested: IngestedAudio) -> AcousticAnalysisResult:
        """
        Analyze a complete IngestedAudio object produced by the Audio Ingestion Pipeline.
        """
        return self.analyze(ingested.processed_audio, ingested.target_sample_rate)

    def analyze_chunks(self, chunks: List[AudioChunk]) -> List[AcousticAnalysisResult]:
        """
        Run parallel/sequential acoustic analysis across sliding analysis windows for real-time risk scoring.
        """
        results = []
        for chunk in chunks:
            if chunk.contains_speech and len(chunk.samples) > 0:
                res = self.analyze(chunk.samples, chunk.sample_rate)
            else:
                # Silent/non-speech chunk
                res = AcousticAnalysisResult(
                    anomaly_score=0.0,
                    confidence=0.1,
                    pitch=PitchAnalysisResult(),
                    formants=FormantResult(),
                    voice_quality=VoiceQualityResult(),
                    acoustic_features={},
                    synthetic_indicators=["Silence / Non-speech window"]
                )
            results.append(res)
        return results

    def _build_feature_vector(
        self,
        pitch: PitchAnalysisResult,
        formants: FormantResult,
        vq: VoiceQualityResult
    ) -> Dict[str, float]:
        """
        Construct a flattened, normalized dictionary of acoustic features for ML models.
        """
        return {
            # Pitch & Intonation
            "pitch_mean_f0": pitch.mean_f0_hz,
            "pitch_std_f0": pitch.std_f0_hz,
            "pitch_min_f0": pitch.min_f0_hz,
            "pitch_max_f0": pitch.max_f0_hz,
            "pitch_voiced_fraction": pitch.voiced_fraction,
            # Micro-perturbation
            "jitter_local_pct": pitch.jitter_local_pct,
            "jitter_rap_pct": pitch.jitter_rap_pct,
            "jitter_ppq5_pct": pitch.jitter_ppq5_pct,
            "shimmer_local_pct": pitch.shimmer_local_pct,
            "shimmer_apq3_pct": pitch.shimmer_apq3_pct,
            "shimmer_apq5_pct": pitch.shimmer_apq5_pct,
            "shimmer_db": pitch.shimmer_db,
            # Formants
            "formant_f1_mean": formants.f1_mean_hz or 0.0,
            "formant_f2_mean": formants.f2_mean_hz or 0.0,
            "formant_f3_mean": formants.f3_mean_hz or 0.0,
            "formant_f4_mean": formants.f4_mean_hz or 0.0,
            "formant_dispersion": formants.formant_dispersion_hz or 0.0,
            "vocal_tract_length_cm": formants.vocal_tract_length_cm or 0.0,
            # Voice Quality
            "hnr_mean_db": vq.hnr_mean_db,
            "hnr_std_db": vq.hnr_std_db,
            "cpp_mean_db": vq.cpp_mean_db,
            "cpp_std_db": vq.cpp_std_db,
            "zcr_mean": vq.zero_crossing_rate_mean,
            "energy_entropy": vq.energy_entropy_mean,
        }

    def _evaluate_acoustic_anomalies(
        self,
        pitch: PitchAnalysisResult,
        formants: FormantResult,
        vq: VoiceQualityResult,
        duration_sec: float
    ) -> tuple[float, float, List[str]]:
        """
        Calculates acoustic anomaly risk score (0.0 to 1.0) and forensic indicators
        based on deviations from natural human acoustic distributions.
        """
        indicators = []
        anomaly_points = 0.0
        total_weights = 0.0

        # If not enough voiced frames, low confidence
        if pitch.voiced_fraction < 0.15 or pitch.mean_f0_hz <= 0:
            confidence = min(0.4, duration_sec / 3.0)
            return 0.1, float(confidence), ["Insufficient voiced phonation for deep acoustic verification"]

        confidence = float(np.clip(0.5 + pitch.voiced_fraction * 0.4 + min(0.1, duration_sec / 10.0), 0.0, 1.0))

        # 1. Jitter Anomaly Check (Neural TTS is often over-smoothed: < 0.20% or robotic: > 2.5%)
        j_min, j_max = self.config.human_jitter_normal_range
        w_jitter = 0.25
        total_weights += w_jitter

        if pitch.jitter_local_pct > 0.0:
            if pitch.jitter_local_pct < j_min:
                dev = (j_min - pitch.jitter_local_pct) / j_min
                score = min(1.0, dev * 1.2)
                anomaly_points += w_jitter * score
                indicators.append(f"Atypical pitch stability / low micro-jitter ({pitch.jitter_local_pct:.2f}% < {j_min}%) typical of neural vocoders")
            elif pitch.jitter_local_pct > j_max:
                dev = (pitch.jitter_local_pct - j_max) / j_max
                score = min(1.0, dev * 0.8)
                anomaly_points += w_jitter * score
                indicators.append(f"Excessive pitch perturbation / jitter ({pitch.jitter_local_pct:.2f}% > {j_max}%)")

        # 2. Shimmer Anomaly Check
        s_min, s_max = self.config.human_shimmer_normal_range
        w_shimmer = 0.25
        total_weights += w_shimmer

        if pitch.shimmer_local_pct > 0.0:
            if pitch.shimmer_local_pct < s_min:
                dev = (s_min - pitch.shimmer_local_pct) / s_min
                score = min(1.0, dev * 1.1)
                anomaly_points += w_shimmer * score
                indicators.append(f"Unnatural amplitude stability / low shimmer ({pitch.shimmer_local_pct:.2f}% < {s_min}%)")
            elif pitch.shimmer_local_pct > s_max:
                dev = (pitch.shimmer_local_pct - s_max) / s_max
                score = min(1.0, dev * 0.7)
                anomaly_points += w_shimmer * score
                indicators.append(f"Excessive amplitude fluctuation / shimmer ({pitch.shimmer_local_pct:.2f}% > {s_max}%)")

        # 3. Harmonics-to-Noise Ratio (HNR) Check
        hnr_min, hnr_max = self.config.human_hnr_normal_range
        w_hnr = 0.20
        total_weights += w_hnr

        if vq.hnr_mean_db > 0:
            if vq.hnr_mean_db > hnr_max:
                dev = (vq.hnr_mean_db - hnr_max) / 10.0
                score = min(1.0, dev)
                anomaly_points += w_hnr * score
                indicators.append(f"Artificially high harmonicity (HNR {vq.hnr_mean_db:.1f} dB > {hnr_max} dB)")
            elif vq.hnr_mean_db < hnr_min:
                dev = (hnr_min - vq.hnr_mean_db) / 10.0
                score = min(1.0, dev)
                anomaly_points += w_hnr * score
                indicators.append(f"Atypical aspiration noise / low HNR ({vq.hnr_mean_db:.1f} dB < {hnr_min} dB)")

        # 4. Vocal Tract Length & Formant Geometry Check
        w_formants = 0.15
        total_weights += w_formants
        if formants.vocal_tract_length_cm is not None:
            if formants.vocal_tract_length_cm < 11.0 or formants.vocal_tract_length_cm > 21.0:
                anomaly_points += w_formants * 0.75
                indicators.append(f"Physiologically atypical vocal tract length ({formants.vocal_tract_length_cm:.1f} cm)")

        # 5. Cepstral Peak Prominence (CPP) Check
        cpp_min, cpp_max = self.config.human_cpp_normal_range
        w_cpp = 0.15
        total_weights += w_cpp
        if vq.cpp_mean_db > 0:
            if vq.cpp_mean_db > cpp_max or vq.cpp_mean_db < cpp_min:
                anomaly_points += w_cpp * 0.6
                indicators.append(f"Cepstral Peak Prominence anomaly (CPP {vq.cpp_mean_db:.1f} dB)")

        final_score = anomaly_points / total_weights if total_weights > 0 else 0.0
        final_score = float(np.clip(final_score, 0.0, 1.0))

        return final_score, confidence, indicators
