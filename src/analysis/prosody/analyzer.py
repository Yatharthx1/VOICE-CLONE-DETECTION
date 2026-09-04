"""
Prosody Analyzer Engine.
Evaluates rhythm, speaking rate, intonation dynamism, and monotonic speech indicators.
"""

from typing import Dict, List, Optional
import numpy as np

from ..base import BaseAnalysisModule
from ..acoustic.pitch import PitchAnalyzer
from ...ingestion.models import IngestedAudio, AudioChunk
from .config import ProsodyAnalysisConfig
from .models import (
    ProsodyAnalysisResult,
    RhythmMetrics,
    IntonationMetrics
)
from .rhythm import RhythmAnalyzer
from .intonation import IntonationAnalyzer


class ProsodyAnalyzer(BaseAnalysisModule):
    """
    Parallel Prosodic Analysis module evaluating rhythmic regularity, speaking rate,
    syllabic timing variability (nPVI), and intonation contours.
    """

    def __init__(self, config: Optional[ProsodyAnalysisConfig] = None):
        self.config = config or ProsodyAnalysisConfig()
        self.rhythm_analyzer = RhythmAnalyzer(config=self.config)
        self.intonation_analyzer = IntonationAnalyzer(config=self.config)
        self._pitch_helper = PitchAnalyzer()

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        speech_duration_sec: Optional[float] = None,
        f0_contour: Optional[np.ndarray] = None
    ) -> ProsodyAnalysisResult:
        """Run prosody analysis on audio array."""
        if audio is None or len(audio) == 0 or sample_rate <= 0 or np.max(np.abs(audio)) < 0.015:
            return ProsodyAnalysisResult(
                anomaly_score=0.0,
                confidence=0.0,
                rhythm=RhythmMetrics(),
                intonation=IntonationMetrics(),
                prosody_features={},
                synthetic_indicators=[]
            )

        total_dur = len(audio) / sample_rate
        sp_dur = speech_duration_sec if speech_duration_sec is not None else total_dur

        # 1. Analyze Rhythm
        rhythm_res = self.rhythm_analyzer.analyze(audio, sample_rate, sp_dur)

        # 2. Extract F0 contour if not provided
        if f0_contour is None:
            p_res = self._pitch_helper.analyze(audio, sample_rate)
            f0_contour = p_res.f0_contour

        # 3. Analyze Intonation
        intonation_res = self.intonation_analyzer.analyze(f0_contour)

        # 4. Build ML Feature Vector
        features = self._build_feature_dict(rhythm_res, intonation_res)

        # 5. Evaluate Prosodic Anomalies
        anomaly_score, confidence, indicators = self._evaluate_prosody_anomalies(
            rhythm_res, intonation_res, total_dur
        )

        return ProsodyAnalysisResult(
            anomaly_score=float(round(anomaly_score, 4)),
            confidence=float(round(confidence, 4)),
            rhythm=rhythm_res,
            intonation=intonation_res,
            prosody_features=features,
            synthetic_indicators=indicators
        )

    def analyze_ingested(self, ingested: IngestedAudio) -> ProsodyAnalysisResult:
        """Analyze preprocessed audio from an IngestedAudio object."""
        return self.analyze(
            audio=ingested.processed_audio,
            sample_rate=ingested.target_sample_rate,
            speech_duration_sec=ingested.speech_duration_sec
        )

    def analyze_chunks(self, chunks: List[AudioChunk]) -> List[ProsodyAnalysisResult]:
        """Analyze sliding analysis chunks."""
        return [self.analyze(c.samples, c.sample_rate) for c in chunks]

    def _build_feature_dict(
        self,
        rhythm: RhythmMetrics,
        intonation: IntonationMetrics
    ) -> Dict[str, float]:
        return {
            "prosody_speaking_rate": rhythm.speaking_rate_sps,
            "prosody_articulation_rate": rhythm.articulation_rate_sps,
            "prosody_syllable_count": float(rhythm.syllable_count),
            "prosody_pause_count": float(rhythm.pause_count),
            "prosody_mean_pause_sec": rhythm.mean_pause_duration_sec,
            "prosody_npvi": rhythm.npvi,
            "prosody_rpvi": rhythm.rpvi,
            "prosody_f0_range_semitones": intonation.f0_range_semitones,
            "prosody_pitch_slope_variance": intonation.pitch_slope_variance,
            "prosody_is_monotone": 1.0 if intonation.is_monotone else 0.0,
            "prosody_direction_changes": float(intonation.pitch_direction_changes),
        }

    def _evaluate_prosody_anomalies(
        self,
        rhythm: RhythmMetrics,
        intonation: IntonationMetrics,
        dur_sec: float
    ) -> tuple[float, float, List[str]]:
        if rhythm.syllable_count == 0 or intonation.f0_range_semitones == 0:
            return 0.0, 0.0, []

        indicators = []
        anomaly_points = 0.0
        total_weights = 0.0

        confidence = float(np.clip(0.5 + min(0.4, dur_sec / 4.0), 0.0, 1.0))

        # 1. Monotone Robot Voice Check (Synthesized voice with almost 0 pitch inflection across > 2.0s)
        w_monotone = 0.60
        total_weights += w_monotone
        if intonation.is_monotone and dur_sec >= 2.0 and intonation.f0_range_semitones < 1.0:
            anomaly_points += w_monotone * 0.9
            indicators.append(f"Unnaturally flat intonation curve (F0 range: {intonation.f0_range_semitones:.1f} semitones)")

        # 2. Strict Mechanical Syllabic Timing Check (Only over long sustained robotic speech > 5s and >= 10 syllables with extreme metronome rhythm)
        w_pvi = 0.40
        total_weights += w_pvi
        if dur_sec >= 5.0 and rhythm.syllable_count >= 10 and 0.0 < rhythm.npvi < 2.0:
            anomaly_points += w_pvi * 0.8
            indicators.append(f"Metronomic syllabic rhythm (Low nPVI: {rhythm.npvi:.1f} < 2.0)")

        score = float(np.clip(anomaly_points / total_weights if total_weights > 0 else 0.0, 0.0, 1.0))
        return score, confidence, indicators
