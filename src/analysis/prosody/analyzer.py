"""
Prosody Analyzer Engine.
Evaluates rhythm, speaking rate, intonation dynamism, and monotonic speech indicators.
"""

from typing import Dict, List, Optional
import numpy as np

from src.analysis.base import BaseAnalysisModule
from src.analysis.acoustic.pitch import PitchAnalyzer
from src.ingestion.models import IngestedAudio, AudioChunk
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
        f0_contour: Optional[List[float]] = None,
        speech_duration_sec: Optional[float] = None
    ) -> ProsodyAnalysisResult:
        """Run prosody analysis on audio array."""
        if audio is None or len(audio) == 0 or sample_rate <= 0:
            return ProsodyAnalysisResult(
                anomaly_score=0.0,
                confidence=0.0,
                rhythm=RhythmMetrics(),
                intonation=IntonationMetrics(),
                prosody_features={},
                synthetic_indicators=["No audio signal available"]
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
        indicators = []
        anomaly_points = 0.0
        total_weights = 0.0

        confidence = float(np.clip(0.5 + min(0.4, dur_sec / 4.0), 0.0, 1.0))

        # 1. Monotone Robot Voice Check
        w_monotone = 0.35
        total_weights += w_monotone
        if intonation.is_monotone:
            anomaly_points += w_monotone * 0.9
            indicators.append(f"Unnaturally flat intonation curve (F0 range: {intonation.f0_range_semitones:.1f} semitones)")

        # 2. Speaking Rate Extreme Check
        sr_min, sr_max = self.config.human_speaking_rate_range
        w_rate = 0.25
        total_weights += w_rate
        if rhythm.speaking_rate_sps > 0:
            if rhythm.speaking_rate_sps < sr_min or rhythm.speaking_rate_sps > sr_max:
                anomaly_points += w_rate * 0.7
                indicators.append(f"Atypical speaking rate ({rhythm.speaking_rate_sps:.1f} syllables/sec)")

        # 3. Rhythm Invariability Check (nPVI too low = mechanical metronomic timing)
        npvi_min, npvi_max = self.config.human_npvi_range
        w_pvi = 0.25
        total_weights += w_pvi
        if rhythm.npvi > 0:
            if rhythm.npvi < npvi_min:
                anomaly_points += w_pvi * 0.8
                indicators.append(f"Metronomic syllabic rhythm (Low nPVI: {rhythm.npvi:.1f} < {npvi_min})")
            elif rhythm.npvi > npvi_max:
                anomaly_points += w_pvi * 0.6
                indicators.append(f"Erratic syllabic rhythm (High nPVI: {rhythm.npvi:.1f} > {npvi_max})")

        # 4. Lack of Pitch Inflections
        w_turns = 0.15
        total_weights += w_turns
        if dur_sec >= 2.0 and intonation.pitch_direction_changes <= 1:
            anomaly_points += w_turns * 0.7
            indicators.append("Deficiency in conversational pitch direction shifts (static pitch declination)")

        score = float(np.clip(anomaly_points / total_weights if total_weights > 0 else 0.0, 0.0, 1.0))
        return score, confidence, indicators
