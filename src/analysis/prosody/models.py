"""
Data models for Prosody Analysis outputs.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.analysis.base import BaseAnalysisResult


class RhythmMetrics(BaseModel):
    """Temporal speech rhythm, syllable timing, and pause distribution metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    syllable_count: int = Field(0, description="Estimated number of acoustic syllable nuclei.")
    speaking_rate_sps: float = Field(0.0, description="Speaking rate in syllables per second.")
    articulation_rate_sps: float = Field(0.0, description="Articulation rate excluding pauses in syllables/sec.")
    pause_count: int = Field(0, description="Number of detected inter-phrase pauses.")
    mean_pause_duration_sec: float = Field(0.0, description="Average duration of pauses in seconds.")
    npvi: float = Field(0.0, description="Normalized Pairwise Variability Index (nPVI) of vocalic intervals.")
    rpvi: float = Field(0.0, description="Raw Pairwise Variability Index (rPVI) in milliseconds.")


class IntonationMetrics(BaseModel):
    """Macro-prosodic pitch trajectory and intonation dynamics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    f0_range_semitones: float = Field(0.0, description="Pitch range in musical semitones.")
    pitch_slope_mean: float = Field(0.0, description="Average pitch contour gradient (Hz/sec).")
    pitch_slope_variance: float = Field(0.0, description="Variance of pitch trajectory slopes.")
    is_monotone: bool = Field(False, description="True if voice exhibits unnaturally flat robotic pitch.")
    pitch_direction_changes: int = Field(0, description="Number of rising/falling inflection turns.")


class ProsodyAnalysisResult(BaseAnalysisResult):
    """
    Consolidated prosodic analysis output containing rhythm, intonation,
    temporal variability indices, and prosody anomaly risk assessment.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    module_name: str = Field(default="prosody_analysis")
    anomaly_score: float = Field(0.0, description="Prosodic anomaly score from 0.0 to 1.0.")
    confidence: float = Field(0.0, description="Confidence in prosody estimation.")
    rhythm: RhythmMetrics = Field(..., description="Speech rhythm and timing metrics.")
    intonation: IntonationMetrics = Field(..., description="Pitch intonation and melody metrics.")
    prosody_features: Dict[str, float] = Field(default_factory=dict, description="Flattened ML feature dictionary.")
    synthetic_indicators: List[str] = Field(default_factory=list, description="Prosodic anomalies indicating synthetic speech.")

    def summary(self) -> str:
        lines = [
            f"=== Prosody Analysis Summary ===",
            f"• Anomaly Score: {self.anomaly_score:.3f} | Confidence: {self.confidence:.2f}",
            f"• Rhythm: {self.rhythm.speaking_rate_sps:.1f} syll/sec ({self.rhythm.syllable_count} syllables) | Articulation Rate: {self.rhythm.articulation_rate_sps:.1f} syll/sec",
            f"• Pauses: {self.rhythm.pause_count} pauses (Mean: {self.rhythm.mean_pause_duration_sec:.2f}s) | nPVI: {self.rhythm.npvi:.1f}",
            f"• Intonation: Range {self.intonation.f0_range_semitones:.1f} semitones | Monotone: {self.intonation.is_monotone} | Turns: {self.intonation.pitch_direction_changes}",
            f"• Indicators: {', '.join(self.synthetic_indicators) if self.synthetic_indicators else 'None (Natural Conversational Prosody)'}"
        ]
        return "\n".join(lines)
