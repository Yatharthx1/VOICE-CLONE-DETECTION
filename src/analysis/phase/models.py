"""
Data models for Phase Analysis outputs.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.analysis.base import BaseAnalysisResult


class InstantaneousFrequencyMetrics(BaseModel):
    """Instantaneous Frequency (IF) phase derivative distribution metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    if_mean_deviation: float = Field(0.0, description="Mean deviation of Instantaneous Frequency from STFT bin centers.")
    if_variance: float = Field(0.0, description="Variance of Instantaneous Frequency trajectory.")
    if_harmonic_clustering_score: float = Field(0.0, description="Clustering sharpness around true harmonics (0.0 to 1.0).")


class GroupDelayMetrics(BaseModel):
    """Modified Group Delay (MGD) phase representation metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mgd_peak_prominence: float = Field(0.0, description="Sharpness of vocal tract formant peaks in MGD domain.")
    phase_dispersion_entropy: float = Field(0.0, description="Entropy of phase derivative distribution.")
    unwrapped_phase_roughness: float = Field(0.0, description="Second derivative roughness of unwrapped phase.")


class PhaseAnalysisResult(BaseAnalysisResult):
    """
    Consolidated phase analysis output.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    module_name: str = Field(default="phase_analysis")
    anomaly_score: float = Field(0.0, description="Phase anomaly score from 0.0 to 1.0.")
    confidence: float = Field(0.0, description="Confidence in phase analysis.")
    instantaneous_frequency: InstantaneousFrequencyMetrics = Field(..., description="Instantaneous Frequency metrics.")
    group_delay: GroupDelayMetrics = Field(..., description="Group Delay and phase consistency metrics.")
    phase_features: Dict[str, float] = Field(default_factory=dict, description="Flattened ML feature dictionary.")
    synthetic_indicators: List[str] = Field(default_factory=list, description="Phase anomalies indicative of neural synthesis.")

    def summary(self) -> str:
        lines = [
            f"=== Phase Analysis Summary ===",
            f"• Anomaly Score: {self.anomaly_score:.3f} | Confidence: {self.confidence:.2f}",
            f"• IF Dynamics: Deviation = {self.instantaneous_frequency.if_mean_deviation:.3f} | Clustering = {self.instantaneous_frequency.if_harmonic_clustering_score:.3f}",
            f"• Group Delay: MGD Prominence = {self.group_delay.mgd_peak_prominence:.3f} | Phase Dispersion Entropy = {self.group_delay.phase_dispersion_entropy:.3f}",
            f"• Indicators: {', '.join(self.synthetic_indicators) if self.synthetic_indicators else 'None (Coherent Phonation Phase)'}"
        ]
        return "\n".join(lines)
