"""
Data models for Spectral Analysis outputs.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from ..base import BaseAnalysisResult


class SpectralMoments(BaseModel):
    """Spectral distribution statistical moments."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    centroid_mean_hz: float = Field(0.0, description="Mean spectral center of mass in Hz.")
    centroid_std_hz: float = Field(0.0, description="Standard deviation of spectral centroid.")
    spread_mean_hz: float = Field(0.0, description="Spectral spread / bandwidth in Hz.")
    skewness_mean: float = Field(0.0, description="Spectral skewness (asymmetry of spectral distribution).")
    kurtosis_mean: float = Field(0.0, description="Spectral kurtosis (peakedness/flatness of spectrum).")


class SpectralDynamics(BaseModel):
    """Temporal spectral flux and energy distribution metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    flux_mean: float = Field(0.0, description="Mean spectral flux (rate of spectral change between frames).")
    flux_std: float = Field(0.0, description="Standard deviation of spectral flux.")
    rolloff_mean_hz: float = Field(0.0, description="Mean spectral rolloff frequency (85% energy point) in Hz.")
    flatness_mean: float = Field(0.0, description="Spectral flatness (ratio of geometric to arithmetic mean of spectrum).")
    crest_factor_mean: float = Field(0.0, description="Mean spectral crest factor (peak to average ratio).")


class HighFrequencyAnalysis(BaseModel):
    """Detection of synthetic vocoder high-frequency cutoffs and brickwall filters."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    has_artificial_cutoff: bool = Field(False, description="True if sharp artificial high-frequency cutoff is detected.")
    cutoff_frequency_hz: Optional[float] = Field(None, description="Detected cutoff frequency in Hz if present.")
    hf_energy_ratio: float = Field(0.0, description="Fraction of total spectral energy situated above 4000 Hz.")


class SpectralAnalysisResult(BaseAnalysisResult):
    """
    Consolidated spectral analysis output containing spectral moments,
    dynamics, high-frequency cutoff diagnostics, and anomaly risk assessment.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    module_name: str = Field(default="spectral_analysis")
    anomaly_score: float = Field(0.0, description="Spectral anomaly score from 0.0 to 1.0.")
    confidence: float = Field(0.0, description="Confidence in spectral evaluation.")
    moments: SpectralMoments = Field(..., description="Statistical spectral moments.")
    dynamics: SpectralDynamics = Field(..., description="Spectral flux and rolloff dynamics.")
    hf_analysis: HighFrequencyAnalysis = Field(..., description="High-frequency cutoff and vocoder artifacts.")
    spectral_features: Dict[str, float] = Field(default_factory=dict, description="Flattened ML feature dictionary.")
    synthetic_indicators: List[str] = Field(default_factory=list, description="Forensic spectral anomalies detected.")

    def summary(self) -> str:
        lines = [
            f"=== Spectral Analysis Summary ===",
            f"• Anomaly Score: {self.anomaly_score:.3f} | Confidence: {self.confidence:.2f}",
            f"• Moments: Centroid {self.moments.centroid_mean_hz:.0f} Hz (±{self.moments.centroid_std_hz:.0f}Hz) | Spread {self.moments.spread_mean_hz:.0f} Hz",
            f"• Dynamics: Flux {self.dynamics.flux_mean:.3f} | Rolloff {self.dynamics.rolloff_mean_hz:.0f} Hz | Flatness {self.dynamics.flatness_mean:.4f}",
            f"• High-Frequency: Artificial Cutoff = {self.hf_analysis.has_artificial_cutoff} (Cutoff: {self.hf_analysis.cutoff_frequency_hz or 'None'}) | HF Ratio: {self.hf_analysis.hf_energy_ratio*100:.1f}%",
            f"• Indicators: {', '.join(self.synthetic_indicators) if self.synthetic_indicators else 'None (Normal Spectral Profile)'}"
        ]
        return "\n".join(lines)
