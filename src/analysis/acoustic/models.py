"""
Data models for Acoustic Analysis outputs (Pitch, Formants, Voice Quality, and Anomaly Scores).
"""

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..base import BaseAnalysisResult


class PitchAnalysisResult(BaseModel):
    """Fundamental frequency (F0) tracking and perturbation (Jitter/Shimmer) metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mean_f0_hz: float = Field(0.0, description="Mean fundamental frequency in Hz.")
    std_f0_hz: float = Field(0.0, description="Standard deviation of F0.")
    min_f0_hz: float = Field(0.0, description="Minimum voiced F0 in Hz.")
    max_f0_hz: float = Field(0.0, description="Maximum voiced F0 in Hz.")
    voiced_fraction: float = Field(0.0, description="Fraction of frames classified as voiced (0.0 - 1.0).")
    f0_contour: List[float] = Field(default_factory=list, description="Per-frame F0 contour in Hz (0.0 for unvoiced).")
    
    # Jitter Metrics (Frequency Perturbations)
    jitter_local_pct: float = Field(0.0, description="Local cycle-to-cycle pitch jitter percentage.")
    jitter_rap_pct: float = Field(0.0, description="Relative Average Perturbation (RAP) jitter %.")
    jitter_ppq5_pct: float = Field(0.0, description="Five-point Period Perturbation Quotient (PPQ5) %.")

    # Shimmer Metrics (Amplitude Perturbations)
    shimmer_local_pct: float = Field(0.0, description="Local cycle-to-cycle amplitude shimmer percentage.")
    shimmer_apq3_pct: float = Field(0.0, description="Three-point Amplitude Perturbation Quotient (APQ3) %.")
    shimmer_apq5_pct: float = Field(0.0, description="Five-point Amplitude Perturbation Quotient (APQ5) %.")
    shimmer_db: float = Field(0.0, description="Local amplitude shimmer in dB.")


class FormantResult(BaseModel):
    """Linear Predictive Coding (LPC) Formant resonances and Vocal Tract metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    f1_mean_hz: Optional[float] = Field(None, description="First formant (F1) mean frequency in Hz (pharyngeal resonance).")
    f2_mean_hz: Optional[float] = Field(None, description="Second formant (F2) mean frequency in Hz (oral cavity resonance).")
    f3_mean_hz: Optional[float] = Field(None, description="Third formant (F3) mean frequency in Hz (nasal/lip resonance).")
    f4_mean_hz: Optional[float] = Field(None, description="Fourth formant (F4) mean frequency in Hz.")
    
    f1_std_hz: Optional[float] = Field(None, description="F1 standard deviation across voiced frames.")
    f2_std_hz: Optional[float] = Field(None, description="F2 standard deviation.")
    f3_std_hz: Optional[float] = Field(None, description="F3 standard deviation.")

    formant_dispersion_hz: Optional[float] = Field(None, description="Average frequency distance between successive formants.")
    vocal_tract_length_cm: Optional[float] = Field(None, description="Estimated physiological Vocal Tract Length (VTL) in cm.")


class VoiceQualityResult(BaseModel):
    """Acoustic voice quality, periodicity, and spectral purity metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hnr_mean_db: float = Field(0.0, description="Harmonics-to-Noise Ratio (HNR) mean in dB.")
    hnr_std_db: float = Field(0.0, description="HNR standard deviation in dB.")
    cpp_mean_db: float = Field(0.0, description="Cepstral Peak Prominence (CPP) mean in dB.")
    cpp_std_db: float = Field(0.0, description="CPP standard deviation in dB.")
    zero_crossing_rate_mean: float = Field(0.0, description="Average Zero Crossing Rate across frames.")
    energy_entropy_mean: float = Field(0.0, description="Average spectral energy entropy.")


class AcousticAnalysisResult(BaseAnalysisResult):
    """
    Consolidated acoustic analysis output containing pitch, formants,
    voice quality metrics, ML-ready feature dictionary, and anomaly risk assessment.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    module_name: str = Field(default="acoustic_analysis")
    anomaly_score: float = Field(
        0.0,
        description="Acoustic anomaly score from 0.0 (natural genuine human) to 1.0 (highly synthetic/cloned)."
    )
    confidence: float = Field(
        0.0,
        description="Confidence score in the acoustic anomaly estimation (0.0 to 1.0)."
    )
    pitch: PitchAnalysisResult = Field(..., description="Pitch and perturbation metrics.")
    formants: FormantResult = Field(..., description="LPC formant tracking metrics.")
    voice_quality: VoiceQualityResult = Field(..., description="Acoustic voice quality and periodicity metrics.")
    acoustic_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Flattened feature dictionary for downstream ML classifier."
    )
    synthetic_indicators: List[str] = Field(
        default_factory=list,
        description="List of detected acoustic anomalies indicating synthetic or cloned speech."
    )

    def summary(self) -> str:
        lines = [
            f"=== Acoustic Analysis Summary ===",
            f"• Anomaly Score: {self.anomaly_score:.3f} | Confidence: {self.confidence:.2f}",
            f"• Pitch (F0): Mean {self.pitch.mean_f0_hz:.1f} Hz | Std: {self.pitch.std_f0_hz:.1f} Hz | Voiced: {self.pitch.voiced_fraction*100:.1f}%",
            f"• Perturbation: Jitter {self.pitch.jitter_local_pct:.2f}% | Shimmer {self.pitch.shimmer_local_pct:.2f}% ({self.pitch.shimmer_db:.2f} dB)",
            f"• Formants: F1={self.formants.f1_mean_hz or 0:.0f}Hz, F2={self.formants.f2_mean_hz or 0:.0f}Hz, F3={self.formants.f3_mean_hz or 0:.0f}Hz | VTL: {self.formants.vocal_tract_length_cm or 0:.1f}cm",
            f"• Voice Quality: HNR {self.voice_quality.hnr_mean_db:.1f} dB | CPP {self.voice_quality.cpp_mean_db:.1f} dB | ZCR: {self.voice_quality.zero_crossing_rate_mean:.3f}",
            f"• Indicators: {', '.join(self.synthetic_indicators) if self.synthetic_indicators else 'None (Natural Acoustic Baseline)'}"
        ]
        return "\n".join(lines)
