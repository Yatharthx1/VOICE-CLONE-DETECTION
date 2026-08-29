"""
Data models for Synthesis Artifacts Analysis outputs.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.analysis.base import BaseAnalysisResult


class NeuralVocoderArtifacts(BaseModel):
    """Artifacts created by transposed convolutions and neural vocoders (HiFi-GAN/MelGAN/WaveGrad)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    checkerboard_energy_ratio: float = Field(0.0, description="Energy ratio of periodic spectral checkerboard patterns.")
    periodic_artifact_detected: bool = Field(False, description="True if transposed convolution periodicity is detected.")
    harmonic_smearing_score: float = Field(0.0, description="Degree of harmonic spectral smearing/broadening (0.0 to 1.0).")


class ConcatenationArtifacts(BaseModel):
    """Discontinuities caused by audio splicing or unit-selection synthesis."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    splice_points_detected: int = Field(0, description="Number of suspicious splice/glitch boundaries.")
    max_energy_jump_db: float = Field(0.0, description="Maximum sudden frame-to-frame energy jump in dB.")
    max_phase_jump_rad: float = Field(0.0, description="Maximum sudden phase jump in radians.")


class SynthesisArtifactsResult(BaseAnalysisResult):
    """
    Consolidated synthesis artifacts analysis output.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    module_name: str = Field(default="synthesis_artifacts")
    anomaly_score: float = Field(0.0, description="Synthesis artifact anomaly score from 0.0 to 1.0.")
    confidence: float = Field(0.0, description="Confidence in artifact detection.")
    vocoder_artifacts: NeuralVocoderArtifacts = Field(..., description="Neural vocoder signature metrics.")
    concatenation_artifacts: ConcatenationArtifacts = Field(..., description="Splice and boundary discontinuity metrics.")
    artifact_features: Dict[str, float] = Field(default_factory=dict, description="Flattened ML feature dictionary.")
    synthetic_indicators: List[str] = Field(default_factory=list, description="Specific synthesis artifacts identified.")

    def summary(self) -> str:
        lines = [
            f"=== Synthesis Artifacts Summary ===",
            f"• Anomaly Score: {self.anomaly_score:.3f} | Confidence: {self.confidence:.2f}",
            f"• Vocoder Check: Periodic Pattern = {self.vocoder_artifacts.periodic_artifact_detected} | Smearing: {self.vocoder_artifacts.harmonic_smearing_score:.3f}",
            f"• Splicing Check: Splices Detected = {self.concatenation_artifacts.splice_points_detected} | Max Step: {self.concatenation_artifacts.max_energy_jump_db:.1f} dB",
            f"• Indicators: {', '.join(self.synthetic_indicators) if self.synthetic_indicators else 'None (Clean Phonation Waveform)'}"
        ]
        return "\n".join(lines)
