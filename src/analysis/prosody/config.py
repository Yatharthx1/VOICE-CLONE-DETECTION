"""
Configuration for Prosody Analysis Subsystem.
"""

from typing import Tuple
from pydantic import BaseModel, ConfigDict, Field


class ProsodyAnalysisConfig(BaseModel):
    """
    Configuration parameters for Speaking Rate, Syllable Nuclei Detection,
    Pairwise Variability Index (nPVI/rPVI), Pitch Intonation Dynamics, and Monotone Detection.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    min_syllable_duration_ms: float = Field(default=80.0, description="Minimum duration of a syllable nucleus in ms.")
    syllable_energy_threshold_db: float = Field(default=-30.0, description="Relative energy threshold for syllable peak detection.")
    min_pause_duration_ms: float = Field(default=200.0, description="Minimum duration to qualify as a natural pause.")
    
    # Normal human conversational speech bounds
    human_speaking_rate_range: Tuple[float, float] = Field(
        default=(2.0, 7.0),
        description="Normal speaking rate range in syllables per second."
    )
    human_npvi_range: Tuple[float, float] = Field(
        default=(30.0, 75.0),
        description="Normalized Pairwise Variability Index (nPVI) range for natural rhythm."
    )
    monotone_f0_std_threshold_hz: float = Field(
        default=8.0,
        description="F0 standard deviation threshold below which speech is considered unnaturally monotonic."
    )
