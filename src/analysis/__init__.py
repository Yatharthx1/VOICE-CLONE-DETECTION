"""
Parallel Analysis Subsystem for Real-Time Voice Cloning Detection.
Houses parallel feature extraction engines: Acoustic, Spectral, Prosody, Synthesis Artifacts, and Phase Analysis.
"""

from .base import BaseAnalysisModule, BaseAnalysisResult
from .acoustic import AcousticAnalyzer, AcousticAnalysisConfig, AcousticAnalysisResult
from .spectral import SpectralAnalyzer, SpectralAnalysisConfig, SpectralAnalysisResult
from .prosody import ProsodyAnalyzer, ProsodyAnalysisConfig, ProsodyAnalysisResult
from .synthesis_artifacts import SynthesisArtifactsAnalyzer, SynthesisArtifactsConfig, SynthesisArtifactsResult
from .phase import PhaseAnalyzer, PhaseAnalysisConfig, PhaseAnalysisResult
from .manager import ParallelAnalysisManager, ParallelAnalysisOutput

__all__ = [
    "BaseAnalysisModule",
    "BaseAnalysisResult",
    "AcousticAnalyzer",
    "AcousticAnalysisConfig",
    "AcousticAnalysisResult",
    "SpectralAnalyzer",
    "SpectralAnalysisConfig",
    "SpectralAnalysisResult",
    "ProsodyAnalyzer",
    "ProsodyAnalysisConfig",
    "ProsodyAnalysisResult",
    "SynthesisArtifactsAnalyzer",
    "SynthesisArtifactsConfig",
    "SynthesisArtifactsResult",
    "PhaseAnalyzer",
    "PhaseAnalysisConfig",
    "PhaseAnalysisResult",
    "ParallelAnalysisManager",
    "ParallelAnalysisOutput",
]
