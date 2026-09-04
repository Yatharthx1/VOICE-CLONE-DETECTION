"""
Parallel Analysis Manager.
Executes Acoustic, Spectral, Prosody, Synthesis Artifacts, and Phase Analysis concurrently.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..ingestion.models import IngestedAudio, AudioChunk
from .acoustic import AcousticAnalyzer, AcousticAnalysisResult
from .spectral import SpectralAnalyzer, SpectralAnalysisResult
from .prosody import ProsodyAnalyzer, ProsodyAnalysisResult
from .synthesis_artifacts import SynthesisArtifactsAnalyzer, SynthesisArtifactsResult
from .phase import PhaseAnalyzer, PhaseAnalysisResult


class ParallelAnalysisOutput(BaseModel):
    """Container holding outputs from all 5 parallel feature extraction modules."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    acoustic: AcousticAnalysisResult
    spectral: SpectralAnalysisResult
    prosody: ProsodyAnalysisResult
    synthesis_artifacts: SynthesisArtifactsResult
    phase: PhaseAnalysisResult

    def aggregated_features(self) -> Dict[str, float]:
        """Aggregate all extracted features into a single flat dictionary."""
        feats = {}
        feats.update(self.acoustic.acoustic_features)
        feats.update(self.spectral.spectral_features)
        feats.update(self.prosody.prosody_features)
        feats.update(self.synthesis_artifacts.artifact_features)
        feats.update(self.phase.phase_features)
        return feats

    def all_synthetic_indicators(self) -> List[str]:
        """Combine all forensic anomaly indicators across all 5 analyzers."""
        indicators = []
        indicators.extend(self.acoustic.synthetic_indicators)
        indicators.extend(self.spectral.synthetic_indicators)
        indicators.extend(self.prosody.synthetic_indicators)
        indicators.extend(self.synthesis_artifacts.synthetic_indicators)
        indicators.extend(self.phase.synthetic_indicators)
        # Filter generic placeholders
        return [ind for ind in indicators if not ind.startswith("None") and not ind.startswith("No audio")]


class ParallelAnalysisManager:
    """
    Coordinates and parallelizes the 5 analysis engines:
    Acoustic, Spectral, Prosody, Synthesis Artifacts, and Phase Analysis.
    """

    def __init__(
        self,
        acoustic_analyzer: Optional[AcousticAnalyzer] = None,
        spectral_analyzer: Optional[SpectralAnalyzer] = None,
        prosody_analyzer: Optional[ProsodyAnalyzer] = None,
        synthesis_analyzer: Optional[SynthesisArtifactsAnalyzer] = None,
        phase_analyzer: Optional[PhaseAnalyzer] = None,
        max_workers: int = 5
    ):
        self.acoustic = acoustic_analyzer or AcousticAnalyzer()
        self.spectral = spectral_analyzer or SpectralAnalyzer()
        self.prosody = prosody_analyzer or ProsodyAnalyzer()
        self.synthesis_artifacts = synthesis_analyzer or SynthesisArtifactsAnalyzer()
        self.phase = phase_analyzer or PhaseAnalyzer()
        self.max_workers = max_workers

    def analyze_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> ParallelAnalysisOutput:
        """
        Execute all 5 parallel analyses concurrently on an audio array.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            fut_acoustic = executor.submit(self.acoustic.analyze, audio, sample_rate)
            fut_spectral = executor.submit(self.spectral.analyze, audio, sample_rate)
            fut_prosody = executor.submit(self.prosody.analyze, audio, sample_rate)
            fut_synthesis = executor.submit(self.synthesis_artifacts.analyze, audio, sample_rate)
            fut_phase = executor.submit(self.phase.analyze, audio, sample_rate)

            res_acoustic = fut_acoustic.result()
            res_spectral = fut_spectral.result()
            res_prosody = fut_prosody.result()
            res_synthesis = fut_synthesis.result()
            res_phase = fut_phase.result()

        return ParallelAnalysisOutput(
            acoustic=res_acoustic,
            spectral=res_spectral,
            prosody=res_prosody,
            synthesis_artifacts=res_synthesis,
            phase=res_phase
        )

    def analyze_ingested(self, ingested: IngestedAudio) -> ParallelAnalysisOutput:
        """
        Execute all 5 parallel analyses on an IngestedAudio object.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            fut_acoustic = executor.submit(self.acoustic.analyze_ingested, ingested)
            fut_spectral = executor.submit(self.spectral.analyze_ingested, ingested)
            fut_prosody = executor.submit(self.prosody.analyze_ingested, ingested)
            fut_synthesis = executor.submit(self.synthesis_artifacts.analyze_ingested, ingested)
            fut_phase = executor.submit(self.phase.analyze_ingested, ingested)

            res_acoustic = fut_acoustic.result()
            res_spectral = fut_spectral.result()
            res_prosody = fut_prosody.result()
            res_synthesis = fut_synthesis.result()
            res_phase = fut_phase.result()

        return ParallelAnalysisOutput(
            acoustic=res_acoustic,
            spectral=res_spectral,
            prosody=res_prosody,
            synthesis_artifacts=res_synthesis,
            phase=res_phase
        )

    def analyze_chunk(self, chunk: AudioChunk) -> ParallelAnalysisOutput:
        """Analyze an individual sliding analysis window chunk."""
        return self.analyze_audio(chunk.samples, chunk.sample_rate)
