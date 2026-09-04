"""
Voice Integrity & Deepfake Detection Framework.
Core SDK package exposing end-to-end voice integrity analysis, deepfake detection,
speaker biometrics, and dynamic risk scoring.
"""

import sys
from pathlib import Path
from typing import Optional, Union

from .engine import VoiceIntegrityEngine, VerificationOutput
from .fusion.risk_engine import RiskScenario, Verdict, RiskLevel, RiskAssessment
from .detector.classifier import DeepfakeClassifier, DeepfakePrediction
from .verification.verifier import SpeakerVerifier, SpeakerVerificationResult
from .verification.database import SpeakerDatabase
from .ingestion.pipeline import AudioIngestionPipeline
from .ingestion.models import IngestedAudio, AudioChunk

__version__ = "1.0.0"

# Maintain backwards compatibility if imported under custom project name
if __name__ != "src" and "src" not in sys.modules:
    sys.modules["src"] = sys.modules[__name__]


def verify(
    audio_path: Union[str, Path],
    claimed_speaker_id: Optional[str] = None,
    scenario: Optional[RiskScenario] = None
) -> VerificationOutput:
    """
    Convenience function: Verify an audio file using VoiceIntegrityEngine in one line.

    Example:
        >>> import voice_clone_detection as vcd
        >>> result = vcd.verify("sample.wav")
        >>> print(result.summary())
    """
    engine = VoiceIntegrityEngine()
    return engine.verify_file(
        file_path=audio_path,
        claimed_speaker_id=claimed_speaker_id,
        scenario=scenario
    )


# Alias detect to verify
detect = verify

__all__ = [
    "VoiceIntegrityEngine",
    "VerificationOutput",
    "RiskScenario",
    "Verdict",
    "RiskLevel",
    "RiskAssessment",
    "DeepfakeClassifier",
    "DeepfakePrediction",
    "SpeakerVerifier",
    "SpeakerVerificationResult",
    "SpeakerDatabase",
    "AudioIngestionPipeline",
    "IngestedAudio",
    "AudioChunk",
    "verify",
    "detect",
    "__version__",
]
