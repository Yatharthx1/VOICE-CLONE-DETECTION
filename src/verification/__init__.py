"""
Speaker Verification Subsystem.
"""

from .embeddings import SpeakerEmbeddingExtractor
from .database import SpeakerDatabase, EnrolledSpeaker
from .verifier import SpeakerVerifier, SpeakerVerificationResult

__all__ = [
    "SpeakerEmbeddingExtractor",
    "SpeakerDatabase",
    "EnrolledSpeaker",
    "SpeakerVerifier",
    "SpeakerVerificationResult",
]
