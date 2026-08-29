from typing import Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .embeddings import SpeakerEmbeddingExtractor
from .database import SpeakerDatabase, EnrolledSpeaker


class SpeakerVerificationResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    claimed_speaker_id: Optional[str] = None
    is_match: bool
    similarity_score: float
    threshold: float
    impersonation_detected: bool
    confidence: float
    enrolled_speaker_name: Optional[str] = None


class SpeakerVerifier:
    # 1-to-1 Cosine similarity check: Is this really the CEO or someone using an AI voice changer?

    def __init__(
        self,
        database: Optional[SpeakerDatabase] = None,
        extractor: Optional[SpeakerEmbeddingExtractor] = None,
        similarity_threshold: float = 0.75
    ):
        self.database = database or SpeakerDatabase()
        self.extractor = extractor or SpeakerEmbeddingExtractor()
        self.similarity_threshold = similarity_threshold

    def verify_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        claimed_speaker_id: Optional[str]
    ) -> SpeakerVerificationResult:
        # If no one claimed an identity, we can't do a 1-to-1 match
        if not claimed_speaker_id:
            return SpeakerVerificationResult(
                claimed_speaker_id=None,
                is_match=True,
                similarity_score=1.0,
                threshold=self.similarity_threshold,
                impersonation_detected=False,
                confidence=0.5,
                enrolled_speaker_name=None
            )

        speaker = self.database.get_speaker(claimed_speaker_id)
        if speaker is None:
            # Caller claimed to be someone we've never seen before. Suspicious.
            return SpeakerVerificationResult(
                claimed_speaker_id=claimed_speaker_id,
                is_match=False,
                similarity_score=0.0,
                threshold=self.similarity_threshold,
                impersonation_detected=True,
                confidence=0.6,
                enrolled_speaker_name="[NOT ENROLLED]"
            )

        # Extract voiceprint from current audio and compute cosine similarity
        claim_emb = self.extractor.extract_embedding(audio, sample_rate)
        cos_sim = float(np.dot(claim_emb, speaker.voiceprint))
        norm_score = float(np.clip((cos_sim + 1.0) / 2.0, 0.0, 1.0))

        is_match = bool(norm_score >= self.similarity_threshold)
        confidence = float(np.clip(0.6 + abs(norm_score - self.similarity_threshold) * 0.8, 0.0, 1.0))

        return SpeakerVerificationResult(
            claimed_speaker_id=claimed_speaker_id,
            is_match=is_match,
            similarity_score=float(round(norm_score, 4)),
            threshold=self.similarity_threshold,
            impersonation_detected=not is_match,
            confidence=float(round(confidence, 4)),
            enrolled_speaker_name=speaker.name
        )
