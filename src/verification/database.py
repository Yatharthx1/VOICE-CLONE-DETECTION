from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class EnrolledSpeaker(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    speaker_id: str
    name: str
    voiceprint: Any
    sample_count: int = 1
    enrolled_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SpeakerDatabase:
    # Keeps track of genuine VIP voiceprints so we can catch cloned impersonators

    def __init__(self, persistence_path: Optional[Path] = None):
        self.persistence_path = persistence_path
        self._speakers: Dict[str, EnrolledSpeaker] = {}

    def enroll(
        self,
        speaker_id: str,
        name: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EnrolledSpeaker:
        # Normalize the incoming vector just in case someone passed raw numbers
        emb = embedding / (np.linalg.norm(embedding) + 1e-8)

        if speaker_id in self._speakers:
            # Running average update: smooth out the voiceprint over multiple samples
            existing = self._speakers[speaker_id]
            updated_vec = (existing.voiceprint * existing.sample_count + emb) / (existing.sample_count + 1)
            updated_vec = updated_vec / np.linalg.norm(updated_vec)
            existing.voiceprint = updated_vec
            existing.sample_count += 1
            if metadata:
                existing.metadata.update(metadata)
            return existing
        else:
            profile = EnrolledSpeaker(
                speaker_id=speaker_id,
                name=name,
                voiceprint=emb,
                sample_count=1,
                metadata=metadata or {}
            )
            self._speakers[speaker_id] = profile
            return profile

    def get_speaker(self, speaker_id: str) -> Optional[EnrolledSpeaker]:
        return self._speakers.get(speaker_id)

    def list_speakers(self) -> List[EnrolledSpeaker]:
        return list(self._speakers.values())

    def delete_speaker(self, speaker_id: str) -> bool:
        if speaker_id in self._speakers:
            del self._speakers[speaker_id]
            return True
        return False
