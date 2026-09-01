"""
Tests for Biometric Speaker Verification and Cross-Session Consistency.
"""

import numpy as np
import pytest

from src.verification.embeddings import SpeakerEmbeddingExtractor
from src.verification.database import SpeakerDatabase
from src.verification.verifier import SpeakerVerifier, SpeakerVerificationResult


def test_speaker_embedding_extractor(speech_wav):
    """Verify 128-D L2-normalized voiceprint extraction."""
    import soundfile as sf
    audio, sr = sf.read(str(speech_wav), dtype="float32")

    extractor = SpeakerEmbeddingExtractor()
    emb = extractor.extract_embedding(audio, sr)

    assert len(emb) == 128
    assert pytest.approx(np.linalg.norm(emb), abs=1e-4) == 1.0


def test_speaker_enrollment_and_verification(speech_wav):
    """Verify speaker enrollment and 1-to-1 biometric matching."""
    import soundfile as sf
    audio, sr = sf.read(str(speech_wav), dtype="float32")

    extractor = SpeakerEmbeddingExtractor()
    emb = extractor.extract_embedding(audio, sr)

    db = SpeakerDatabase()
    db.enroll(speaker_id="CXO_001", name="Chief Executive Officer", embedding=emb)

    verifier = SpeakerVerifier(database=db, extractor=extractor)

    # 1. Test Match (same audio against claimed CXO_001)
    res_match = verifier.verify_audio(audio, sr, claimed_speaker_id="CXO_001")
    assert res_match.is_match is True
    assert res_match.similarity_score > 0.85
    assert res_match.impersonation_detected is False

    # 2. Test Mismatch / Impersonation (different random audio claiming CXO_001)
    diff_audio = np.random.normal(0, 0.2, len(audio)).astype(np.float32)
    res_mismatch = verifier.verify_audio(diff_audio, sr, claimed_speaker_id="CXO_001")
    assert res_mismatch.is_match is False
    assert res_mismatch.impersonation_detected is True
