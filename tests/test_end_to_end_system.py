"""
End-to-End System Tests for Master VoiceIntegrityEngine.
"""

from pathlib import Path
import pytest

from src.engine import VoiceIntegrityEngine, VerificationOutput
from src.fusion.risk_engine import RiskScenario, Verdict


def test_voice_integrity_engine_end_to_end_file(speech_wav):
    """Verify complete end-to-end processing across all subsystems."""
    engine = VoiceIntegrityEngine(default_scenario=RiskScenario.GENERAL_TELEPHONY)
    result = engine.verify_file(speech_wav)

    assert isinstance(result, VerificationOutput)
    assert result.session_id is not None
    assert result.assessment.verdict in [Verdict.REAL, Verdict.SYNTHETIC, Verdict.MANIPULATED, Verdict.UNCERTAIN]
    assert 0.0 <= result.assessment.dynamic_risk_score <= 100.0
    assert result.ingested_audio.forensic.verified_integrity is True

    # Check summary string
    summary = result.summary()
    assert "VOICE INTEGRITY VERIFICATION FINAL REPORT" in summary
    assert "DYNAMIC RISK SCORE" in summary

    # Check dict conversion
    d = result.to_dict()
    assert d["session_id"] == result.session_id
    assert "dynamic_risk_score" in d


def test_voice_integrity_engine_with_claimed_speaker(speech_wav):
    """Verify end-to-end execution with speaker enrollment and claim verification."""
    engine = VoiceIntegrityEngine()
    
    # 1. Enroll speaker
    ingested = engine.ingestion_pipeline.process_file(speech_wav)
    emb = engine.speaker_verifier.extractor.extract_embedding(
        ingested.processed_audio,
        ingested.target_sample_rate
    )
    engine.speaker_database.enroll(speaker_id="VIP_USER", name="Vip Customer", embedding=emb)

    # 2. Verify with matching claim
    res_match = engine.verify_file(speech_wav, claimed_speaker_id="VIP_USER")
    assert res_match.speaker_verification.is_match is True
    assert res_match.speaker_verification.impersonation_detected is False

    # 3. Verify with wrong claim
    res_wrong = engine.verify_file(speech_wav, claimed_speaker_id="UNKNOWN_IMPOSTOR")
    assert res_wrong.speaker_verification.is_match is False
