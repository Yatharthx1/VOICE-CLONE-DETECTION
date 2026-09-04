"""
FastAPI REST API Integration Tests.
"""

from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.api.server import app

client = TestClient(app)


def test_api_health_endpoint():
    """Verify /api/v1/health returns 200 OK."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "framework" in data


def test_api_verify_file_upload(speech_wav):
    """Verify /api/v1/verify processes uploaded audio file."""
    with open(speech_wav, "rb") as f:
        response = client.post(
            "/api/v1/verify",
            files={"file": ("test_speech.wav", f, "audio/wav")},
            data={"scenario": "high_value_transaction"}
        )

    assert response.status_code == 200
    data = response.json()
    assert "verdict" in data
    assert "dynamic_risk_score" in data
    assert "recommended_action" in data
    assert "indicators" in data
    assert "spectralCutoff" in data["indicators"]
    assert "pitchJitter" in data["indicators"]
    assert "phaseCoherence" in data["indicators"]
    assert "formantTransitions" in data["indicators"]
    assert data["scenario"] == "high_value_transaction"


def test_api_benchmark_samples():
    """Verify /api/v1/samples and /api/v1/samples/{filename} endpoints."""
    res = client.get("/api/v1/samples")
    assert res.status_code == 200
    samples = res.json()
    assert len(samples) >= 2
    assert any(s["filename"] == "sample_genuine_human.wav" for s in samples)

    # Test downloading sample audio
    sample_res = client.get("/api/v1/samples/sample_genuine_human.wav")
    assert sample_res.status_code == 200
    assert len(sample_res.content) > 0


def test_api_stream_chunk():
    """Verify /api/v1/stream/chunk evaluates live packet."""
    samples = list(np.random.normal(0, 0.1, 1600).astype(float))
    response = client.post(
        "/api/v1/stream/chunk",
        json={
            "samples": samples,
            "sample_rate": 16000,
            "scenario": "general_telephony"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "verdict" in data
    assert "dynamic_risk_score" in data


def test_api_speaker_enrollment(speech_wav):
    """Verify /api/v1/speakers/enroll and /api/v1/speakers listing."""
    with open(speech_wav, "rb") as f:
        enroll_res = client.post(
            "/api/v1/speakers/enroll",
            files={"file": ("cxo.wav", f, "audio/wav")},
            data={"speaker_id": "API_CXO_1", "name": "API Enrolled CXO"}
        )

    assert enroll_res.status_code == 200
    assert enroll_res.json()["status"] == "SUCCESS"

    list_res = client.get("/api/v1/speakers")
    assert list_res.status_code == 200
    speakers = list_res.json()
    assert any(s["speaker_id"] == "API_CXO_1" for s in speakers)
