"""
Pydantic API Request/Response Schemas
Author: Yatharth
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class VerifyResponse(BaseModel):
    session_id: str
    verdict: str
    dynamic_risk_score: float
    risk_level: str
    confidence: float
    scenario: str
    recommended_action: str
    risk_factors: List[str]
    speaker_match: bool
    speaker_similarity: float
    ml_synthetic_probability: float
    forensics: Dict[str, Any]


class StreamChunkRequest(BaseModel):
    session_id: Optional[str] = None
    samples: List[float]
    sample_rate: int = 16000
    claimed_speaker_id: Optional[str] = None
    scenario: str = "general_telephony"


class StreamChunkResponse(BaseModel):
    verdict: str
    dynamic_risk_score: float
    risk_level: str
    confidence: float
    recommended_action: str
    risk_factors: List[str]


class SpeakerEnrollRequest(BaseModel):
    speaker_id: str
    name: str
    metadata: Optional[Dict[str, Any]] = None


class SpeakerEnrollResponse(BaseModel):
    speaker_id: str
    name: str
    sample_count: int
    enrolled_at: str
    status: str = "SUCCESS"


class HealthResponse(BaseModel):
    status: str
    version: str
    framework: str
    gpu_available: bool
    enrolled_speakers_count: int
