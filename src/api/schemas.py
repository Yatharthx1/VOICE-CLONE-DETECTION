"""
Pydantic API Request/Response Schemas
Author: Yatharth
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class VerifyResponse(BaseModel):
    session_id: str
    verdict: str
    verdict_category: str = "Ambiguous"
    verdict_explanation: str = ""
    ai_probability: float = 0.0
    raw_score: float = 0.0
    dynamic_risk_score: float
    risk_level: str
    confidence: float
    window_consistency: float = 1.0
    window_predictions: List[float] = []
    scenario: str
    recommended_action: str
    risk_factors: List[str]
    speaker_match: bool
    speaker_similarity: float
    ml_synthetic_probability: float
    forensics: Dict[str, Any]
    indicators: Optional[Dict[str, Any]] = None


class AudioSampleItem(BaseModel):
    id: str
    name: str
    filename: str
    tag: str
    is_ai: bool
    description: str
    duration_sec: Optional[float] = None
    sample_rate: Optional[int] = None


class StreamChunkRequest(BaseModel):
    session_id: Optional[str] = None
    samples: List[float]
    sample_rate: int = 16000
    claimed_speaker_id: Optional[str] = None
    scenario: str = "general_telephony"


class StreamChunkResponse(BaseModel):
    verdict: str
    verdict_category: str = "Ambiguous"
    verdict_explanation: str = ""
    ai_probability: float = 0.0
    raw_score: float = 0.0
    dynamic_risk_score: float
    risk_level: str
    confidence: float
    window_consistency: float = 1.0
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
