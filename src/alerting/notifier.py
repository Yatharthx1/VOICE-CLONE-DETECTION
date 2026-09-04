from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from ..fusion.risk_engine import RiskAssessment, RiskLevel


class AlertPayload(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    alert_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    risk_level: RiskLevel
    verdict: str
    dynamic_risk_score: float
    recommended_action: str
    risk_factors: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None
    target_channels: List[str] = Field(default_factory=list)


class AlertDispatcher:
    # Screams into SIEM/Webhooks/Consoles whenever someone tries an AI voice attack

    def __init__(self):
        self._sent_alerts: List[AlertPayload] = []

    def dispatch(
        self,
        assessment: RiskAssessment,
        session_id: Optional[str] = None,
        channels: Optional[List[str]] = None
    ) -> AlertPayload:
        import uuid
        selected_channels = channels or ["console", "in_app", "webhook"]
        
        payload = AlertPayload(
            alert_id=str(uuid.uuid4()),
            risk_level=assessment.risk_level,
            verdict=assessment.verdict.value,
            dynamic_risk_score=assessment.dynamic_risk_score,
            recommended_action=assessment.recommended_action,
            risk_factors=assessment.risk_factors,
            session_id=session_id,
            target_channels=selected_channels
        )

        if assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            try:
                print(f"\n🚨 [CRITICAL SECURITY ALERT] Deepfake Impersonation Detected!")
                print(f"   • Verdict: {payload.verdict} | Risk Score: {payload.dynamic_risk_score:.1f}/100")
                print(f"   • Action: {payload.recommended_action}")
                if payload.risk_factors:
                    print(f"   • Factors: {', '.join(payload.risk_factors[:3])}")
            except Exception:
                print(f"\n[CRITICAL SECURITY ALERT] Deepfake Impersonation Detected: {payload.verdict} ({payload.dynamic_risk_score:.1f}/100)")

        self._sent_alerts.append(payload)
        return payload

    def get_alert_history(self) -> List[AlertPayload]:
        return list(self._sent_alerts)
