from typing import Optional
from ..fusion.risk_engine import RiskAssessment, RiskLevel
from .notifier import AlertDispatcher, AlertPayload


class InterventionWorkflow:
    # Auto-triggers security alarms if risk climbs past medium

    def __init__(self, dispatcher: Optional[AlertDispatcher] = None):
        self.dispatcher = dispatcher or AlertDispatcher()

    def process_assessment(
        self,
        assessment: RiskAssessment,
        session_id: Optional[str] = None
    ) -> Optional[AlertPayload]:
        # Only bother humans if the risk is actually concerning
        if assessment.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]:
            return self.dispatcher.dispatch(assessment, session_id=session_id)
        return None
