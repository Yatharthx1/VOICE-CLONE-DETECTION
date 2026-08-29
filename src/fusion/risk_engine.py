from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.analysis.manager import ParallelAnalysisOutput
from .fusion_engine import FusedScore
from .calibrator import CalibratedResult


class RiskScenario(str, Enum):
    HIGH_VALUE_TRANSACTION = "high_value_transaction"
    CONFIDENTIAL_DISCLOSURE = "confidential_disclosure"
    STANDARD_SUPPORT = "standard_support"
    GENERAL_TELEPHONY = "general_telephony"


class Verdict(str, Enum):
    REAL = "REAL"
    SYNTHETIC = "SYNTHETIC"
    MANIPULATED = "MANIPULATED"
    UNCERTAIN = "UNCERTAIN"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskAssessment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    verdict: Verdict
    dynamic_risk_score: float
    risk_level: RiskLevel
    confidence: float
    scenario: RiskScenario
    recommended_action: str
    risk_factors: List[str] = Field(default_factory=list)
    fused_score_details: FusedScore
    calibrated_details: CalibratedResult

    def summary(self) -> str:
        lines = [
            f"=== Risk Assessment & Final Verdict ===",
            f"• Verdict: [{self.verdict.value}] | Risk Level: [{self.risk_level.value}]",
            f"• Dynamic Risk Score: {self.dynamic_risk_score:.1f} / 100.0 (Confidence: {self.confidence * 100:.1f}%)",
            f"• Scenario: {self.scenario.value.upper()}",
            f"• Recommended Action: {self.recommended_action}",
            f"• Key Risk Factors: {', '.join(self.risk_factors) if self.risk_factors else 'None (Clean Phonation Baseline)'}"
        ]
        return "\n".join(lines)


class RiskScoringEngine:
    # Adapts risk thresholds depending on whether someone is ordering pizza or wiring $500k

    SCENARIO_THRESHOLDS = {
        RiskScenario.HIGH_VALUE_TRANSACTION: {"synthetic_thresh": 0.35, "uncertain_window": 0.10},
        RiskScenario.CONFIDENTIAL_DISCLOSURE: {"synthetic_thresh": 0.45, "uncertain_window": 0.08},
        RiskScenario.STANDARD_SUPPORT: {"synthetic_thresh": 0.65, "uncertain_window": 0.12},
        RiskScenario.GENERAL_TELEPHONY: {"synthetic_thresh": 0.50, "uncertain_window": 0.10},
    }

    def __init__(self, scenario: RiskScenario = RiskScenario.GENERAL_TELEPHONY):
        self.scenario = scenario

    def assess_risk(
        self,
        fused: FusedScore,
        calibrated: CalibratedResult,
        analysis: ParallelAnalysisOutput,
        scenario: Optional[RiskScenario] = None
    ) -> RiskAssessment:
        active_scenario = scenario or self.scenario
        threshold_config = self.SCENARIO_THRESHOLDS.get(
            active_scenario,
            self.SCENARIO_THRESHOLDS[RiskScenario.GENERAL_TELEPHONY]
        )
        thresh = threshold_config["synthetic_thresh"]
        uncert_win = threshold_config["uncertain_window"]

        prob = calibrated.calibrated_probability
        conf = calibrated.confidence
        risk_score_100 = float(round(prob * 100.0, 1))

        # Check if someone frankensteined genuine audio with splices
        is_spliced = analysis.synthesis_artifacts.concatenation_artifacts.splice_points_detected >= 2

        # 1. Determine Verdict
        if conf < 0.40 or (abs(prob - thresh) < uncert_win and conf < 0.65):
            verdict = Verdict.UNCERTAIN
        elif is_spliced:
            verdict = Verdict.MANIPULATED
        elif prob >= thresh:
            verdict = Verdict.SYNTHETIC
        else:
            verdict = Verdict.REAL

        # 2. Determine Risk Tier
        if risk_score_100 >= 75.0 or (verdict == Verdict.SYNTHETIC and active_scenario == RiskScenario.HIGH_VALUE_TRANSACTION):
            risk_level = RiskLevel.CRITICAL
        elif risk_score_100 >= 50.0 or verdict in [Verdict.SYNTHETIC, Verdict.MANIPULATED]:
            risk_level = RiskLevel.HIGH
        elif risk_score_100 >= 30.0 or verdict == Verdict.UNCERTAIN:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # 3. Prescribe action for the frontline staff
        if risk_level == RiskLevel.CRITICAL:
            action = "BLOCK_TRANSACTION_AND_ESCALATE_TO_FRAUD_CELL"
        elif risk_level == RiskLevel.HIGH:
            action = "TRIGGER_STEP_UP_MFA_AND_VOICE_CALLBACK"
        elif risk_level == RiskLevel.MEDIUM:
            action = "ADVISE_SECONDARY_VERIFICATION"
        else:
            action = "AUTHORIZE_STANDARD_PROCESSING"

        # 4. Gather the forensic rap sheet
        risk_factors = analysis.all_synthetic_indicators()
        if fused.impersonation_penalty > 0:
            risk_factors.append(f"Speaker biometric mismatch (Impersonation penalty +{fused.impersonation_penalty * 100:.0f}%)")

        return RiskAssessment(
            verdict=verdict,
            dynamic_risk_score=risk_score_100,
            risk_level=risk_level,
            confidence=conf,
            scenario=active_scenario,
            recommended_action=action,
            risk_factors=risk_factors,
            fused_score_details=fused,
            calibrated_details=calibrated
        )
