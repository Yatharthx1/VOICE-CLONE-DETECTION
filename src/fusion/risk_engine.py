from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from ..analysis.manager import ParallelAnalysisOutput
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
    NO_SPEECH = "NO_SPEECH"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskAssessment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    verdict: Verdict
    ai_probability: float = 0.0
    raw_score: float = 0.0
    dynamic_risk_score: float
    risk_level: RiskLevel
    confidence: float
    window_consistency: float = 1.0
    verdict_category: str = "Ambiguous"
    verdict_explanation: str = ""
    scenario: RiskScenario
    recommended_action: str
    risk_factors: List[str] = Field(default_factory=list)
    fused_score_details: FusedScore
    calibrated_details: CalibratedResult

    def summary(self) -> str:
        lines = [
            f"=== Risk Assessment & Final Verdict ===",
            f"• Verdict: [{self.verdict.value}] ({self.verdict_category}) | Risk Level: [{self.risk_level.value}]",
            f"• AI Probability: {self.ai_probability * 100.0:.1f}% | Raw Score: {self.raw_score:.3f} | Confidence: {self.confidence * 100:.1f}%",
            f"• Window Consistency: {self.window_consistency * 100:.1f}%",
            f"• Scenario: {self.scenario.value.upper()}",
            f"• Recommended Action: {self.recommended_action}",
            f"• Verdict Explanation: {self.verdict_explanation}",
            f"• Key Risk Factors: {', '.join(self.risk_factors) if self.risk_factors else 'None (Clean Phonation Baseline)'}"
        ]
        return "\n".join(lines)


class RiskScoringEngine:
    """
    Evaluates multi-modal evidence against configurable, scenario-adaptive thresholds.
    Thresholds determine the VERDICT; they never manipulate or overwrite the underlying probability.
    """

    # Target Probability Tiers
    CLEARLY_HUMAN_MAX = 0.20
    MOSTLY_HUMAN_MAX = 0.40
    AMBIGUOUS_MAX = 0.60
    LIKELY_AI_MAX = 0.80
    STRONG_AI_MAX = 0.95

    SCENARIO_THRESHOLDS = {
        RiskScenario.HIGH_VALUE_TRANSACTION: {"synthetic_thresh": 0.70, "uncertain_window": 0.10},
        RiskScenario.CONFIDENTIAL_DISCLOSURE: {"synthetic_thresh": 0.75, "uncertain_window": 0.10},
        RiskScenario.STANDARD_SUPPORT: {"synthetic_thresh": 0.80, "uncertain_window": 0.10},
        RiskScenario.GENERAL_TELEPHONY: {"synthetic_thresh": 0.75, "uncertain_window": 0.10},
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
        raw_s = calibrated.raw_score
        cons = calibrated.window_consistency
        category = calibrated.verdict_category
        risk_score_100 = float(round(prob * 100.0, 1))
        risk_factors = analysis.all_synthetic_indicators()
        modality_scores = fused.modality_scores or {}
        strong_signal_count = sum(
            1
            for name, score in modality_scores.items()
            if name != "ml_detector" and score >= 0.55
        )
        ml_score = modality_scores.get("ml_detector", raw_s)
        has_strong_ai_evidence = (
            prob >= 0.90
            or ml_score >= 0.90
            or (prob >= thresh and (strong_signal_count >= 2 or len(risk_factors) >= 2 or ml_score >= 0.85))
        )

        # Check for frankenstein audio splicing
        is_spliced = analysis.synthesis_artifacts.concatenation_artifacts.splice_points_detected >= 2

        # 1. Determine Verdict and Detailed Explanation
        if 0.45 <= prob < thresh or (abs(prob - thresh) < uncert_win and conf < 0.65):
            verdict = Verdict.UNCERTAIN
            explanation = "Acoustic evidence is inconclusive or in the ambiguous transition zone."
        elif is_spliced and 0.25 < prob < 0.80:
            verdict = Verdict.MANIPULATED
            explanation = "Audio contains multiple abrupt splicing boundaries and spectral discontinuities."
        elif prob >= thresh and has_strong_ai_evidence:
            verdict = Verdict.SYNTHETIC
            if prob >= self.STRONG_AI_MAX:
                explanation = "Extremely strong AI synthesis evidence detected across multiple acoustic and neural vocoder modalities."
            elif prob >= self.LIKELY_AI_MAX:
                explanation = "Strong evidence of synthetic voice generation and neural vocoder artifacts."
            else:
                explanation = "Likely AI-generated voice characteristics detected."
        elif prob >= thresh:
            verdict = Verdict.UNCERTAIN
            explanation = "AI probability is elevated, but supporting artifact evidence is not strong enough to label the speaker synthetic."
        else:
            verdict = Verdict.REAL
            if prob <= self.CLEARLY_HUMAN_MAX:
                explanation = "Phonation, micro-perturbations, and vocal tract resonances match natural biological human speech."
            else:
                explanation = "Acoustic characteristics are predominantly human with minor background or recording variations."

        # 2. Determine Risk Tier
        if verdict == Verdict.SYNTHETIC and risk_score_100 >= 75.0:
            risk_level = RiskLevel.CRITICAL
        elif verdict in [Verdict.SYNTHETIC, Verdict.MANIPULATED] or risk_score_100 >= 70.0:
            risk_level = RiskLevel.HIGH
        elif risk_score_100 >= 30.0 or verdict == Verdict.UNCERTAIN or (active_scenario == RiskScenario.HIGH_VALUE_TRANSACTION and (risk_score_100 >= 5.0 or raw_s >= 0.25)):
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

        # 4. Gather forensic risk factors
        if fused.impersonation_penalty > 0:
            risk_factors.append(f"Speaker biometric mismatch (Impersonation penalty +{fused.impersonation_penalty * 100:.0f}%)")

        return RiskAssessment(
            verdict=verdict,
            ai_probability=float(round(prob, 4)),
            raw_score=float(round(raw_s, 4)),
            dynamic_risk_score=risk_score_100,
            risk_level=risk_level,
            confidence=float(round(conf, 4)),
            window_consistency=float(round(cons, 4)),
            verdict_category=category,
            verdict_explanation=explanation,
            scenario=active_scenario,
            recommended_action=action,
            risk_factors=risk_factors,
            fused_score_details=fused,
            calibrated_details=calibrated
        )

    def create_no_speech_assessment(
        self,
        scenario: Optional[RiskScenario] = None,
        reason: str = "Audio contains no active speech phonation."
    ) -> RiskAssessment:
        active_scenario = scenario or self.scenario
        dummy_fused = FusedScore(raw_fused_score=0.0)
        dummy_calibrated = CalibratedResult(
            raw_score=0.0,
            calibrated_probability=0.0,
            confidence=0.0,
            window_consistency=1.0,
            modality_agreement_ratio=1.0,
            verdict_category="No Speech Detected"
        )
        return RiskAssessment(
            verdict=Verdict.NO_SPEECH,
            ai_probability=0.0,
            raw_score=0.0,
            dynamic_risk_score=0.0,
            risk_level=RiskLevel.LOW,
            confidence=0.0,
            window_consistency=1.0,
            verdict_category="No Speech Detected",
            verdict_explanation=reason,
            scenario=active_scenario,
            recommended_action="AWAIT_CALLER_SPEECH",
            risk_factors=[reason],
            fused_score_details=dummy_fused,
            calibrated_details=dummy_calibrated
        )
