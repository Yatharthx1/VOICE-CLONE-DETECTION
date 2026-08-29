from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.ingestion.config import IngestionConfig
from src.ingestion.pipeline import AudioIngestionPipeline
from src.ingestion.models import IngestedAudio, AudioChunk
from src.analysis.manager import ParallelAnalysisManager, ParallelAnalysisOutput
from src.features.extractor import FeatureExtractor
from src.features.normalizer import FeatureNormalizer
from src.detector.classifier import DeepfakeClassifier, DeepfakePrediction
from src.verification.verifier import SpeakerVerifier, SpeakerVerificationResult
from src.verification.database import SpeakerDatabase
from src.fusion.fusion_engine import ScoreFusionEngine, FusedScore
from src.fusion.calibrator import ConfidenceCalibrator, CalibratedResult
from src.fusion.risk_engine import RiskScoringEngine, RiskAssessment, RiskScenario, Verdict, RiskLevel
from src.alerting.notifier import AlertDispatcher, AlertPayload
from src.alerting.workflow import InterventionWorkflow


class VerificationOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    assessment: RiskAssessment
    ingested_audio: IngestedAudio
    parallel_analysis: ParallelAnalysisOutput
    ml_prediction: DeepfakePrediction
    speaker_verification: SpeakerVerificationResult
    alert: Optional[AlertPayload] = None

    def summary(self) -> str:
        lines = [
            f"==================================================================",
            f"  VOICE INTEGRITY VERIFICATION FINAL REPORT [ID: {self.session_id[:8]}]",
            f"==================================================================",
            f"• FINAL VERDICT      : [{self.assessment.verdict.value}]",
            f"• DYNAMIC RISK SCORE : {self.assessment.dynamic_risk_score:.1f} / 100.0 (Risk Tier: {self.assessment.risk_level.value})",
            f"• CONFIDENCE         : {self.assessment.confidence * 100:.1f}%",
            f"• ACTION REQUIRED    : {self.assessment.recommended_action}",
            f"------------------------------------------------------------------",
            f"• Speaker Match      : {'YES' if self.speaker_verification.is_match else 'NO / MISMATCH'} (Similarity: {self.speaker_verification.similarity_score * 100:.1f}%)",
            f"• ML Detector Score  : {self.ml_prediction.synthetic_probability * 100:.1f}% synthetic probability",
            f"• Forensic Findings  : {', '.join(self.assessment.risk_factors) if self.assessment.risk_factors else 'Natural human voice production'}",
            f"=================================================================="
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "verdict": self.assessment.verdict.value,
            "dynamic_risk_score": self.assessment.dynamic_risk_score,
            "risk_level": self.assessment.risk_level.value,
            "confidence": self.assessment.confidence,
            "scenario": self.assessment.scenario.value,
            "recommended_action": self.assessment.recommended_action,
            "risk_factors": self.assessment.risk_factors,
            "ml_synthetic_probability": self.ml_prediction.synthetic_probability,
            "speaker_verification": {
                "claimed_speaker": self.speaker_verification.claimed_speaker_id,
                "is_match": self.speaker_verification.is_match,
                "similarity": self.speaker_verification.similarity_score
            },
            "forensics": {
                "sha256": self.ingested_audio.forensic.sha256_hash,
                "verified": self.ingested_audio.forensic.verified_integrity
            }
        }


class VoiceIntegrityEngine:
    # The grand conductor tying everything together so scammers can't clone grandma

    def __init__(
        self,
        ingestion_config: Optional[IngestionConfig] = None,
        default_scenario: RiskScenario = RiskScenario.GENERAL_TELEPHONY,
        speaker_db: Optional[SpeakerDatabase] = None
    ):
        # Fire up all the sub-engines and pray the GPU doesn't run out of memory
        self.ingestion_pipeline = AudioIngestionPipeline(config=ingestion_config)
        self.analysis_manager = ParallelAnalysisManager()
        self.feature_extractor = FeatureExtractor()
        self.feature_normalizer = FeatureNormalizer()
        self.classifier = DeepfakeClassifier(normalizer=self.feature_normalizer)
        self.speaker_database = speaker_db or SpeakerDatabase()
        self.speaker_verifier = SpeakerVerifier(database=self.speaker_database)
        self.fusion_engine = ScoreFusionEngine()
        self.calibrator = ConfidenceCalibrator()
        self.risk_engine = RiskScoringEngine(scenario=default_scenario)
        self.alert_workflow = InterventionWorkflow()

    def verify_file(
        self,
        file_path: Union[str, Path],
        claimed_speaker_id: Optional[str] = None,
        scenario: Optional[RiskScenario] = None
    ) -> VerificationOutput:
        # Step 1: Decode the file and take cryptographic fingerprints before anyone messes with it
        ingested = self.ingestion_pipeline.process_file(file_path)

        # Step 2: Run all 5 DSP feature extractors concurrently
        analysis_out = self.analysis_manager.analyze_ingested(ingested)

        # Step 3: Flatten all the fancy acoustics into a clean 60-D vector
        feat_vector = self.feature_extractor.extract_vector(analysis_out)

        # Step 4: Ask the neural net if this smells like a synthetic clone
        ml_pred = self.classifier.predict(feat_vector)

        # Step 5: Check if the voice actually matches the claimed VIP or if it's an imposter
        speaker_res = self.speaker_verifier.verify_audio(
            audio=ingested.processed_audio,
            sample_rate=ingested.target_sample_rate,
            claimed_speaker_id=claimed_speaker_id
        )

        # Step 6: Fuse all the evidence together and calibrate the confidence
        fused = self.fusion_engine.fuse(
            analysis=analysis_out,
            ml_prediction=ml_pred,
            speaker_res=speaker_res
        )
        calibrated = self.calibrator.calibrate(
            fused=fused,
            analysis=analysis_out,
            speech_duration_sec=ingested.speech_duration_sec
        )

        # Step 7: Decide whether to approve, ask for MFA, or sound the fraud alarm
        assessment = self.risk_engine.assess_risk(
            fused=fused,
            calibrated=calibrated,
            analysis=analysis_out,
            scenario=scenario
        )
        alert = self.alert_workflow.process_assessment(
            assessment=assessment,
            session_id=ingested.audio_id
        )

        return VerificationOutput(
            session_id=ingested.audio_id,
            assessment=assessment,
            ingested_audio=ingested,
            parallel_analysis=analysis_out,
            ml_prediction=ml_pred,
            speaker_verification=speaker_res,
            alert=alert
        )

    def verify_stream_chunk(
        self,
        chunk: AudioChunk,
        claimed_speaker_id: Optional[str] = None,
        scenario: Optional[RiskScenario] = None
    ) -> RiskAssessment:
        # Fast-lane evaluation for live audio chunks coming in over VoIP
        analysis_out = self.analysis_manager.analyze_chunk(chunk)
        feat_vector = self.feature_extractor.extract_vector(analysis_out)
        ml_pred = self.classifier.predict(feat_vector)
        speaker_res = self.speaker_verifier.verify_audio(
            audio=chunk.samples,
            sample_rate=chunk.sample_rate,
            claimed_speaker_id=claimed_speaker_id
        )
        fused = self.fusion_engine.fuse(analysis_out, ml_pred, speaker_res)
        calibrated = self.calibrator.calibrate(fused, analysis_out, speech_duration_sec=chunk.duration_sec)
        return self.risk_engine.assess_risk(fused, calibrated, analysis_out, scenario=scenario)
