from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .ingestion.config import IngestionConfig
from .ingestion.pipeline import AudioIngestionPipeline
from .ingestion.models import IngestedAudio, AudioChunk
from .analysis.manager import ParallelAnalysisManager, ParallelAnalysisOutput
from .features.extractor import FeatureExtractor
from .features.normalizer import FeatureNormalizer
from .detector.classifier import DeepfakeClassifier, DeepfakePrediction
from .verification.verifier import SpeakerVerifier, SpeakerVerificationResult
from .verification.database import SpeakerDatabase
from .fusion.fusion_engine import ScoreFusionEngine, FusedScore
from .fusion.calibrator import ConfidenceCalibrator, CalibratedResult
from .fusion.risk_engine import RiskScoringEngine, RiskAssessment, RiskScenario, Verdict, RiskLevel
from .alerting.notifier import AlertDispatcher, AlertPayload
from .alerting.workflow import InterventionWorkflow


class VerificationOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    assessment: RiskAssessment
    ingested_audio: IngestedAudio
    parallel_analysis: ParallelAnalysisOutput
    ml_prediction: DeepfakePrediction
    speaker_verification: SpeakerVerificationResult
    window_predictions: List[float] = Field(default_factory=list)
    window_consistency: float = 1.0
    alert: Optional[AlertPayload] = None
    debug_info: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"==================================================================",
            f"  VOICE INTEGRITY VERIFICATION FINAL REPORT [ID: {self.session_id[:8]}]",
            f"==================================================================",
            f"• FINAL VERDICT        : [{self.assessment.verdict.value}] ({self.assessment.verdict_category})",
            f"• CALIBRATED AI PROB   : {self.assessment.ai_probability * 100:.1f}%",
            f"• EVIDENCE CONFIDENCE  : {self.assessment.confidence * 100:.1f}%",
            f"• WINDOW CONSISTENCY   : {self.window_consistency * 100:.1f}%",
            f"• RAW MODEL SCORE      : {self.assessment.raw_score:.3f}",
            f"• DYNAMIC RISK SCORE   : {self.assessment.dynamic_risk_score:.1f} / 100.0 (Risk Tier: {self.assessment.risk_level.value})",
            f"• ACTION REQUIRED      : {self.assessment.recommended_action}",
            f"• VERDICT EXPLANATION  : {self.assessment.verdict_explanation}",
            f"------------------------------------------------------------------",
            f"• Window Predictions   : {[round(p * 100, 1) for p in self.window_predictions]} (%)" if self.window_predictions else "• Window Predictions   : Single window",
            f"• Speaker Match        : {'YES' if self.speaker_verification.is_match else 'NO / MISMATCH'} (Similarity: {self.speaker_verification.similarity_score * 100:.1f}%)",
            f"• Checkpoint Loaded    : {'YES (Trained Checkpoint)' if self.ml_prediction.is_checkpoint_loaded else 'NO (Baseline Distributions)'}",
            f"• Forensic Findings    : {', '.join(self.assessment.risk_factors) if self.assessment.risk_factors else 'Natural human voice production'}",
            f"=================================================================="
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "verdict": self.assessment.verdict.value,
            "verdict_category": self.assessment.verdict_category,
            "verdict_explanation": self.assessment.verdict_explanation,
            "ai_probability": self.assessment.ai_probability,
            "calibrated_probability": self.assessment.ai_probability,
            "confidence": self.assessment.confidence,
            "raw_score": self.assessment.raw_score,
            "window_consistency": self.window_consistency,
            "window_predictions": self.window_predictions,
            "dynamic_risk_score": self.assessment.dynamic_risk_score,
            "risk_level": self.assessment.risk_level.value,
            "scenario": self.assessment.scenario.value,
            "recommended_action": self.assessment.recommended_action,
            "risk_factors": self.assessment.risk_factors,
            "ml_synthetic_probability": self.ml_prediction.synthetic_probability,
            "is_checkpoint_loaded": self.ml_prediction.is_checkpoint_loaded,
            "speaker_verification": {
                "claimed_speaker": self.speaker_verification.claimed_speaker_id,
                "is_match": self.speaker_verification.is_match,
                "similarity": self.speaker_verification.similarity_score
            },
            "forensics": {
                "sha256": self.ingested_audio.forensic.sha256_hash,
                "verified": self.ingested_audio.forensic.verified_integrity
            },
            "debug_info": self.debug_info
        }


class VoiceIntegrityEngine:
    """
    Master orchestrator tying Ingestion, Multi-Window Temporal Analysis,
    Parallel DSP Analyzers, Neural Deepfake Classifier, Speaker Biometrics,
    Platt Calibration, and Risk Scoring into a unified pipeline.
    """

    def __init__(
        self,
        ingestion_config: Optional[IngestionConfig] = None,
        default_scenario: RiskScenario = RiskScenario.GENERAL_TELEPHONY,
        speaker_db: Optional[SpeakerDatabase] = None,
        checkpoint_path: Optional[Union[str, Path]] = None
    ):
        self.ingestion_pipeline = AudioIngestionPipeline(config=ingestion_config)
        self.analysis_manager = ParallelAnalysisManager()
        self.feature_extractor = FeatureExtractor()
        self.feature_normalizer = FeatureNormalizer()
        self.classifier = DeepfakeClassifier(
            normalizer=self.feature_normalizer,
            checkpoint_path=checkpoint_path
        )
        self.speaker_database = speaker_db or SpeakerDatabase()
        self.speaker_verifier = SpeakerVerifier(database=self.speaker_database)
        self.fusion_engine = ScoreFusionEngine()
        self.calibrator = ConfidenceCalibrator(
            platt_a=self.classifier.platt_a,
            platt_b=self.classifier.platt_b
        )
        self.risk_engine = RiskScoringEngine(scenario=default_scenario)
        self.alert_workflow = InterventionWorkflow()

    def verify_file(
        self,
        file_path: Union[str, Path],
        claimed_speaker_id: Optional[str] = None,
        scenario: Optional[RiskScenario] = None
    ) -> VerificationOutput:
        # Step 1: Decode and take cryptographic fingerprints
        ingested = self.ingestion_pipeline.process_file(file_path)

        # Silence / Phonation Gate: Reject silence or non-speech/music audio immediately
        is_silent = (
            not ingested.contains_speech or
            ingested.speech_duration_sec < 0.25 or
            np.max(np.abs(ingested.processed_audio)) < 1e-4 or
            ingested.metadata.rms_dbfs < -55.0
        )

        if is_silent:
            if np.max(np.abs(ingested.processed_audio)) < 0.005 or ingested.metadata.rms_dbfs < -52.0:
                reason = "Audio contains silence or no active audio signal."
            else:
                reason = "Non-speech audio detected (Music / Instrumental / Noise - no active human speech phonation)."

            assessment = self.risk_engine.create_no_speech_assessment(
                scenario=scenario,
                reason=reason
            )
            analysis_out = self.analysis_manager.analyze_ingested(ingested)
            ml_pred = DeepfakePrediction(
                raw_score=0.0,
                synthetic_probability=0.0,
                is_synthetic=False,
                confidence=0.0,
                is_checkpoint_loaded=self.classifier.is_checkpoint_loaded
            )
            speaker_res = SpeakerVerificationResult(
                is_match=False,
                similarity_score=0.0,
                threshold=0.75,
                impersonation_detected=False,
                confidence=0.0,
                claimed_speaker_id=claimed_speaker_id
            )
            debug_info = {
                "verdict": "NO_SPEECH",
                "verdict_category": "No Speech Detected",
                "ai_probability": 0.0,
                "confidence": 0.0,
                "reason": "Silence / Phonation Gate triggered"
            }
            return VerificationOutput(
                session_id=ingested.audio_id,
                assessment=assessment,
                ingested_audio=ingested,
                parallel_analysis=analysis_out,
                ml_prediction=ml_pred,
                speaker_verification=speaker_res,
                window_predictions=[0.0],
                window_consistency=1.0,
                alert=None,
                debug_info=debug_info
            )

        # Step 2: Run global DSP feature extraction
        analysis_out = self.analysis_manager.analyze_ingested(ingested)

        # Step 3: Multi-Window Temporal Analysis
        # Split audio into sliding windows to evaluate consistency across time
        speech_chunks = [c for c in ingested.chunks if c.contains_speech and len(c.samples) > 0]
        window_predictions = []
        window_raw_scores = []

        if len(speech_chunks) >= 2:
            for chunk in speech_chunks:
                chunk_analysis = self.analysis_manager.analyze_chunk(chunk)
                chunk_feat = self.feature_extractor.extract_vector(chunk_analysis)
                chunk_pred = self.classifier.predict(chunk_feat)
                window_predictions.append(chunk_pred.synthetic_probability)
                window_raw_scores.append(chunk_pred.raw_score)

            # Robust aggregation across temporal windows: Median
            agg_prob = float(np.median(window_predictions))
            agg_raw = float(np.median(window_raw_scores))
            std_dev = float(np.std(window_predictions))
            # Consistency: 1.0 if identical, drops as variance between windows increases
            window_consistency = float(np.clip(1.0 - 2.0 * std_dev, 0.0, 1.0))
        else:
            global_feat = self.feature_extractor.extract_vector(analysis_out)
            global_pred = self.classifier.predict(global_feat)
            agg_prob = global_pred.synthetic_probability
            agg_raw = global_pred.raw_score
            window_consistency = 1.0
            window_predictions = [agg_prob]
            window_raw_scores = [agg_raw]

        # Construct unified ML prediction
        global_feat = self.feature_extractor.extract_vector(analysis_out)
        global_pred = self.classifier.predict(global_feat)
        ml_pred = DeepfakePrediction(
            raw_score=agg_raw,
            synthetic_probability=agg_prob,
            is_synthetic=bool(agg_prob >= self.classifier.threshold),
            confidence=float(np.clip(0.5 + 2.0 * abs(agg_prob - 0.5) * 0.5 * window_consistency, 0.1, 1.0)),
            is_checkpoint_loaded=self.classifier.is_checkpoint_loaded,
            top_contributing_features=global_pred.top_contributing_features
        )

        # Step 4: Speaker Biometric Verification
        speaker_res = self.speaker_verifier.verify_audio(
            audio=ingested.processed_audio,
            sample_rate=ingested.target_sample_rate,
            claimed_speaker_id=claimed_speaker_id
        )

        # Step 5: Multi-Modal Score Fusion
        fused = self.fusion_engine.fuse(
            analysis=analysis_out,
            ml_prediction=ml_pred,
            speaker_res=speaker_res
        )

        # Step 6: Probability Calibration with Window Consistency
        calibrated = self.calibrator.calibrate(
            fused=fused,
            analysis=analysis_out,
            speech_duration_sec=ingested.speech_duration_sec,
            window_consistency=window_consistency,
            window_scores=window_predictions
        )

        # Step 7: Risk Scoring and Intervention
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

        debug_info = {
            "window_count": len(window_predictions),
            "window_predictions": [float(round(p, 4)) for p in window_predictions],
            "window_consistency": float(round(window_consistency, 4)),
            "raw_fused_score": fused.raw_fused_score,
            "calibrated_ai_probability": calibrated.calibrated_probability,
            "confidence": calibrated.confidence,
            "modality_agreement_ratio": calibrated.modality_agreement_ratio,
            "active_scenario": assessment.scenario.value,
            "verdict": assessment.verdict.value,
            "verdict_category": assessment.verdict_category,
            "is_checkpoint_loaded": self.classifier.is_checkpoint_loaded
        }

        return VerificationOutput(
            session_id=ingested.audio_id,
            assessment=assessment,
            ingested_audio=ingested,
            parallel_analysis=analysis_out,
            ml_prediction=ml_pred,
            speaker_verification=speaker_res,
            window_predictions=window_predictions,
            window_consistency=window_consistency,
            alert=alert,
            debug_info=debug_info
        )

    def verify_stream_chunk(
        self,
        chunk: AudioChunk,
        claimed_speaker_id: Optional[str] = None,
        scenario: Optional[RiskScenario] = None
    ) -> RiskAssessment:
        # Silence check for incoming chunk
        peak = float(np.max(np.abs(chunk.samples))) if len(chunk.samples) > 0 else 0.0
        rms = float(np.sqrt(np.mean(chunk.samples ** 2) + 1e-12))
        rms_dbfs = 20.0 * np.log10(max(rms, 1e-9))

        if not chunk.contains_speech or peak < 0.015 or chunk.speech_ratio < 0.10 or rms_dbfs < -50.0:
            if peak < 0.01 or rms_dbfs < -50.0:
                reason = "Incoming window contains silence / no active audio signal."
            else:
                reason = "Incoming window contains non-speech audio (music / instrumental / noise). No human voice detected."

            return self.risk_engine.create_no_speech_assessment(
                scenario=scenario,
                reason=reason
            )

        # Fast-lane evaluation for live audio chunks
        analysis_out = self.analysis_manager.analyze_chunk(chunk)
        feat_vector = self.feature_extractor.extract_vector(analysis_out)
        ml_pred = self.classifier.predict(feat_vector)
        speaker_res = self.speaker_verifier.verify_audio(
            audio=chunk.samples,
            sample_rate=chunk.sample_rate,
            claimed_speaker_id=claimed_speaker_id
        )
        fused = self.fusion_engine.fuse(analysis_out, ml_pred, speaker_res)
        calibrated = self.calibrator.calibrate(
            fused=fused,
            analysis=analysis_out,
            speech_duration_sec=chunk.duration_sec,
            window_consistency=1.0
        )
        return self.risk_engine.assess_risk(fused, calibrated, analysis_out, scenario=scenario)
