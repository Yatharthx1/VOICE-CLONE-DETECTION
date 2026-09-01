import tempfile
import time
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
import numpy as np
import torch

from src.engine import VoiceIntegrityEngine
from src.fusion.risk_engine import RiskScenario
from src.ingestion.models import AudioChunk
from .schemas import (
    VerifyResponse,
    StreamChunkRequest,
    StreamChunkResponse,
    SpeakerEnrollResponse,
    HealthResponse
)

router = APIRouter(prefix="/api/v1")

# Global singleton so we don't reload PyTorch models on every HTTP request
_engine: Optional[VoiceIntegrityEngine] = None


def get_engine() -> VoiceIntegrityEngine:
    global _engine
    if _engine is None:
        _engine = VoiceIntegrityEngine()
    return _engine


@router.get("/health", response_model=HealthResponse)
async def health_check():
    engine = get_engine()
    return HealthResponse(
        status="HEALTHY",
        version="1.0.0",
        framework="Voice Integrity Verification Framework",
        gpu_available=torch.cuda.is_available(),
        enrolled_speakers_count=len(engine.speaker_database.list_speakers())
    )


@router.post("/verify", response_model=VerifyResponse)
async def verify_audio_file(
    file: UploadFile = File(...),
    claimed_speaker_id: Optional[str] = Form(None),
    scenario: str = Form("general_telephony")
):
    engine = get_engine()

    try:
        risk_scenario = RiskScenario(scenario)
    except ValueError:
        risk_scenario = RiskScenario.GENERAL_TELEPHONY

    # Write to a temp file because FFprobe / libsndfile need a real file descriptor
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        out = engine.verify_file(
            file_path=tmp_path,
            claimed_speaker_id=claimed_speaker_id,
            scenario=risk_scenario
        )
        return VerifyResponse(
            session_id=out.session_id,
            verdict=out.assessment.verdict.value,
            verdict_category=out.assessment.verdict_category,
            verdict_explanation=out.assessment.verdict_explanation,
            ai_probability=out.assessment.ai_probability,
            raw_score=out.assessment.raw_score,
            dynamic_risk_score=out.assessment.dynamic_risk_score,
            risk_level=out.assessment.risk_level.value,
            confidence=out.assessment.confidence,
            window_consistency=out.window_consistency,
            window_predictions=out.window_predictions,
            scenario=out.assessment.scenario.value,
            recommended_action=out.assessment.recommended_action,
            risk_factors=out.assessment.risk_factors,
            speaker_match=out.speaker_verification.is_match,
            speaker_similarity=out.speaker_verification.similarity_score,
            ml_synthetic_probability=out.ml_prediction.synthetic_probability,
            is_checkpoint_loaded=out.ml_prediction.is_checkpoint_loaded,
            forensics={
                "sha256": out.ingested_audio.forensic.sha256_hash,
                "verified": out.ingested_audio.forensic.verified_integrity
            }
        )
    finally:
        # Clean up temp file so our disk doesn't fill up
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


@router.post("/stream/chunk", response_model=StreamChunkResponse)
async def verify_stream_chunk(payload: StreamChunkRequest):
    engine = get_engine()
    samples = np.array(payload.samples, dtype=np.float32)

    try:
        risk_scenario = RiskScenario(payload.scenario)
    except ValueError:
        risk_scenario = RiskScenario.GENERAL_TELEPHONY

    peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
    rms = float(np.sqrt(np.mean(samples ** 2) + 1e-12))
    rms_dbfs = 20.0 * np.log10(max(rms, 1e-9))
    has_speech = bool(peak >= 0.015 and rms_dbfs > -45.0)

    chunk = AudioChunk(
        chunk_index=0,
        start_time_sec=0.0,
        end_time_sec=len(samples) / payload.sample_rate if payload.sample_rate > 0 else 0.0,
        sample_rate=payload.sample_rate,
        samples=samples,
        contains_speech=has_speech,
        speech_ratio=1.0 if has_speech else 0.0,
        is_padded=False
    )

    assessment = engine.verify_stream_chunk(
        chunk=chunk,
        claimed_speaker_id=payload.claimed_speaker_id,
        scenario=risk_scenario
    )

    return StreamChunkResponse(
        verdict=assessment.verdict.value,
        verdict_category=assessment.verdict_category,
        verdict_explanation=assessment.verdict_explanation,
        ai_probability=assessment.ai_probability,
        raw_score=assessment.raw_score,
        dynamic_risk_score=assessment.dynamic_risk_score,
        risk_level=assessment.risk_level.value,
        confidence=assessment.confidence,
        window_consistency=assessment.window_consistency,
        scenario=assessment.scenario.value,
        recommended_action=assessment.recommended_action,
        risk_factors=assessment.risk_factors
    )


@router.websocket("/stream/ws")
async def websocket_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    engine = get_engine()
    stream_buffer = engine.ingestion_pipeline.create_streaming_buffer(
        window_duration_sec=3.0,
        hop_duration_sec=1.0
    )
    
    scenario = RiskScenario.GENERAL_TELEPHONY
    claimed_speaker_id = None
    client_sample_rate = 16000

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                try:
                    import json
                    meta = json.loads(message["text"])
                    if "scenario" in meta:
                        try:
                            scenario = RiskScenario(meta["scenario"])
                        except ValueError:
                            pass
                    if "claimed_speaker_id" in meta:
                        claimed_speaker_id = meta["claimed_speaker_id"]
                    if "sample_rate" in meta:
                        client_sample_rate = int(meta["sample_rate"])
                        stream_buffer.set_input_sample_rate(client_sample_rate)
                    await websocket.send_json({
                        "type": "CONFIG_ACK",
                        "status": "READY",
                        "scenario": scenario.value,
                        "claimed_speaker": claimed_speaker_id,
                        "sample_rate": client_sample_rate
                    })
                except Exception as e:
                    await websocket.send_json({"type": "ERROR", "message": str(e)})

            elif "bytes" in message:
                start_t = time.perf_counter()
                raw_bytes = message["bytes"]
                
                # Handle Float32 vs Int16
                if len(raw_bytes) % 4 == 0:
                    samples = np.frombuffer(raw_bytes, dtype=np.float32).copy()
                else:
                    samples = (np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0).copy()

                if np.isnan(samples).any() or np.isinf(samples).any():
                    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

                new_windows = stream_buffer.add_samples(samples)

                if new_windows:
                    for window_chunk in new_windows:
                        assessment = engine.verify_stream_chunk(
                            chunk=window_chunk,
                            claimed_speaker_id=claimed_speaker_id,
                            scenario=scenario
                        )
                        latency_ms = (time.perf_counter() - start_t) * 1000.0

                        await websocket.send_json({
                            "type": "LIVE_VERDICT",
                            "chunk_index": window_chunk.chunk_index,
                            "start_time_sec": window_chunk.start_time_sec,
                            "end_time_sec": window_chunk.end_time_sec,
                            "verdict": assessment.verdict.value,
                            "verdict_category": assessment.verdict_category,
                            "verdict_explanation": assessment.verdict_explanation,
                            "ai_probability": assessment.ai_probability,
                            "raw_score": assessment.raw_score,
                            "dynamic_risk_score": assessment.dynamic_risk_score,
                            "risk_level": assessment.risk_level.value,
                            "confidence": assessment.confidence,
                            "window_consistency": assessment.window_consistency,
                            "recommended_action": assessment.recommended_action,
                            "risk_factors": assessment.risk_factors,
                            "latency_ms": round(latency_ms, 2)
                        })
                else:
                    await websocket.send_json({
                        "type": "STREAM_BUFFERING",
                        "samples_buffered": len(stream_buffer.buffer),
                        "windows_evaluated": stream_buffer.chunk_count
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "ERROR", "message": str(e)})
        except Exception:
            pass


@router.post("/speakers/enroll", response_model=SpeakerEnrollResponse)
async def enroll_speaker_file(
    speaker_id: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...)
):
    engine = get_engine()

    suffix = Path(file.filename).suffix if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        ingested = engine.ingestion_pipeline.process_file(tmp_path)
        embedding = engine.speaker_verifier.extractor.extract_embedding(
            ingested.processed_audio,
            ingested.target_sample_rate
        )
        enrolled = engine.speaker_database.enroll(
            speaker_id=speaker_id,
            name=name,
            embedding=embedding
        )
        return SpeakerEnrollResponse(
            speaker_id=enrolled.speaker_id,
            name=enrolled.name,
            sample_count=enrolled.sample_count,
            enrolled_at=enrolled.enrolled_at,
            status="SUCCESS"
        )
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


@router.get("/speakers")
async def list_speakers():
    engine = get_engine()
    return [
        {
            "speaker_id": s.speaker_id,
            "name": s.name,
            "sample_count": s.sample_count,
            "enrolled_at": s.enrolled_at,
        }
        for s in engine.speaker_database.list_speakers()
    ]
