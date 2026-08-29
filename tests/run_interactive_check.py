#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf

# Force UTF-8 so Windows cmd doesn't choke on fancy symbols
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import VoiceIntegrityEngine, VerificationOutput
from src.fusion.risk_engine import RiskScenario, RiskLevel, Verdict
from src.ingestion.models import VADSegment


def render_timeline_bar(segments: list[VADSegment], total_duration: float, width: int = 40) -> str:
    # Quick ASCII timeline: '#' is speech, '-' is awkward silence
    if total_duration <= 0:
        return "-" * width
    bar = ["-"] * width
    for seg in segments:
        if seg.is_speech:
            start_idx = int((seg.start_seconds / total_duration) * width)
            end_idx = min(width, int((seg.end_seconds / total_duration) * width) + 1)
            for i in range(start_idx, end_idx):
                if 0 <= i < width:
                    bar[i] = "#"
    return "".join(bar)


def print_master_dashboard(output: VerificationOutput, title: str = "VOICE INTEGRITY VERIFICATION REPORT"):
    ass = output.assessment
    meta = output.ingested_audio.metadata
    forensic = output.ingested_audio.forensic
    ana = output.parallel_analysis
    ml = output.ml_prediction
    spk = output.speaker_verification

    print("\n" + "=" * 80)
    print(f"  [>] {title.center(72)}  ")
    print("=" * 80)

    # 1. Verdict
    print(f"\n🏆 [1. FINAL AUTHENTICITY VERDICT & RISK ASSESSMENT]")
    verdict_badge = f"[{ass.verdict.value}]"
    risk_badge = f"[{ass.risk_level.value}]"
    print(f"  * Verdict              : {verdict_badge} (Confidence: {ass.confidence * 100:.1f}%)")
    print(f"  * Dynamic Risk Score   : {ass.dynamic_risk_score:.1f} / 100.0  Tier: {risk_badge}")
    print(f"  * Operational Scenario : {ass.scenario.value.upper()}")
    print(f"  * Prescriptive Action  : {ass.recommended_action}")

    # 2. Forensic Record
    print(f"\n🔒 [2. INGESTION & FORENSIC CHAIN OF CUSTODY]")
    print(f"  * Session ID           : {output.session_id}")
    print(f"  * Source File          : {meta.file_path or forensic.original_filename}")
    print(f"  * Format / Codec       : {meta.container_format.upper()} ({meta.codec}) | {meta.duration_seconds:.2f}s | {meta.sample_rate:,} Hz")
    print(f"  * SHA-256 Checksum     : {forensic.sha256_hash}")
    print(f"  * Integrity Status     : {'[PASS] VERIFIED INTACT (UNMODIFIED)' if forensic.verified_integrity else '[FAIL] CORRUPTED'}")

    # 3. VAD
    print(f"\n🗣️ [3. VOICE ACTIVITY DETECTION & SEGMENTATION]")
    print(f"  * Active Phonation     : {output.ingested_audio.speech_duration_sec:.2f}s ({output.ingested_audio.speech_ratio * 100:.1f}%) | Silence: {output.ingested_audio.silence_duration_sec:.2f}s")
    timeline = render_timeline_bar(output.ingested_audio.vad_segments, meta.duration_seconds, width=40)
    print(f"  * Speech Timeline      : [{timeline}]  (# = Speech, - = Silence)")

    # 4. Feature Branches
    print(f"\n🔬 [4. PARALLEL FEATURE EXTRACTION MODULES]")
    print(f"  ┌─────────────────────────┬──────────────┬─────────────────────────────────────────────────┐")
    print(f"  │ Analysis Branch         │ Anomaly Risk │ Key Extracted Properties                        │")
    print(f"  ├─────────────────────────┼──────────────┼─────────────────────────────────────────────────┤")
    print(f"  │ 1. Acoustic & Pitch     │ {ana.acoustic.anomaly_score * 100:5.1f}%       │ F0: {ana.acoustic.pitch.mean_f0_hz:.0f}Hz | Jitter: {ana.acoustic.pitch.jitter_local_pct:.2f}% | Shimmer: {ana.acoustic.pitch.shimmer_local_pct:.2f}%   │")
    print(f"  │ 2. Spectral Dynamics    │ {ana.spectral.anomaly_score * 100:5.1f}%       │ Centroid: {ana.spectral.moments.centroid_mean_hz:.0f}Hz | Flatness: {ana.spectral.dynamics.flatness_mean:.3f} | HF Cut: {ana.spectral.hf_analysis.has_artificial_cutoff} │")
    print(f"  │ 3. Prosody & Rhythm     │ {ana.prosody.anomaly_score * 100:5.1f}%       │ Rate: {ana.prosody.rhythm.speaking_rate_sps:.1f} syll/s | nPVI: {ana.prosody.rhythm.npvi:.1f} | Monotone: {ana.prosody.intonation.is_monotone}   │")
    print(f"  │ 4. Synthesis Artifacts  │ {ana.synthesis_artifacts.anomaly_score * 100:5.1f}%       │ Vocoder 2D Periodic: {ana.synthesis_artifacts.vocoder_artifacts.periodic_artifact_detected} | Splices: {ana.synthesis_artifacts.concatenation_artifacts.splice_points_detected}       │")
    print(f"  │ 5. Phase Coherence      │ {ana.phase.anomaly_score * 100:5.1f}%       │ IF Cluster: {ana.phase.instantaneous_frequency.if_harmonic_clustering_score:.2f} | MGD Prom: {ana.phase.group_delay.mgd_peak_prominence:.2f}          │")
    print(f"  └─────────────────────────┴──────────────┴─────────────────────────────────────────────────┘")

    # 5. ML & Biometrics
    print(f"\n🤖 [5. ML DEEPFAKE DETECTOR & SPEAKER BIOMETRICS]")
    print(f"  * Deepfake Probability : {ml.synthetic_probability * 100:.1f}% ({'SYNTHETIC CLASSIFICATION' if ml.is_synthetic else 'GENUINE CLASSIFICATION'})")
    if spk.claimed_speaker_id:
        print(f"  * Speaker Verification : Claimed: '{spk.claimed_speaker_id}' ({spk.enrolled_speaker_name or 'Unknown'})")
        print(f"  * Biometric Match      : {'[MATCH]' if spk.is_match else '[MISMATCH / IMPERSONATION]'} (Similarity: {spk.similarity_score * 100:.1f}%)")
    else:
        print(f"  * Speaker Verification : [ANONYMOUS / NO SPEAKER CLAIM]")

    # 6. Risk Drivers
    print(f"\n🚨 [6. FORENSIC RISK DRIVERS & ANOMALIES]")
    if ass.risk_factors:
        for factor in ass.risk_factors:
            print(f"  [!] {factor}")
    else:
        print("  [*] No synthetic or manipulative anomalies detected. Phonation matches biological human baseline.")

    if output.alert:
        print(f"\n🔔 [7. AUTOMATED INTERVENTION DISPATCH]")
        print(f"  * Alert Dispatched     : ID #{output.alert.alert_id[:8]} -> Channels: {', '.join(output.alert.target_channels)}")

    print("\n" + "=" * 80)


def generate_sample_suite(output_dir: Path) -> list[Path]:
    # Synthesize multi-format test samples so we have something to chew on
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = []

    # 1. 48kHz WAV speech simulation
    wav_path = output_dir / "sample_voice_call_48k.wav"
    sr = 48000
    dur = 4.5
    total_samples = int(dur * sr)
    t = np.linspace(0, dur, total_samples, endpoint=False)
    
    f0 = 135.0 + 15.0 * np.sin(2 * np.pi * 1.8 * t)
    phase = np.cumsum(2 * np.pi * f0 / sr)
    sig = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase) + 0.15 * np.sin(5 * phase)
    
    env = np.zeros_like(t)
    env[(t >= 0.0) & (t < 1.5)] = 0.75
    env[(t >= 2.2) & (t < 3.8)] = 0.85
    sig = sig * env + np.random.normal(0, 0.003, total_samples)

    stereo = np.column_stack([sig * 0.9, sig * 0.85]).astype(np.float32)
    sf.write(str(wav_path), stereo, sr, format="WAV", subtype="PCM_16")
    samples.append(wav_path)

    # 2. MP3 compressed
    try:
        mp3_path = output_dir / "sample_compressed_speech.mp3"
        cmd = ["ffmpeg", "-v", "quiet", "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "128k", str(mp3_path)]
        subprocess.run(cmd, check=True)
        samples.append(mp3_path)
    except Exception:
        pass

    # 3. M4A / AAC
    try:
        m4a_path = output_dir / "sample_voip_aac.m4a"
        cmd = ["ffmpeg", "-v", "quiet", "-y", "-i", str(wav_path), "-codec:a", "aac", "-b:a", "128k", str(m4a_path)]
        subprocess.run(cmd, check=True)
        samples.append(m4a_path)
    except Exception:
        pass

    return samples


def run_streaming_simulation(scenario: RiskScenario = RiskScenario.GENERAL_TELEPHONY, claimed_speaker: str = None):
    print("\n" + "=" * 80)
    print("  [LIVE REAL-TIME STREAMING VOIP DEEPFAKE SCORING RECEIVER]".center(80))
    print("=" * 80)
    print(f"Active Scenario: {scenario.value.upper()} | Claimed Identity: {claimed_speaker or 'None'}")
    print("Simulating arrival of 100ms real-time incoming voice packets @ 16kHz...\n")

    engine = VoiceIntegrityEngine()
    stream_buffer = engine.ingestion_pipeline.create_streaming_buffer(
        window_duration_sec=3.0,
        hop_duration_sec=1.0
    )

    packet_samples = 1600
    total_packets = 50
    emitted_count = 0

    for p_idx in range(1, total_packets + 1):
        elapsed = p_idx * 0.1
        t = np.linspace(0, 0.1, packet_samples, endpoint=False)
        
        # Mid-call caller switches from synthetic voice to a spliced phrase
        if p_idx < 30:
            freq = 140.0
            packet = (0.6 * np.sin(2 * np.pi * freq * (elapsed + t))).astype(np.float32)
        else:
            packet = (0.8 * np.sin(2 * np.pi * 320 * (elapsed + t))).astype(np.float32)

        start_t = time.perf_counter()
        new_chunks = stream_buffer.add_samples(packet)
        proc_latency_ms = (time.perf_counter() - start_t) * 1000.0

        time.sleep(0.04)
        sys.stdout.write(f"\rIncoming Packet #{p_idx:02d}/{total_packets} ({elapsed:.1f}s) | Buffer: {len(stream_buffer.buffer):,}/48,000 samples | Windows Scored: {stream_buffer.chunk_count}")
        sys.stdout.flush()

        if new_chunks:
            print()
            for chunk in new_chunks:
                emitted_count += 1
                assessment = engine.verify_stream_chunk(chunk, claimed_speaker_id=claimed_speaker, scenario=scenario)
                print(f"   ⚡ [INCOMING WINDOW #{chunk.chunk_index}: {chunk.start_time_sec:4.2f}s - {chunk.end_time_sec:4.2f}s | Latency: {proc_latency_ms:4.1f}ms]")
                print(f"      • Verdict: [{assessment.verdict.value}] | Dynamic Risk: {assessment.dynamic_risk_score:5.1f}/100 | Tier: [{assessment.risk_level.value}]")
                print(f"      • Action:  {assessment.recommended_action}")
                if assessment.risk_factors:
                    print(f"      • Triggers: {', '.join(assessment.risk_factors[:2])}")

    print(f"\n\n[SUCCESS] Live stream receiver simulation completed. Windows evaluated: {emitted_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Voice Cloning & Deepfake Detection Engine"
    )
    parser.add_argument("--stream", action="store_true", help="Run live real-time incoming voice stream receiver")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to received audio/voicemail file to inspect")
    parser.add_argument("--scenario", "-s", type=str, default="general_telephony", choices=["high_value_transaction", "confidential_disclosure", "standard_support", "general_telephony"], help="Operational risk scenario")
    parser.add_argument("--speaker", type=str, default=None, help="Claimed speaker ID for biometric check")
    parser.add_argument("--enroll", type=str, default=None, help="Audio file path to enroll as genuine reference")
    parser.add_argument("--speaker-id", type=str, default=None, help="Speaker ID for enrollment")
    parser.add_argument("--speaker-name", type=str, default="Enrolled Speaker", help="Speaker Name for enrollment")
    parser.add_argument("--server", action="store_true", help="Launch FastAPI REST server with Web UI")
    parser.add_argument("--port", type=int, default=8000, help="Port for server (default: 8000)")
    parser.add_argument("--export-dir", "-o", type=str, default=str(PROJECT_ROOT / "output"), help="Output directory for reports")

    args = parser.parse_args()

    if args.server:
        print(f"\n🚀 Launching Voice Integrity Web App & API on http://localhost:{args.port} ...")
        print(f"👉 Open in browser: http://localhost:{args.port}")
        print(f"📖 Swagger Docs: http://localhost:{args.port}/docs\n")
        import uvicorn
        from src.api.server import app
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        return

    if args.stream:
        run_streaming_simulation(
            scenario=RiskScenario(args.scenario),
            claimed_speaker=args.speaker
        )
        return

    export_path = Path(args.export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    engine = VoiceIntegrityEngine(default_scenario=RiskScenario(args.scenario))

    if args.enroll:
        enroll_audio_path = Path(args.enroll)
        if not enroll_audio_path.exists():
            print(f"[ERROR] Enrollment audio file not found: {args.enroll}", file=sys.stderr)
            sys.exit(1)
        spk_id = args.speaker_id or "SPEAKER_001"
        ingested = engine.ingestion_pipeline.process_file(enroll_audio_path)
        emb = engine.speaker_verifier.extractor.extract_embedding(ingested.processed_audio, ingested.target_sample_rate)
        enrolled = engine.speaker_database.enroll(speaker_id=spk_id, name=args.speaker_name, embedding=emb)
        print(f"\n✅ Speaker Successfully Enrolled!")
        print(f"  • Speaker ID   : {enrolled.speaker_id}")
        print(f"  • Name         : {enrolled.name}")
        print(f"  • Sample Count : {enrolled.sample_count}")
        print(f"  • Enrolled At  : {enrolled.enrolled_at}\n")
        return

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[ERROR] File not found: {args.input}", file=sys.stderr)
            sys.exit(1)

        print(f"\n[INFO] Running Full Voice Integrity Verification on: {input_path}")
        output = engine.verify_file(
            file_path=input_path,
            claimed_speaker_id=args.speaker,
            scenario=RiskScenario(args.scenario)
        )
        print_master_dashboard(output, title=f"VERIFICATION REPORT: {input_path.name}")

        json_path = export_path / f"{input_path.stem}_full_report.json"
        with open(json_path, "w") as f:
            json.dump(output.to_dict(), f, indent=2)
        print(f"💾 Full Verification Report saved to: {json_path}")

        wav_path = export_path / f"{input_path.stem}_preprocessed_16k.wav"
        output.ingested_audio.save_processed_wav(wav_path)
        print(f"💾 Preprocessed 16kHz WAV saved to: {wav_path}\n")

    else:
        print("\n" + "=" * 80)
        print("  [EXECUTING MASTER VERIFICATION SUITE ON RECEIVED AUDIO CALL SAMPLES]".center(80))
        print("=" * 80)

        samples_dir = PROJECT_ROOT / "samples"
        test_samples = generate_sample_suite(samples_dir)

        for sample_path in test_samples:
            print(f"\nProcessing Received Call: {sample_path.name}...")
            output = engine.verify_file(sample_path, scenario=RiskScenario(args.scenario))
            print_master_dashboard(output, title=f"SAMPLE REPORT: {sample_path.name}")

            json_path = export_path / f"{sample_path.stem}_report.json"
            with open(json_path, "w") as f:
                json.dump(output.to_dict(), f, indent=2)

        print("\n" + "=" * 80)
        print("  [ALL VERIFICATION PIPELINES & SAMPLES COMPLETED SUCCESSFULLY!]".center(80))
        print("  To launch the Live Stream Interceptor Web UI:".center(80))
        print("  python tests/run_interactive_check.py --server".center(80))
        print("  To run real-time incoming VoIP stream evaluation in CLI:".center(80))
        print("  python tests/run_interactive_check.py --stream".center(80))
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
