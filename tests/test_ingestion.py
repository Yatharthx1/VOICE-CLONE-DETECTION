"""
Comprehensive Unit and Functional Tests for the Audio Ingestion Subsystem.
Covers metadata extraction, multi-format decoding, forensic hashing,
preprocessing, VAD segmentation, sliding window chunking, and error handling.
"""

import os
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from src.ingestion.config import IngestionConfig
from src.ingestion.forensic import ForensicPreserver
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.decoder import AudioDecoder, AudioDecodeError, CorruptAudioError
from src.ingestion.preprocessor import AudioPreprocessor
from src.ingestion.vad import VoiceActivityDetector
from src.ingestion.chunker import AudioChunker, StreamingAudioBuffer
from src.ingestion.pipeline import AudioIngestionPipeline


# ============================================================================
# 1. Forensic Preservation & Hash Integrity Tests
# ============================================================================

def test_forensic_preservation_file(clean_wav_stereo, temp_audio_dir):
    """Verify cryptographic hashing and forensic raw copy preservation."""
    preserver = ForensicPreserver(storage_dir=temp_audio_dir / "forensics")
    record = preserver.preserve_file(clean_wav_stereo)

    assert record.audio_id is not None
    assert len(record.sha256_hash) == 64
    assert len(record.md5_hash) == 32
    assert record.original_filename == clean_wav_stereo.name
    assert record.original_size_bytes == clean_wav_stereo.stat().st_size
    assert record.forensic_copy_path is not None
    assert os.path.exists(record.forensic_copy_path)
    assert record.verified_integrity is True

    # Check tamper verification
    assert ForensicPreserver.verify_integrity(record) is True

    # Tamper with forensic copy to ensure tamper detection works
    with open(record.forensic_copy_path, "ab") as f:
        f.write(b"TAMPER_PAYLOAD")

    assert ForensicPreserver.verify_integrity(record) is False


def test_forensic_preservation_bytes(temp_audio_dir):
    """Verify hashing and preservation of in-memory raw audio byte streams."""
    preserver = ForensicPreserver(storage_dir=temp_audio_dir / "forensics")
    fake_data = b"RIFF_TEST_HEADER_RAW_BYTES_FOR_FORENSIC_CHECK"
    record = preserver.preserve_bytes(fake_data, original_filename="stream.wav")

    assert record.original_size_bytes == len(fake_data)
    assert record.sha256_hash == preserver.compute_hashes_from_bytes(fake_data)[0]
    assert ForensicPreserver.verify_integrity(record) is True


# ============================================================================
# 2. Metadata Extraction Tests
# ============================================================================

def test_metadata_extraction_wav(clean_wav_stereo):
    """Verify container, codec, channel, and acoustic metrics extraction on WAV."""
    extractor = MetadataExtractor()
    meta = extractor.extract_from_file(clean_wav_stereo)

    assert "wav" in meta.container_format.lower()
    assert meta.sample_rate == 44100
    assert meta.channels == 2
    assert pytest.approx(meta.duration_seconds, abs=0.05) == 3.0
    assert meta.total_samples > 0
    assert -10.0 <= meta.peak_dbfs <= 0.0
    assert meta.rms_dbfs < 0.0
    assert meta.is_clipped is False


def test_clipping_detection(clipped_wav):
    """Verify that clipped audio triggers the is_clipped metadata flag."""
    extractor = MetadataExtractor()
    meta = extractor.extract_from_file(clipped_wav)
    assert meta.is_clipped is True
    assert meta.peak_dbfs >= -0.05


# ============================================================================
# 3. Multi-Format Audio Decoding Tests
# ============================================================================

def test_decode_wav(clean_wav_stereo):
    """Test decoding stereo WAV."""
    decoder = AudioDecoder()
    audio, sr, ch = decoder.decode_file(clean_wav_stereo)
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert sr == 44100
    assert ch == 2
    assert audio.shape[1] == 2


def test_decode_compressed_formats(compressed_mp3, compressed_m4a, compressed_flac):
    """Test decoding MP3, M4A (AAC), and FLAC audio formats."""
    decoder = AudioDecoder()

    for path in [compressed_mp3, compressed_m4a, compressed_flac]:
        audio, sr, ch = decoder.decode_file(path)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert sr > 0
        assert ch >= 1
        assert len(audio) > 0
        # Values should be normalized float in [-1.0, 1.0]
        assert np.max(np.abs(audio)) <= 1.05


# ============================================================================
# 4. Audio Preprocessing Tests
# ============================================================================

def test_preprocessing_mono_conversion():
    """Verify multi-channel stereo downmixing to 1D mono."""
    preprocessor = AudioPreprocessor()
    stereo_sig = np.ones((1000, 2), dtype=np.float32)
    stereo_sig[:, 0] = 0.4
    stereo_sig[:, 1] = 0.8

    mono = preprocessor.to_mono(stereo_sig)
    assert mono.ndim == 1
    assert len(mono) == 1000
    assert pytest.approx(mono[0], abs=1e-4) == 0.6


def test_preprocessing_resampling():
    """Verify polyphase resampling between standard sample rates."""
    preprocessor = AudioPreprocessor()
    
    # 48000 Hz to 16000 Hz (3:1 exact ratio)
    sig_48k = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 48000, endpoint=False)).astype(np.float32)
    res_16k = preprocessor.resample(sig_48k, orig_sr=48000, target_sr=16000)
    assert len(res_16k) == 16000

    # 44100 Hz to 16000 Hz
    sig_44k = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 44100, endpoint=False)).astype(np.float32)
    res_16k_from_44k = preprocessor.resample(sig_44k, orig_sr=44100, target_sr=16000)
    assert abs(len(res_16k_from_44k) - 16000) <= 2


def test_preprocessing_dc_removal_and_normalization():
    """Verify DC offset bias removal and loudness normalization."""
    cfg = IngestionConfig(target_sample_rate=16000, target_dbfs=-20.0, normalize_audio=True)
    preprocessor = AudioPreprocessor(config=cfg)

    # Audio with DC bias +0.3
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sig_with_dc = 0.2 * np.sin(2 * np.pi * 300 * t) + 0.3

    processed = preprocessor.preprocess(sig_with_dc, source_sample_rate=16000)

    # Mean should now be centered near 0.0
    assert abs(np.mean(processed)) < 0.05
    # Max peak should be within ceiling
    assert np.max(np.abs(processed)) <= 1.0


# ============================================================================
# 5. Voice Activity Detection (VAD) Tests
# ============================================================================

def test_vad_speech_segmentation(speech_wav):
    """Verify VAD correctly distinguishes speech from silence intervals."""
    audio, sr = sf.read(str(speech_wav), dtype="float32")
    detector = VoiceActivityDetector()

    segments, speech_sec, silence_sec, speech_ratio = detector.detect_voice_activity(audio, sr)

    assert len(segments) >= 2
    assert speech_sec > 1.0
    assert silence_sec > 0.5
    assert 0.3 <= speech_ratio <= 0.85
    assert any(seg.is_speech for seg in segments)
    assert any(not seg.is_speech for seg in segments)


def test_vad_silence(silence_wav):
    """Verify VAD detects 0% speech on pure silence audio."""
    audio, sr = sf.read(str(silence_wav), dtype="float32")
    detector = VoiceActivityDetector()

    segments, speech_sec, silence_sec, speech_ratio = detector.detect_voice_activity(audio, sr)
    assert speech_ratio == 0.0
    assert speech_sec == 0.0
    assert silence_sec == pytest.approx(2.0, abs=0.1)


def test_vad_rejects_music_and_instruments():
    """Verify VAD rejects musical beats, polyphonic instrument chords, drums, and environmental hum."""
    detector = VoiceActivityDetector()
    sr = 16000
    dur = 4.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)

    # 1. Pop / EDM music with 55Hz kick drum and 65Hz bassline
    kick = 0.8 * np.sin(2 * np.pi * 55 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
    bass = 0.4 * np.sin(2 * np.pi * 65.4 * t)
    synth = 0.3 * np.sin(2 * np.pi * 329.6 * t)
    edm_music = kick + bass + synth
    _, sp_edm, _, ratio_edm = detector.detect_voice_activity(edm_music, sr)
    assert ratio_edm == 0.0
    assert sp_edm == 0.0

    # 2. Piano / Guitar polyphonic chords with overtones (C-major: C4=261.6Hz, E4=329.6Hz, G4=392.0Hz)
    chords_harm = np.zeros_like(t)
    for f in [261.63, 329.63, 392.00]:
        for k in range(1, 5):
            chords_harm += (0.2 / k) * np.sin(2 * np.pi * k * f * t)
    chords_harm = chords_harm * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
    chords_harm = (chords_harm / np.max(np.abs(chords_harm))) * 0.7
    _, sp_chords, _, ratio_chords = detector.detect_voice_activity(chords_harm, sr)
    assert ratio_chords == 0.0
    assert sp_chords == 0.0

    # 3. Drum kit solo (sub-bass kick + noise snare)
    drums = kick + 0.4 * np.random.normal(0, 1, len(t)) * (np.sin(2 * np.pi * 2 * t) > 0.7)
    _, sp_drums, _, ratio_drums = detector.detect_voice_activity(drums, sr)
    assert ratio_drums == 0.0
    assert sp_drums == 0.0

    # 4. Environmental hum / rumble (fan, motor, air conditioner)
    hum = np.cumsum(np.random.normal(0, 0.05, len(t)))
    hum = (hum / np.max(np.abs(hum))) * 0.5
    _, sp_hum, _, ratio_hum = detector.detect_voice_activity(hum, sr)
    assert ratio_hum == 0.0
    assert sp_hum == 0.0

    # 5. Pop song with singing melody (C4, E4, G4, A4) and acoustic guitar backing
    singing = np.zeros_like(t)
    for i, f0 in enumerate([261.63, 329.63, 392.00, 440.00]):
        st = int(i * 1.0 * sr)
        en = int((i + 1) * 1.0 * sr)
        t_seg = t[st:en] - t[st]
        singing[st:en] = 0.5 * np.sin(2 * np.pi * f0 * t_seg) + 0.2 * np.sin(4 * np.pi * f0 * t_seg)
    guitar = 0.2 * np.sin(2 * np.pi * 196.0 * t) + 0.2 * np.sin(2 * np.pi * 246.9 * t)
    song = (singing + guitar) * 0.7
    _, sp_song, _, ratio_song = detector.detect_voice_activity(song, sr)
    assert ratio_song == 0.0
    assert sp_song == 0.0


def test_vad_detects_speech_with_background_music(speech_wav):
    """Verify VAD ignores background music and detects spoken human voice."""
    audio, sr = sf.read(str(speech_wav), dtype="float32")
    detector = VoiceActivityDetector()
    dur = len(audio) / sr
    t = np.linspace(0, dur, len(audio), endpoint=False)
    # Add background music (55Hz bass beat + 330Hz synth)
    bg_music = 0.08 * (np.sin(2 * np.pi * 55 * t) + np.sin(2 * np.pi * 329.6 * t))
    mixed = audio + bg_music

    segments, sp_sec, sil_sec, sp_ratio = detector.detect_voice_activity(mixed, sr)
    assert sp_sec > 0.5
    assert sp_ratio >= 0.20
    assert any(seg.is_speech for seg in segments)


# ============================================================================
# 6. Audio Window Chunker & Streaming Buffer Tests
# ============================================================================

def test_sliding_window_chunker(speech_wav):
    """Verify fixed-length sliding analysis window creation for ML scoring."""
    cfg = IngestionConfig(target_sample_rate=16000, chunk_window_sec=2.0, chunk_hop_sec=1.0)
    chunker = AudioChunker(config=cfg)

    # 4-second audio resampled to 16kHz -> 64000 samples
    audio, _ = sf.read(str(speech_wav), dtype="float32")
    preprocessor = AudioPreprocessor(config=cfg)
    processed = preprocessor.preprocess(audio, 48000)

    chunks = chunker.create_chunks(processed, sample_rate=16000)

    # For 4.0s with window=2.0s and hop=1.0s:
    # Windows: [0-2s], [1-3s], [2-4s] -> 3 chunks
    assert len(chunks) == 3
    assert chunks[0].start_time_sec == 0.0
    assert chunks[0].end_time_sec == 2.0
    assert chunks[0].num_samples == 32000  # 2.0s * 16000
    assert chunks[1].start_time_sec == 1.0
    assert chunks[1].end_time_sec == 3.0


def test_streaming_audio_buffer():
    """Verify ring buffer functionality for live real-time audio streams."""
    cfg = IngestionConfig(target_sample_rate=16000, chunk_window_sec=1.0, chunk_hop_sec=0.5)
    buffer = StreamingAudioBuffer(config=cfg)

    # Feed 100ms packets (1600 samples each) sequentially
    packet = np.random.normal(0, 0.1, 1600).astype(np.float32)
    emitted_chunks = []

    # Feed 15 packets (1.5 seconds total)
    for _ in range(15):
        new_chunks = buffer.add_samples(packet)
        emitted_chunks.extend(new_chunks)

    # In 1.5s with window=1.0s and hop=0.5s:
    # Chunk 0 @ 1.0s, Chunk 1 @ 1.5s -> 2 chunks emitted
    assert len(emitted_chunks) == 2
    assert emitted_chunks[0].num_samples == 16000
    assert emitted_chunks[0].start_time_sec == 0.0
    assert emitted_chunks[1].start_time_sec == 0.5


# ============================================================================
# 7. End-to-End Pipeline & Serialization Tests
# ============================================================================

def test_pipeline_end_to_end_file(speech_wav, temp_audio_dir):
    """Verify full end-to-end ingestion pipeline on a file."""
    cfg = IngestionConfig(
        target_sample_rate=16000,
        forensic_storage_dir=temp_audio_dir / "forensics",
        vad_enabled=True,
        chunk_window_sec=2.0,
        chunk_hop_sec=1.0
    )
    pipeline = AudioIngestionPipeline(config=cfg)
    result = pipeline.process_file(speech_wav)

    assert result.audio_id is not None
    assert result.raw_sample_rate == 48000
    assert result.target_sample_rate == 16000
    assert len(result.processed_audio) == 64000  # 4.0s * 16000
    assert result.processed_audio.ndim == 1
    assert result.speech_duration_sec > 0.0
    assert len(result.chunks) > 0
    assert result.forensic.verified_integrity is True

    # Test summary string
    summary = result.summary()
    assert "=== Audio Ingestion Summary" in summary
    assert "SHA-256" in summary

    # Test dictionary serialization
    d = result.to_dict()
    assert d["audio_id"] == result.audio_id
    assert d["target_sample_rate"] == 16000
    assert len(d["chunks_meta"]) == len(result.chunks)

    # Test exporting preprocessed WAV
    out_wav = temp_audio_dir / "exported_16k.wav"
    exported_path = result.save_processed_wav(out_wav)
    assert exported_path.exists()
    exp_audio, exp_sr = sf.read(str(exported_path))
    assert exp_sr == 16000
    assert len(exp_audio) == 64000


def test_pipeline_end_to_end_bytes(speech_wav):
    """Verify full ingestion pipeline directly from in-memory bytes."""
    with open(speech_wav, "rb") as f:
        raw_bytes = f.read()

    pipeline = AudioIngestionPipeline()
    result = pipeline.process_bytes(raw_bytes, original_filename="stream_test.wav")

    assert result.audio_id is not None
    assert result.forensic.original_size_bytes == len(raw_bytes)
    assert result.target_sample_rate == 16000
    assert len(result.processed_audio) > 0


# ============================================================================
# 8. Error Handling & Edge Cases
# ============================================================================

def test_error_nonexistent_file():
    """Verify FileNotFoundError is raised when file does not exist."""
    pipeline = AudioIngestionPipeline()
    with pytest.raises(FileNotFoundError):
        pipeline.process_file("non_existent_file_path.wav")


def test_error_corrupt_file(corrupt_file):
    """Verify AudioDecodeError/CorruptAudioError is raised for corrupt files."""
    pipeline = AudioIngestionPipeline()
    with pytest.raises(AudioDecodeError):
        pipeline.process_file(corrupt_file)
