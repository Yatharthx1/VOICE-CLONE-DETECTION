"""
Pytest configuration and synthetic audio fixtures for testing the Audio Ingestion pipeline.
"""

import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf


def generate_speech_like_signal(duration_sec: float = 4.0, sample_rate: int = 48000) -> np.ndarray:
    """
    Generate synthetic harmonic speech-like audio with formants (F0=130Hz, F1=600Hz, F2=1700Hz)
    and alternating speech/silence intervals to test VAD and acoustic analysis.
    """
    total_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)

    # Base glottal pitch pulse
    f0 = 130.0 + 10.0 * np.sin(2 * np.pi * 1.5 * t)  # natural pitch modulation
    phase = np.cumsum(2 * np.pi * f0 / sample_rate)

    # Harmonic series (vocal tract formants)
    signal = (
        0.5 * np.sin(phase) +
        0.3 * np.sin(2 * phase) +
        0.2 * np.sin(3 * phase) +
        0.15 * np.sin(5 * phase) +
        0.1 * np.sin(7 * phase)
    )

    # Envelope modulation: 0-1.2s speech, 1.2-1.8s silence, 1.8-3.2s speech, 3.2-4.0s silence
    envelope = np.zeros_like(t)
    envelope[(t >= 0.0) & (t < 1.2)] = 0.8
    envelope[(t >= 1.8) & (t < 3.2)] = 0.8

    # Smooth the transitions
    signal = signal * envelope

    # Add gentle pink/white noise floor (-45 dB)
    noise = np.random.normal(0, 0.005, total_samples)
    signal = signal + noise

    # Normalize to [-0.8, 0.8]
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = (signal / max_val) * 0.8

    return signal.astype(np.float32)


@pytest.fixture
def temp_audio_dir():
    """Create a temporary directory for test audio files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def clean_wav_stereo(temp_audio_dir) -> Path:
    """Creates a 44.1 kHz stereo 16-bit WAV file."""
    sr = 44100
    dur = 3.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    ch1 = 0.5 * np.sin(2 * np.pi * 440 * t)
    ch2 = 0.5 * np.sin(2 * np.pi * 880 * t)
    stereo = np.column_stack([ch1, ch2]).astype(np.float32)

    path = temp_audio_dir / "clean_stereo.wav"
    sf.write(str(path), stereo, sr, format="WAV", subtype="PCM_16")
    return path


@pytest.fixture
def speech_wav(temp_audio_dir) -> Path:
    """Creates a 48 kHz speech-like WAV file with speech/silence segments."""
    sr = 48000
    audio = generate_speech_like_signal(duration_sec=4.0, sample_rate=sr)
    path = temp_audio_dir / "speech_test.wav"
    sf.write(str(path), audio, sr, format="WAV", subtype="PCM_16")
    return path


@pytest.fixture
def compressed_mp3(temp_audio_dir, speech_wav) -> Path:
    """Creates a compressed MP3 audio file using FFmpeg if available."""
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg executable not found in system PATH")
    mp3_path = temp_audio_dir / "speech_test.mp3"
    cmd = [
        "ffmpeg", "-v", "quiet", "-y",
        "-i", str(speech_wav),
        "-codec:a", "libmp3lame",
        "-b:a", "128k",
        str(mp3_path)
    ]
    subprocess.run(cmd, check=True)
    return mp3_path


@pytest.fixture
def compressed_m4a(temp_audio_dir, speech_wav) -> Path:
    """Creates an AAC M4A audio file using FFmpeg if available."""
    import shutil
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg executable not found in system PATH")
    m4a_path = temp_audio_dir / "speech_test.m4a"
    cmd = [
        "ffmpeg", "-v", "quiet", "-y",
        "-i", str(speech_wav),
        "-codec:a", "aac",
        "-b:a", "128k",
        str(m4a_path)
    ]
    subprocess.run(cmd, check=True)
    return m4a_path


@pytest.fixture
def compressed_flac(temp_audio_dir, speech_wav) -> Path:
    """Creates a lossless FLAC audio file."""
    flac_path = temp_audio_dir / "speech_test.flac"
    audio, sr = sf.read(str(speech_wav))
    sf.write(str(flac_path), audio, sr, format="FLAC")
    return flac_path


@pytest.fixture
def clipped_wav(temp_audio_dir) -> Path:
    """Creates an audio file with heavy clipping."""
    sr = 16000
    dur = 2.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # Exaggerated sine wave that gets clipped hard at 1.0
    sig = 2.5 * np.sin(2 * np.pi * 300 * t)
    sig = np.clip(sig, -1.0, 1.0).astype(np.float32)

    path = temp_audio_dir / "clipped.wav"
    sf.write(str(path), sig, sr, format="WAV", subtype="FLOAT")
    return path


@pytest.fixture
def silence_wav(temp_audio_dir) -> Path:
    """Creates a pure silence WAV file."""
    sr = 16000
    dur = 2.0
    sig = np.zeros(int(sr * dur), dtype=np.float32)
    path = temp_audio_dir / "silence.wav"
    sf.write(str(path), sig, sr, format="WAV", subtype="PCM_16")
    return path


@pytest.fixture
def corrupt_file(temp_audio_dir) -> Path:
    """Creates a corrupt non-audio file."""
    path = temp_audio_dir / "corrupt_sample.wav"
    with open(path, "wb") as f:
        f.write(b"NOT_A_VALID_RIFF_HEADER_CORRUPT_BYTES_XYZ123")
    return path
