"""
Robust Multi-Format Audio Decoder.
Decodes WAV, MP3, M4A, MP4, FLAC, OGG, AAC, WebM, and other audio/video containers
into standardized float32 numpy arrays.
"""

import io
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import soundfile as sf


class AudioDecodeError(Exception):
    """Raised when audio decoding fails."""
    pass


class CorruptAudioError(AudioDecodeError):
    """Raised when the audio bitstream or file header is corrupt."""
    pass


class UnsupportedAudioFormatError(AudioDecodeError):
    """Raised when the audio format/codec cannot be decoded."""
    pass


class AudioDecoder:
    """
    Decodes audio files and raw bitstreams from various formats and containers.
    """

    def __init__(self):
        self._has_ffmpeg = shutil.which("ffmpeg") is not None

    def decode_file(
        self,
        file_path: Union[str, Path]
    ) -> Tuple[np.ndarray, int, int]:
        """
        Decode an audio or video file into a float32 numpy array.
        
        Returns:
            Tuple of (audio_array, sample_rate, channels)
            where audio_array is float32 with values in range [-1.0, 1.0].
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")

        if path.stat().st_size == 0:
            raise CorruptAudioError(f"Audio file is empty (0 bytes): {file_path}")

        # Attempt 1: Fast direct read using soundfile for native formats (WAV, FLAC, OGG)
        try:
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
            channels = audio.shape[1] if audio.ndim > 1 else 1
            return audio, sr, channels
        except Exception:
            pass  # Fallback to ffmpeg stream decode

        # Attempt 2: High-compatibility FFmpeg decode for MP3, M4A, MP4, AAC, etc.
        if self._has_ffmpeg:
            try:
                return self._decode_with_ffmpeg(path)
            except Exception as e:
                raise AudioDecodeError(f"Failed to decode audio file {file_path} with FFmpeg: {e}") from e

        # Attempt 3: Torchaudio fallback if installed
        try:
            import torchaudio
            tensor, sr = torchaudio.load(str(path))
            audio = tensor.numpy()
            if audio.ndim == 2:
                # Shape (channels, samples) -> (samples, channels)
                audio = audio.T
                channels = audio.shape[1]
                if channels == 1:
                    audio = audio.squeeze(-1)
            else:
                channels = 1
            return audio.astype(np.float32), sr, channels
        except Exception as e:
            raise AudioDecodeError(
                f"Could not decode {file_path}. FFmpeg is required for compressed formats (MP3/M4A/MP4): {e}"
            )

    def decode_bytes(
        self,
        audio_bytes: bytes,
        format_hint: Optional[str] = None
    ) -> Tuple[np.ndarray, int, int]:
        """
        Decode raw audio bytes in-memory without requiring a physical disk file.
        """
        if not audio_bytes or len(audio_bytes) == 0:
            raise CorruptAudioError("Audio byte stream is empty.")

        # Attempt 1: Soundfile in-memory buffer
        try:
            byte_io = io.BytesIO(audio_bytes)
            audio, sr = sf.read(byte_io, dtype="float32", always_2d=False)
            channels = audio.shape[1] if audio.ndim > 1 else 1
            return audio, sr, channels
        except Exception:
            pass

        # Attempt 2: FFmpeg stdin pipe decode
        if self._has_ffmpeg:
            try:
                return self._decode_bytes_with_ffmpeg(audio_bytes, format_hint)
            except Exception as e:
                raise AudioDecodeError(f"Failed to decode audio bytes with FFmpeg: {e}") from e

        raise UnsupportedAudioFormatError(
            "Cannot decode audio byte buffer. Ensure format is standard WAV/FLAC/OGG or FFmpeg is available."
        )

    def _decode_with_ffmpeg(self, file_path: Path) -> Tuple[np.ndarray, int, int]:
        """
        Decode any media file (including MP4 video audio track, M4A, MP3, etc.)
        by streaming raw 32-bit float PCM via stdout pipe.
        """
        # First probe stream sample rate and channel count
        probe_cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "a:0",
            str(file_path)
        ]
        probe_res = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        import json
        info = json.loads(probe_res.stdout)
        streams = info.get("streams", [])
        if not streams:
            raise CorruptAudioError(f"No audio stream found in {file_path}")

        sample_rate = int(streams[0].get("sample_rate", 16000))
        channels = int(streams[0].get("channels", 1))

        # Run ffmpeg to output raw float32 LE PCM to stdout
        cmd = [
            "ffmpeg", "-v", "error",
            "-i", str(file_path),
            "-vn",                      # Disable video
            "-f", "f32le",              # Output 32-bit float Little-Endian PCM
            "-acodec", "pcm_f32le",
            "-"
        ]
        proc = subprocess.run(cmd, capture_output=True, check=True)
        raw_pcm = proc.stdout

        audio = np.frombuffer(raw_pcm, dtype=np.float32)
        if channels > 1:
            audio = audio.reshape(-1, channels)

        return audio, sample_rate, channels

    def _decode_bytes_with_ffmpeg(
        self,
        audio_bytes: bytes,
        format_hint: Optional[str] = None
    ) -> Tuple[np.ndarray, int, int]:
        """
        Decode raw in-memory bytes using FFmpeg stdin pipe.
        """
        # We output standard 16kHz or native float32 pcm
        cmd = [
            "ffmpeg", "-v", "error",
            "-i", "pipe:0",
            "-vn",
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-"
        ]
        proc = subprocess.run(cmd, input=audio_bytes, capture_output=True, check=True)
        raw_pcm = proc.stdout

        # Also probe sample rate/channels via ffprobe stdin
        probe_cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-select_streams", "a:0",
            "pipe:0"
        ]
        probe_proc = subprocess.run(probe_cmd, input=audio_bytes, capture_output=True, text=True)
        import json
        try:
            info = json.loads(probe_proc.stdout)
            streams = info.get("streams", [])
            sample_rate = int(streams[0].get("sample_rate", 16000)) if streams else 16000
            channels = int(streams[0].get("channels", 1)) if streams else 1
        except Exception:
            sample_rate = 16000
            channels = 1

        audio = np.frombuffer(raw_pcm, dtype=np.float32)
        if channels > 1 and len(audio) % channels == 0:
            audio = audio.reshape(-1, channels)

        return audio, sample_rate, channels
