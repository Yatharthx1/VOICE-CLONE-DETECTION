"""
Data models for audio ingestion, metadata, forensic records, VAD segments, and analysis chunks.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class AudioMetadata(BaseModel):
    """
    Extracted container and acoustic stream metadata.
    """
    file_path: Optional[str] = Field(None, description="Original file path if loaded from disk.")
    file_size_bytes: int = Field(0, description="Size of the raw audio file in bytes.")
    container_format: str = Field("unknown", description="Audio/video container format (wav, mp3, mp4, etc.).")
    codec: str = Field("unknown", description="Audio codec name (pcm_s16le, aac, mp3, flac, opus, etc.).")
    sample_rate: int = Field(..., description="Original sampling rate in Hz.")
    channels: int = Field(..., description="Original number of audio channels (1=mono, 2=stereo).")
    bit_depth: Optional[int] = Field(None, description="Bit depth per sample (16, 24, 32, etc.) if applicable.")
    bitrate_kbps: Optional[float] = Field(None, description="Bitrate in kilobits per second.")
    duration_seconds: float = Field(..., description="Calculated duration in seconds.")
    total_samples: int = Field(..., description="Total number of audio samples.")
    peak_dbfs: float = Field(0.0, description="Peak signal amplitude in dBFS.")
    rms_dbfs: float = Field(-100.0, description="Root Mean Square (RMS) loudness in dBFS.")
    is_clipped: bool = Field(False, description="True if any samples exceed or reach clipping thresholds.")
    dynamic_range_db: float = Field(0.0, description="Estimated dynamic range in dB.")
    snr_estimate_db: Optional[float] = Field(None, description="Estimated Signal-to-Noise Ratio (SNR) in dB.")

    def summary(self) -> str:
        return (
            f"Format: {self.container_format.upper()} ({self.codec}) | "
            f"SR: {self.sample_rate} Hz | Channels: {self.channels} | "
            f"Duration: {self.duration_seconds:.2f}s | RMS: {self.rms_dbfs:.1f} dBFS | "
            f"Clipped: {'YES' if self.is_clipped else 'NO'}"
        )


class ForensicRecord(BaseModel):
    """
    Forensic preservation and integrity record ensuring chain of custody.
    """
    audio_id: str = Field(..., description="Unique UUID for this ingested audio session.")
    sha256_hash: str = Field(..., description="SHA-256 cryptographic checksum of the raw input bytes.")
    md5_hash: str = Field(..., description="MD5 checksum for secondary integrity verification.")
    ingestion_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of ingestion."
    )
    original_filename: str = Field("memory_stream", description="Original filename or stream identifier.")
    original_size_bytes: int = Field(0, description="Original file size in bytes.")
    forensic_copy_path: Optional[str] = Field(None, description="Path to archived unmodified raw audio copy.")
    verified_integrity: bool = Field(True, description="Integrity check status against original hash.")


class VADSegment(BaseModel):
    """
    Voice Activity Detection segment boundary.
    """
    segment_id: int = Field(..., description="Sequential segment index.")
    is_speech: bool = Field(..., description="True if segment contains active speech, False if silence/noise.")
    start_seconds: float = Field(..., description="Start timestamp in seconds.")
    end_seconds: float = Field(..., description="End timestamp in seconds.")
    duration_seconds: float = Field(..., description="Duration of the segment in seconds.")
    start_sample: int = Field(..., description="Start sample index at target sample rate.")
    end_sample: int = Field(..., description="End sample index at target sample rate.")
    mean_energy_db: float = Field(..., description="Average energy in dBFS during this segment.")


class AudioChunk(BaseModel):
    """
    Windowed audio chunk prepared for downstream ML deepfake / acoustic analysis models.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_index: int = Field(..., description="0-indexed chunk sequence number.")
    start_time_sec: float = Field(..., description="Chunk start time in seconds relative to recording start.")
    end_time_sec: float = Field(..., description="Chunk end time in seconds.")
    sample_rate: int = Field(..., description="Sample rate of the audio chunk array.")
    samples: Any = Field(..., description="1D numpy array of audio samples (float32).")
    contains_speech: bool = Field(True, description="True if chunk overlaps with active speech.")
    speech_ratio: float = Field(1.0, description="Fraction of chunk duration containing active speech (0.0 - 1.0).")
    is_padded: bool = Field(False, description="True if chunk was zero-padded to reach target window length.")

    @property
    def num_samples(self) -> int:
        return len(self.samples) if isinstance(self.samples, np.ndarray) else 0

    @property
    def duration_sec(self) -> float:
        return self.end_time_sec - self.start_time_sec


class IngestedAudio(BaseModel):
    """
    Complete container returned by the Audio Ingestion Pipeline, ready for downstream
    acoustic, spectral, prosody, synthesis artifact, and phase analysis.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_id: str = Field(..., description="Unique UUID for this audio session.")
    metadata: AudioMetadata = Field(..., description="Container & acoustic metadata.")
    forensic: ForensicRecord = Field(..., description="Forensic integrity record & hashes.")
    
    # Audio data
    raw_audio: Any = Field(..., description="Original decoded numpy array (float32, multi-channel or mono).")
    raw_sample_rate: int = Field(..., description="Original sampling rate in Hz.")
    raw_channels: int = Field(..., description="Original channel count.")
    
    # Preprocessed audio
    processed_audio: Any = Field(..., description="Normalized, mono, resampled 1D numpy array (float32).")
    target_sample_rate: int = Field(..., description="Preprocessed sampling rate in Hz.")
    
    # Segmentation & Windows
    vad_segments: List[VADSegment] = Field(default_factory=list, description="Voice Activity Detection segments.")
    speech_duration_sec: float = Field(0.0, description="Total active speech duration in seconds.")
    silence_duration_sec: float = Field(0.0, description="Total silence / non-speech duration in seconds.")
    speech_ratio: float = Field(0.0, description="Active speech fraction (speech_duration / total_duration).")
    chunks: List[AudioChunk] = Field(default_factory=list, description="Sliding analysis windows for ML scoring.")

    def summary(self) -> str:
        lines = [
            f"=== Audio Ingestion Summary [ID: {self.audio_id[:8]}] ===",
            f"• Source: {self.metadata.file_path or self.forensic.original_filename} ({self.metadata.file_size_bytes / 1024:.1f} KB)",
            f"• Original Audio: {self.metadata.container_format.upper()} | {self.raw_sample_rate} Hz | {self.raw_channels} ch | {self.metadata.duration_seconds:.2f}s",
            f"• Preprocessed Audio: Mono | {self.target_sample_rate} Hz | {len(self.processed_audio)} samples ({len(self.processed_audio)/self.target_sample_rate:.2f}s)",
            f"• Acoustics: Peak {self.metadata.peak_dbfs:.1f} dBFS | RMS {self.metadata.rms_dbfs:.1f} dBFS | Clipped: {self.metadata.is_clipped}",
            f"• VAD Speech: {self.speech_duration_sec:.2f}s ({self.speech_ratio*100:.1f}%) | Silence: {self.silence_duration_sec:.2f}s ({len(self.vad_segments)} segments)",
            f"• Analysis Windows: {len(self.chunks)} chunks ready for deepfake detection",
            f"• Forensic Hash: SHA-256: {self.forensic.sha256_hash[:16]}... (Integrity Verified: {self.forensic.verified_integrity})"
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata and analysis results to a JSON-serializable dictionary."""
        return {
            "audio_id": self.audio_id,
            "metadata": self.metadata.model_dump(),
            "forensic": self.forensic.model_dump(),
            "target_sample_rate": self.target_sample_rate,
            "processed_samples_count": len(self.processed_audio) if isinstance(self.processed_audio, np.ndarray) else 0,
            "speech_duration_sec": self.speech_duration_sec,
            "silence_duration_sec": self.silence_duration_sec,
            "speech_ratio": self.speech_ratio,
            "vad_segments": [seg.model_dump() for seg in self.vad_segments],
            "chunk_count": len(self.chunks),
            "chunks_meta": [
                {
                    "chunk_index": c.chunk_index,
                    "start_time_sec": c.start_time_sec,
                    "end_time_sec": c.end_time_sec,
                    "contains_speech": c.contains_speech,
                    "speech_ratio": c.speech_ratio,
                    "num_samples": c.num_samples,
                }
                for c in self.chunks
            ]
        }

    def save_processed_wav(self, output_path: str | Path) -> Path:
        """Export the preprocessed 16kHz mono audio as a standard WAV file."""
        import soundfile as sf
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), self.processed_audio, self.target_sample_rate, format='WAV', subtype='PCM_16')
        return path
