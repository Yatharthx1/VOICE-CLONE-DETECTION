"""
Configuration models for the Audio Ingestion & Preprocessing pipeline.
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class IngestionConfig(BaseModel):
    """
    Configuration options for audio decoding, forensic preservation,
    preprocessing, voice activity detection (VAD), and analysis chunking.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # --- Preprocessing Targets ---
    target_sample_rate: int = Field(
        default=16000,
        description="Target sample rate in Hz for downstream analysis (typically 16000 or 24000)."
    )
    to_mono: bool = Field(
        default=True,
        description="Convert stereo / multi-channel audio to single-channel mono."
    )
    normalize_audio: bool = Field(
        default=True,
        description="Normalize audio levels to prevent clipping and balance volume."
    )
    target_dbfs: float = Field(
        default=-20.0,
        description="Target RMS loudness level in dBFS when normalization is enabled."
    )
    peak_normalize_ceiling: float = Field(
        default=0.95,
        description="Ceiling for peak normalization (amplitude between 0.0 and 1.0)."
    )
    remove_dc_offset: bool = Field(
        default=True,
        description="Remove DC bias/offset using mean subtraction and highpass filtering."
    )

    # --- Forensic Preservation ---
    preserve_forensic_copy: bool = Field(
        default=True,
        description="Preserve an exact byte-for-byte copy and cryptographic hashes for chain of custody."
    )
    forensic_storage_dir: Optional[Path] = Field(
        default=None,
        description="Directory to store pristine forensic raw audio copies. If None, uses system temp/forensic directory."
    )

    # --- Voice Activity Detection (VAD) ---
    vad_enabled: bool = Field(
        default=True,
        description="Enable Voice Activity Detection to segment speech vs silence/noise."
    )
    vad_energy_threshold_db: float = Field(
        default=-40.0,
        description="Energy threshold in dBFS to classify frame as speech."
    )
    vad_frame_duration_ms: int = Field(
        default=30,
        description="Duration of each VAD analysis frame in milliseconds (e.g. 10, 20, 30 ms)."
    )
    vad_min_speech_duration_ms: int = Field(
        default=100,
        description="Minimum duration in ms to treat a speech burst as valid speech."
    )
    vad_min_silence_duration_ms: int = Field(
        default=250,
        description="Minimum duration in ms of non-speech to register a silence gap (hangover duration)."
    )

    # --- Analysis Windowing & Chunking ---
    chunk_window_sec: float = Field(
        default=3.0,
        description="Duration in seconds of analysis windows for ML deepfake scoring."
    )
    chunk_hop_sec: float = Field(
        default=1.0,
        description="Step/hop size in seconds between consecutive sliding analysis windows."
    )
    min_chunk_duration_sec: float = Field(
        default=0.5,
        description="Minimum audio duration required to form an analysis chunk."
    )
    pad_short_chunks: bool = Field(
        default=True,
        description="Zero-pad audio shorter than chunk_window_sec to reach full window size."
    )

    # --- Real-Time Streaming ---
    stream_chunk_duration_ms: int = Field(
        default=100,
        description="Buffer chunk duration in milliseconds for live streaming audio ingestion."
    )
