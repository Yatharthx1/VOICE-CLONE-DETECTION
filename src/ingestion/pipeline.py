"""
Unified Audio Ingestion Pipeline.
End-to-end orchestration of decoding, forensic preservation, metadata extraction,
preprocessing, Voice Activity Detection, and analysis window chunking.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np

from .config import IngestionConfig
from .forensic import ForensicPreserver
from .metadata_extractor import MetadataExtractor
from .decoder import AudioDecoder
from .preprocessor import AudioPreprocessor
from .vad import VoiceActivityDetector
from .chunker import AudioChunker, StreamingAudioBuffer
from .models import IngestedAudio, AudioChunk


class AudioIngestionPipeline:
    """
    Main entrypoint for the Voice Integrity Verification Audio Ingestion subsystem.
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()
        self.preserver = ForensicPreserver(storage_dir=self.config.forensic_storage_dir)
        self.extractor = MetadataExtractor()
        self.decoder = AudioDecoder()
        self.preprocessor = AudioPreprocessor(config=self.config)
        self.vad = VoiceActivityDetector(config=self.config)
        self.chunker = AudioChunker(config=self.config)

    def process_file(
        self,
        file_path: Union[str, Path],
        audio_id: Optional[str] = None
    ) -> IngestedAudio:
        """
        Ingest and preprocess an audio or video file from disk.

        Steps:
        1. Forensic preservation (calculate SHA-256 / MD5, archive unmodified file)
        2. Decode audio bitstream into raw float32 array
        3. Extract comprehensive metadata (format, codec, channels, SR, peak, RMS, clipping)
        4. Preprocess (convert to mono, resample to 16kHz/24kHz, DC filter, normalize)
        5. Perform Voice Activity Detection (VAD) & speech/silence segmentation
        6. Generate sliding analysis windows / chunks for deepfake detection models
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")

        # 1. Forensic preservation
        forensic = self.preserver.preserve_file(path, audio_id=audio_id)
        current_audio_id = forensic.audio_id

        # 2. Decode audio
        raw_audio, raw_sr, raw_channels = self.decoder.decode_file(path)

        # 3. Extract metadata
        metadata = self.extractor.extract_from_file(
            path,
            decoded_audio=raw_audio,
            sample_rate=raw_sr
        )

        # 4. Preprocess audio (mono, target SR, DC remove, normalize)
        processed_audio = self.preprocessor.preprocess(raw_audio, raw_sr)

        # 5. Voice Activity Detection
        vad_segments = []
        speech_dur = 0.0
        silence_dur = 0.0
        speech_ratio = 1.0

        if self.config.vad_enabled:
            vad_segments, speech_dur, silence_dur, speech_ratio = self.vad.detect_voice_activity(
                processed_audio,
                self.config.target_sample_rate
            )
        else:
            total_dur = len(processed_audio) / self.config.target_sample_rate if self.config.target_sample_rate > 0 else 0.0
            speech_dur = total_dur
            speech_ratio = 1.0

        # 6. Create sliding analysis chunks
        chunks = self.chunker.create_chunks(
            processed_audio,
            self.config.target_sample_rate,
            vad_segments=vad_segments
        )

        return IngestedAudio(
            audio_id=current_audio_id,
            metadata=metadata,
            forensic=forensic,
            raw_audio=raw_audio,
            raw_sample_rate=raw_sr,
            raw_channels=raw_channels,
            processed_audio=processed_audio,
            target_sample_rate=self.config.target_sample_rate,
            vad_segments=vad_segments,
            speech_duration_sec=speech_dur,
            silence_duration_sec=silence_dur,
            speech_ratio=speech_ratio,
            chunks=chunks
        )

    def process_bytes(
        self,
        audio_bytes: bytes,
        original_filename: str = "stream_sample.wav",
        audio_id: Optional[str] = None
    ) -> IngestedAudio:
        """
        Ingest and preprocess in-memory audio bytes (e.g. from an HTTP upload or network stream).
        """
        # 1. Forensic preservation
        forensic = self.preserver.preserve_bytes(
            audio_bytes,
            original_filename=original_filename,
            audio_id=audio_id
        )
        current_audio_id = forensic.audio_id

        # 2. Decode bytes
        raw_audio, raw_sr, raw_channels = self.decoder.decode_bytes(audio_bytes)

        # 3. Extract metadata
        metadata = self.extractor.extract_from_array(
            raw_audio,
            sample_rate=raw_sr,
            container_format=Path(original_filename).suffix.lstrip(".") or "raw_bytes"
        )

        # 4. Preprocess audio
        processed_audio = self.preprocessor.preprocess(raw_audio, raw_sr)

        # 5. Voice Activity Detection
        vad_segments = []
        speech_dur = 0.0
        silence_dur = 0.0
        speech_ratio = 1.0

        if self.config.vad_enabled:
            vad_segments, speech_dur, silence_dur, speech_ratio = self.vad.detect_voice_activity(
                processed_audio,
                self.config.target_sample_rate
            )
        else:
            total_dur = len(processed_audio) / self.config.target_sample_rate if self.config.target_sample_rate > 0 else 0.0
            speech_dur = total_dur
            speech_ratio = 1.0

        # 6. Create sliding analysis chunks
        chunks = self.chunker.create_chunks(
            processed_audio,
            self.config.target_sample_rate,
            vad_segments=vad_segments
        )

        return IngestedAudio(
            audio_id=current_audio_id,
            metadata=metadata,
            forensic=forensic,
            raw_audio=raw_audio,
            raw_sample_rate=raw_sr,
            raw_channels=raw_channels,
            processed_audio=processed_audio,
            target_sample_rate=self.config.target_sample_rate,
            vad_segments=vad_segments,
            speech_duration_sec=speech_dur,
            silence_duration_sec=silence_dur,
            speech_ratio=speech_ratio,
            chunks=chunks
        )

    def create_streaming_buffer(
        self,
        window_duration_sec: Optional[float] = None,
        hop_duration_sec: Optional[float] = None,
        min_speech_duration_sec: Optional[float] = None
    ) -> StreamingAudioBuffer:
        """Create a real-time ring buffer for live streaming audio ingestion."""
        return StreamingAudioBuffer(
            config=self.config,
            window_duration_sec=window_duration_sec,
            hop_duration_sec=hop_duration_sec,
            min_speech_duration_sec=min_speech_duration_sec
        )
