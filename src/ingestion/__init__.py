"""
Audio Ingestion Subsystem for Real-Time Voice Cloning Detection Framework.
"""

from .config import IngestionConfig
from .models import AudioMetadata, ForensicRecord, VADSegment, AudioChunk, IngestedAudio
from .forensic import ForensicPreserver
from .metadata_extractor import MetadataExtractor
from .decoder import AudioDecoder, AudioDecodeError, CorruptAudioError, UnsupportedAudioFormatError
from .preprocessor import AudioPreprocessor
from .vad import VoiceActivityDetector
from .chunker import AudioChunker, StreamingAudioBuffer
from .pipeline import AudioIngestionPipeline

__all__ = [
    "IngestionConfig",
    "AudioMetadata",
    "ForensicRecord",
    "VADSegment",
    "AudioChunk",
    "IngestedAudio",
    "ForensicPreserver",
    "MetadataExtractor",
    "AudioDecoder",
    "AudioDecodeError",
    "CorruptAudioError",
    "UnsupportedAudioFormatError",
    "AudioPreprocessor",
    "VoiceActivityDetector",
    "AudioChunker",
    "StreamingAudioBuffer",
    "AudioIngestionPipeline",
]
