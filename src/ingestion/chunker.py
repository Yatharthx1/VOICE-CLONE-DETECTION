"""
Audio Analysis Chunker & Streaming Window Buffer.
Generates sliding analysis windows and streaming buffers for real-time
deepfake and voice cloning detection.
"""

from typing import Generator, List, Optional
import numpy as np

from .config import IngestionConfig
from .models import AudioChunk, VADSegment
from .vad import VoiceActivityDetector


class AudioChunker:
    """
    Splits continuous or streaming audio into fixed-duration sliding analysis windows
    annotated with VAD speech activity metadata.
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()

    def create_chunks(
        self,
        audio: np.ndarray,
        sample_rate: int,
        vad_segments: Optional[List[VADSegment]] = None
    ) -> List[AudioChunk]:
        """
        Partition audio into sliding analysis windows.

        Args:
            audio: 1D preprocessed numpy array (float32).
            sample_rate: Sampling rate in Hz.
            vad_segments: Optional list of VAD segments to compute speech ratio per chunk.

        Returns:
            List of AudioChunk objects ready for ML feature extraction.
        """
        if audio is None or len(audio) == 0 or sample_rate <= 0:
            return []

        total_samples = len(audio)
        total_duration = total_samples / sample_rate

        window_samples = int(self.config.chunk_window_sec * sample_rate)
        hop_samples = int(self.config.chunk_hop_sec * sample_rate)
        min_samples = int(self.config.min_chunk_duration_sec * sample_rate)

        chunks: List[AudioChunk] = []

        # Handle short audio smaller than one window
        if total_samples < window_samples:
            if total_samples >= min_samples:
                if self.config.pad_short_chunks:
                    padded_audio = np.zeros(window_samples, dtype=np.float32)
                    padded_audio[:total_samples] = audio
                    chunk_samples = padded_audio
                    is_padded = True
                else:
                    chunk_samples = audio
                    is_padded = False

                speech_ratio, has_speech = self._calculate_chunk_vad(
                    0.0, total_duration, vad_segments
                )
                chunks.append(AudioChunk(
                    chunk_index=0,
                    start_time_sec=0.0,
                    end_time_sec=float(round(total_duration, 3)),
                    sample_rate=sample_rate,
                    samples=chunk_samples,
                    contains_speech=has_speech,
                    speech_ratio=speech_ratio,
                    is_padded=is_padded
                ))
            return chunks

        # Sliding window chunking
        chunk_idx = 0
        start_sample = 0
        last_end_sample = 0

        while start_sample < total_samples:
            end_sample = start_sample + window_samples

            if end_sample <= total_samples:
                chunk_data = audio[start_sample:end_sample]
                start_sec = start_sample / sample_rate
                end_sec = end_sample / sample_rate
                is_padded = False
                last_end_sample = end_sample
            else:
                # If all audio has already been fully included in the previous window, stop
                if last_end_sample >= total_samples:
                    break

                remaining_samples = total_samples - start_sample
                if remaining_samples < min_samples:
                    break

                if self.config.pad_short_chunks:
                    chunk_data = np.zeros(window_samples, dtype=np.float32)
                    chunk_data[:remaining_samples] = audio[start_sample:]
                    is_padded = True
                else:
                    chunk_data = audio[start_sample:]
                    is_padded = False

                start_sec = start_sample / sample_rate
                end_sec = total_duration
                last_end_sample = total_samples

            speech_ratio, has_speech = self._calculate_chunk_vad(
                start_sec, end_sec, vad_segments
            )

            chunks.append(AudioChunk(
                chunk_index=chunk_idx,
                start_time_sec=float(round(start_sec, 3)),
                end_time_sec=float(round(end_sec, 3)),
                sample_rate=sample_rate,
                samples=chunk_data.astype(np.float32),
                contains_speech=has_speech,
                speech_ratio=speech_ratio,
                is_padded=is_padded
            ))

            chunk_idx += 1
            start_sample += hop_samples

            if hop_samples <= 0:
                break

        return chunks

    @staticmethod
    def _calculate_chunk_vad(
        start_sec: float,
        end_sec: float,
        vad_segments: Optional[List[VADSegment]]
    ) -> tuple[float, bool]:
        """
        Calculate the active speech ratio and presence within [start_sec, end_sec].
        """
        if not vad_segments:
            return 1.0, True

        chunk_dur = end_sec - start_sec
        if chunk_dur <= 0:
            return 0.0, False

        overlap_speech_sec = 0.0
        for seg in vad_segments:
            if not seg.is_speech:
                continue
            overlap_start = max(start_sec, seg.start_seconds)
            overlap_end = min(end_sec, seg.end_seconds)
            if overlap_end > overlap_start:
                overlap_speech_sec += (overlap_end - overlap_start)

        speech_ratio = min(1.0, max(0.0, overlap_speech_sec / chunk_dur))
        contains_speech = speech_ratio > 0.05

        return float(round(speech_ratio, 3)), contains_speech


class StreamingAudioBuffer:
    """
    Ring buffer for real-time live streaming audio ingestion (e.g. VoIP / Telephony packets / WebAudio).
    Accumulates incoming live frames at native input sample rate and yields ready, continuous 16kHz
    analysis windows for continuous scoring without chunk-boundary phase clicks.
    """

    def __init__(
        self,
        config: Optional[IngestionConfig] = None,
        input_sample_rate: Optional[int] = None,
        window_duration_sec: Optional[float] = None,
        hop_duration_sec: Optional[float] = None,
        min_speech_duration_sec: Optional[float] = None
    ):
        self.config = config or IngestionConfig()
        self.target_sr = self.config.target_sample_rate
        self.input_sr = input_sample_rate or self.target_sr
        self.win_sec = window_duration_sec if window_duration_sec is not None else self.config.chunk_window_sec
        self.hop_sec = hop_duration_sec if hop_duration_sec is not None else self.config.chunk_hop_sec
        self.min_sp_sec = min_speech_duration_sec if min_speech_duration_sec is not None else self.config.min_chunk_duration_sec

        self.window_samples = int(round(self.win_sec * self.input_sr))
        self.hop_samples = int(round(self.hop_sec * self.input_sr))
        self.min_samples = int(round(self.min_sp_sec * self.input_sr))
        self.buffer = np.zeros(0, dtype=np.float32)
        self.total_samples_received = 0
        self.chunk_count = 0
        self.vad = VoiceActivityDetector(config=self.config)

    def set_input_sample_rate(self, input_sr: int):
        """Dynamically update input sample rate when WebSocket handshake config is received."""
        if input_sr > 0 and input_sr != self.input_sr:
            self.input_sr = input_sr
            self.window_samples = int(round(self.win_sec * self.input_sr))
            self.hop_samples = int(round(self.hop_sec * self.input_sr))
            self.min_samples = int(round(self.min_sp_sec * self.input_sr))
            self.buffer = np.zeros(0, dtype=np.float32)

    def add_samples(self, new_samples: np.ndarray) -> List[AudioChunk]:
        """
        Add incoming real-time audio packet and return any ready analysis chunks.
        """
        if len(new_samples) == 0:
            return []

        self.buffer = np.concatenate([self.buffer, new_samples.astype(np.float32)])
        ready_chunks: List[AudioChunk] = []

        while len(self.buffer) >= self.window_samples:
            raw_chunk = self.buffer[:self.window_samples].copy()
            
            start_sample = self.total_samples_received
            end_sample = start_sample + self.window_samples
            start_sec = start_sample / self.input_sr
            end_sec = end_sample / self.input_sr

            # Resample continuous window to target 16kHz with polyphase filter if needed
            if self.input_sr != self.target_sr:
                from .preprocessor import AudioPreprocessor
                chunk_samples = AudioPreprocessor.resample(raw_chunk, self.input_sr, self.target_sr)
            else:
                chunk_samples = raw_chunk

            # Multi-feature VAD on live stream window at target 16kHz
            peak = float(np.max(np.abs(chunk_samples))) if len(chunk_samples) > 0 else 0.0
            rms = float(np.sqrt(np.mean(chunk_samples ** 2) + 1e-12))
            rms_dbfs = 20.0 * np.log10(max(rms, 1e-9))

            vad_segments, sp_sec, sil_sec, speech_ratio = self.vad.detect_voice_activity(
                chunk_samples,
                self.target_sr
            )
            contains_speech = bool(sp_sec >= 0.20 and speech_ratio >= 0.10 and peak >= 0.015 and rms_dbfs > -50.0)

            ready_chunks.append(AudioChunk(
                chunk_index=self.chunk_count,
                start_time_sec=float(round(start_sec, 3)),
                end_time_sec=float(round(end_sec, 3)),
                sample_rate=self.target_sr,
                samples=chunk_samples,
                contains_speech=contains_speech,
                speech_ratio=float(round(speech_ratio, 3)),
                is_padded=False
            ))

            self.chunk_count += 1
            self.total_samples_received += self.hop_samples
            self.buffer = self.buffer[self.hop_samples:]

        return ready_chunks

    def reset(self):
        """Reset internal buffer state."""
        self.buffer = np.zeros(0, dtype=np.float32)
        self.total_samples_received = 0
        self.chunk_count = 0
