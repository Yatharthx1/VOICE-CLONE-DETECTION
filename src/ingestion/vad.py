"""
Voice Activity Detection (VAD) & Segmentation Module.
Accurately splits continuous audio into speech and non-speech/silence intervals,
calculates speech-to-silence ratios, and generates boundary timestamps for ML feature extraction.
"""

from typing import List, Optional, Tuple
import numpy as np

from .config import IngestionConfig
from .models import VADSegment


class VoiceActivityDetector:
    """
    Robust, dependency-free Voice Activity Detector combining Short-Time Energy (STE),
    Zero-Crossing Rate (ZCR), adaptive noise floor tracking, and temporal hangover smoothing.
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()

    def detect_voice_activity(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Tuple[List[VADSegment], float, float, float]:
        """
        Segment the audio into speech and non-speech regions.

        Returns:
            Tuple of:
            - List[VADSegment]: list of classified contiguous segments
            - speech_duration_sec: total speech duration in seconds
            - silence_duration_sec: total silence duration in seconds
            - speech_ratio: fraction of audio containing speech (0.0 to 1.0)
        """
        if audio is None or len(audio) == 0 or sample_rate <= 0:
            return [], 0.0, 0.0, 0.0

        # Frame parameters
        frame_len = int(sample_rate * (self.config.vad_frame_duration_ms / 1000.0))
        hop_len = frame_len // 2  # 50% overlap for smooth boundary detection

        if len(audio) < frame_len:
            # Too short for framing, evaluate globally
            energy = np.mean(audio ** 2)
            energy_db = 20.0 * np.log10(np.sqrt(energy) + 1e-9)
            is_speech = energy_db > self.config.vad_energy_threshold_db
            dur = len(audio) / sample_rate
            seg = VADSegment(
                segment_id=0,
                is_speech=is_speech,
                start_seconds=0.0,
                end_seconds=dur,
                duration_seconds=dur,
                start_sample=0,
                end_sample=len(audio),
                mean_energy_db=float(round(energy_db, 2))
            )
            sp_dur = dur if is_speech else 0.0
            sil_dur = 0.0 if is_speech else dur
            return [seg], sp_dur, sil_dur, (1.0 if is_speech else 0.0)

        # 1. Compute frame energies and ZCR
        num_frames = (len(audio) - frame_len) // hop_len + 1
        frame_energies_db = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            start = i * hop_len
            frame = audio[start: start + frame_len]
            rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
            frame_energies_db[i] = 20.0 * np.log10(rms)

        # 2. Adaptive noise floor estimation
        noise_floor_db = np.percentile(frame_energies_db, 15)
        # Dynamic threshold adapts to noisy environments while respecting configured base threshold
        adaptive_threshold = max(
            self.config.vad_energy_threshold_db,
            noise_floor_db + 10.0
        )

        # Initial raw speech decisions per frame
        raw_speech_flags = frame_energies_db > adaptive_threshold

        # 3. Temporal Smoothing (Hangover and Minimum Duration filter)
        min_speech_frames = max(1, int(self.config.vad_min_speech_duration_ms / (self.config.vad_frame_duration_ms / 2)))
        min_silence_frames = max(1, int(self.config.vad_min_silence_duration_ms / (self.config.vad_frame_duration_ms / 2)))

        smoothed_flags = self._apply_hangover_smoothing(
            raw_speech_flags,
            min_speech_frames,
            min_silence_frames
        )

        # 4. Group consecutive frames into VADSegments
        segments: List[VADSegment] = []
        if len(smoothed_flags) == 0:
            return [], 0.0, 0.0, 0.0

        current_is_speech = bool(smoothed_flags[0])
        current_start_frame = 0
        seg_id = 0

        for f_idx in range(1, len(smoothed_flags)):
            if bool(smoothed_flags[f_idx]) != current_is_speech:
                # Segment boundary reached
                start_sample = current_start_frame * hop_len
                end_sample = min(len(audio), f_idx * hop_len + frame_len)
                start_sec = start_sample / sample_rate
                end_sec = end_sample / sample_rate
                dur_sec = max(0.0, end_sec - start_sec)
                
                seg_frames = frame_energies_db[current_start_frame:f_idx]
                mean_e = float(np.mean(seg_frames)) if len(seg_frames) > 0 else -100.0

                segments.append(VADSegment(
                    segment_id=seg_id,
                    is_speech=current_is_speech,
                    start_seconds=float(round(start_sec, 3)),
                    end_seconds=float(round(end_sec, 3)),
                    duration_seconds=float(round(dur_sec, 3)),
                    start_sample=start_sample,
                    end_sample=end_sample,
                    mean_energy_db=float(round(mean_e, 2))
                ))
                seg_id += 1
                current_is_speech = bool(smoothed_flags[f_idx])
                current_start_frame = f_idx

        # Append final segment
        start_sample = current_start_frame * hop_len
        end_sample = len(audio)
        start_sec = start_sample / sample_rate
        end_sec = end_sample / sample_rate
        dur_sec = max(0.0, end_sec - start_sec)
        seg_frames = frame_energies_db[current_start_frame:]
        mean_e = float(np.mean(seg_frames)) if len(seg_frames) > 0 else -100.0

        segments.append(VADSegment(
            segment_id=seg_id,
            is_speech=current_is_speech,
            start_seconds=float(round(start_sec, 3)),
            end_seconds=float(round(end_sec, 3)),
            duration_seconds=float(round(dur_sec, 3)),
            start_sample=start_sample,
            end_sample=end_sample,
            mean_energy_db=float(round(mean_e, 2))
        ))

        # Compute aggregate metrics
        total_audio_sec = len(audio) / sample_rate
        speech_sec = sum(seg.duration_seconds for seg in segments if seg.is_speech)
        silence_sec = max(0.0, total_audio_sec - speech_sec)
        speech_ratio = speech_sec / total_audio_sec if total_audio_sec > 0 else 0.0

        return segments, float(round(speech_sec, 3)), float(round(silence_sec, 3)), float(round(speech_ratio, 4))

    @staticmethod
    def _apply_hangover_smoothing(
        flags: np.ndarray,
        min_speech_frames: int,
        min_silence_frames: int
    ) -> np.ndarray:
        """
        Applies forward hangover and minimum speech duration filtering:
        1. Fills small silence gaps between speech bursts (hangover).
        2. Discards isolated speech spikes shorter than min_speech_frames.
        """
        smoothed = flags.copy()
        n = len(smoothed)

        # 1. Fill small silence gaps (hangover bridging)
        silence_count = 0
        in_speech = False
        gap_start = 0

        for i in range(n):
            if smoothed[i]:
                if not in_speech:
                    in_speech = True
                else:
                    if 0 < silence_count <= min_silence_frames:
                        # Bridge this silence gap
                        smoothed[gap_start:i] = True
                silence_count = 0
            else:
                if in_speech:
                    if silence_count == 0:
                        gap_start = i
                    silence_count += 1
                    if silence_count > min_silence_frames:
                        in_speech = False

        # 2. Filter out isolated short speech bursts
        speech_count = 0
        speech_start = 0
        for i in range(n):
            if smoothed[i]:
                if speech_count == 0:
                    speech_start = i
                speech_count += 1
            else:
                if 0 < speech_count < min_speech_frames:
                    smoothed[speech_start:i] = False
                speech_count = 0

        if 0 < speech_count < min_speech_frames:
            smoothed[speech_start:n] = False

        return smoothed

    @staticmethod
    def extract_speech_only(audio: np.ndarray, segments: List[VADSegment]) -> np.ndarray:
        """
        Concatenate all active speech segments, stripping out silence and background noise.
        """
        speech_parts = []
        for seg in segments:
            if seg.is_speech:
                speech_parts.append(audio[seg.start_sample: seg.end_sample])

        if not speech_parts:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(speech_parts).astype(np.float32)
