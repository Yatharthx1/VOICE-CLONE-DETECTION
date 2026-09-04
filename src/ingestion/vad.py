"""
Voice Activity Detection (VAD) & Segmentation Module.
Accurately splits continuous audio into speech and non-speech/silence intervals,
calculates speech-to-silence ratios, and generates boundary timestamps for ML feature extraction.
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy import signal

from .config import IngestionConfig
from .models import VADSegment


class VoiceActivityDetector:
    """
    Robust Voice Activity Detector & Non-Speech Discriminator combining:
    1. Short-Time Energy (STE) with adaptive noise floor estimation.
    2. Speech Formant Band (140 - 3800 Hz) spectral concentration vs Sub-Bass (< 85 Hz).
    3. Normalized Autocorrelation Phonation Tracking in human pitch range (75 - 420 Hz).
    4. Zero-Crossing Rate (ZCR) dynamics for unvoiced speech consonants (/s/, /sh/, /t/, /f/).
    5. Music & Instrumental Rejection:
       - Rejection of low-frequency music beats (kick drum, 808 bass, bass guitar).
       - Rejection of polyphonic chords (multiple non-harmonic instrument notes sounding simultaneously).
       - Rejection of non-speech percussive hits, ambient hum, and pure synthetic tones.
    6. Temporal hangover smoothing and minimum speech burst filtering.
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()

    def detect_voice_activity(
        self,
        audio: np.ndarray,
        sample_rate: int
    ) -> Tuple[List[VADSegment], float, float, float]:
        """
        Segment the audio into speech and non-speech regions while rejecting
        music, instrumentals, environmental noise, and pure silence.

        Returns:
            Tuple of:
            - List[VADSegment]: list of classified contiguous segments
            - speech_duration_sec: total speech duration in seconds
            - silence_duration_sec: total silence duration in seconds
            - speech_ratio: fraction of audio containing speech (0.0 to 1.0)
        """
        if audio is None or len(audio) == 0 or sample_rate <= 0:
            return [], 0.0, 0.0, 0.0

        dur = len(audio) / sample_rate

        # 1. Silence / extreme low amplitude check
        peak = float(np.max(np.abs(audio)))
        if peak < 0.005:
            seg = VADSegment(
                segment_id=0,
                is_speech=False,
                start_seconds=0.0,
                end_seconds=dur,
                duration_seconds=dur,
                start_sample=0,
                end_sample=len(audio),
                mean_energy_db=-100.0
            )
            return [seg], 0.0, float(round(dur, 3)), 0.0

        rms_full = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        rms_db_full = 20.0 * np.log10(rms_full)
        if rms_db_full < -55.0:
            seg = VADSegment(
                segment_id=0,
                is_speech=False,
                start_seconds=0.0,
                end_seconds=dur,
                duration_seconds=dur,
                start_sample=0,
                end_sample=len(audio),
                mean_energy_db=float(round(rms_db_full, 2))
            )
            return [seg], 0.0, float(round(dur, 3)), 0.0

        # 2. Global spectral check: Sub-bass (< 85 Hz) & Speech Band (120 - 3800 Hz)
        if self.config.vad_reject_music and len(audio) >= int(sample_rate * 0.5):
            n_glob = min(len(audio), int(sample_rate * 3.0))
            glob_w = audio[:n_glob] * np.hanning(n_glob)
            glob_spec = np.abs(np.fft.rfft(glob_w)) ** 2
            glob_f = np.fft.rfftfreq(n_glob, 1.0 / sample_rate)
            glob_tot = np.sum(glob_spec) + 1e-12
            glob_subbass = np.sum(glob_spec[glob_f < self.config.vad_subbass_cutoff_hz]) / glob_tot
            glob_speechband = np.sum(
                glob_spec[(glob_f >= self.config.vad_speech_band_min_hz) & (glob_f <= self.config.vad_speech_band_max_hz)]
            ) / glob_tot

            # Heavy sub-bass music / EDM beat / kick drum dominance
            if glob_subbass > 0.40 and glob_speechband < 0.30:
                seg = VADSegment(
                    segment_id=0,
                    is_speech=False,
                    start_seconds=0.0,
                    end_seconds=dur,
                    duration_seconds=dur,
                    start_sample=0,
                    end_sample=len(audio),
                    mean_energy_db=float(round(rms_db_full, 2))
                )
                return [seg], 0.0, float(round(dur, 3)), 0.0

        # High-pass filter at 85 Hz to strip background sub-bass music/kicks
        # This isolates spoken human voice from background music / song accompaniment
        sos = signal.butter(4, 85.0, btype='highpass', fs=sample_rate, output='sos')
        audio_proc = signal.sosfilt(sos, audio)

        # 3. Frame-level parameters
        frame_len = int(sample_rate * (self.config.vad_frame_duration_ms / 1000.0))
        hop_len = frame_len // 2  # 50% overlap for boundary precision

        if len(audio_proc) < frame_len:
            is_sp = (rms_db_full > self.config.vad_energy_threshold_db)
            seg = VADSegment(
                segment_id=0,
                is_speech=is_sp,
                start_seconds=0.0,
                end_seconds=dur,
                duration_seconds=dur,
                start_sample=0,
                end_sample=len(audio),
                mean_energy_db=float(round(rms_db_full, 2))
            )
            sp_dur = dur if is_sp else 0.0
            sil_dur = 0.0 if is_sp else dur
            return [seg], sp_dur, sil_dur, (1.0 if is_sp else 0.0)

        num_frames = (len(audio_proc) - frame_len) // hop_len + 1
        freqs = np.fft.rfftfreq(frame_len, 1.0 / sample_rate)
        speechband_m = (freqs >= self.config.vad_speech_band_min_hz) & (freqs <= self.config.vad_speech_band_max_hz)

        min_lag = max(1, int(sample_rate / self.config.vad_pitch_max_hz))
        max_lag = min(frame_len - 1, int(sample_rate / self.config.vad_pitch_min_hz))

        frame_rms_db = np.zeros(num_frames, dtype=np.float32)
        for i in range(num_frames):
            start = i * hop_len
            frame = audio_proc[start: start + frame_len]
            rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
            frame_rms_db[i] = 20.0 * np.log10(rms)

        noise_floor_db = float(np.percentile(frame_rms_db, 15))
        max_energy_db = float(np.max(frame_rms_db))

        if noise_floor_db < -35.0:
            adaptive_threshold = max(-48.0, noise_floor_db + 5.0)
        else:
            adaptive_threshold = -48.0
        adaptive_threshold = min(adaptive_threshold, max_energy_db - 2.0)

        raw_speech_flags = np.zeros(num_frames, dtype=bool)
        pitches = []
        p_vals = []
        frame_zcrs = []
        frame_energies = []
        unvoiced_count = 0

        for i in range(num_frames):
            start = i * hop_len
            frame = audio_proc[start: start + frame_len]
            edb = frame_rms_db[i]
            zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)
            frame_zcrs.append(zcr)
            frame_energies.append(10.0 ** (edb / 20.0))

            # Silence rejection
            if edb < adaptive_threshold or edb < -52.0:
                continue

            win = frame * np.hanning(len(frame))
            spec = np.abs(np.fft.rfft(win)) ** 2
            fe = np.sum(spec) + 1e-12
            sp = np.sum(spec[speechband_m]) / fe

            # Autocorrelation in human voice pitch range [75, 420 Hz]
            corr = signal.correlate(win, win, mode='full')
            corr = corr[len(corr) // 2:]
            norm_corr = corr / (corr[0] + 1e-12)
            region = norm_corr[min_lag:max_lag]

            peak_lag = (np.argmax(region) + min_lag) if len(region) > 0 else 0
            peak_val = float(region[peak_lag - min_lag]) if len(region) > 0 else 0.0
            pitch_hz = sample_rate / peak_lag if peak_lag > 0 else 0.0

            is_voiced = (peak_val >= 0.30 and self.config.vad_pitch_min_hz <= pitch_hz <= self.config.vad_pitch_max_hz and sp >= 0.20)
            is_unvoiced = (zcr >= 0.12 and sp >= 0.16)

            if is_voiced:
                pitches.append(pitch_hz)
                p_vals.append(peak_val)
            if is_unvoiced:
                unvoiced_count += 1

            if is_voiced or is_unvoiced:
                raw_speech_flags[i] = True

        # 4. Multi-Feature Music & Song Rejection
        if self.config.vad_reject_music and num_frames >= 20:
            # Minimum active vocal or speech frames
            if len(pitches) < max(2, int(0.02 * num_frames)) and np.sum(raw_speech_flags) < max(3, int(0.03 * num_frames)):
                seg = VADSegment(
                    segment_id=0,
                    is_speech=False,
                    start_seconds=0.0,
                    end_seconds=dur,
                    duration_seconds=dur,
                    start_sample=0,
                    end_sample=len(audio),
                    mean_energy_db=float(round(rms_db_full, 2))
                )
                return [seg], 0.0, float(round(dur, 3)), 0.0

            # Rejection of pure polyphonic instrumental chords (piano, guitar chords)
            if unvoiced_count == 0 and len(p_vals) > 0 and float(np.mean(p_vals)) < 0.45 and np.max(frame_zcrs) < 0.10:
                seg = VADSegment(
                    segment_id=0,
                    is_speech=False,
                    start_seconds=0.0,
                    end_seconds=dur,
                    duration_seconds=dur,
                    start_sample=0,
                    end_sample=len(audio),
                    mean_energy_db=float(round(rms_db_full, 2))
                )
                return [seg], 0.0, float(round(dur, 3)), 0.0

            # Rejection of songs with singing melodies & continuous accompaniment
            mean_e = np.mean(frame_energies)
            lefr = np.mean(np.array(frame_energies) < 0.35 * mean_e)
            zcr_consonants = np.mean(np.array(frame_zcrs) > 0.12)

            if lefr < 0.08 and zcr_consonants < 0.03:
                cur_hold = 0.0
                cur_p = 0.0
                distinct_notes = []
                total_held = 0.0
                for p in pitches:
                    if p > 0:
                        if cur_p == 0.0:
                            cur_p = p
                            cur_hold = 0.015
                        elif abs(p - cur_p) / cur_p < 0.04:
                            cur_hold += 0.015
                        else:
                            if cur_hold >= 0.35:
                                total_held += cur_hold
                                if not any(abs(cur_p - dn) / dn < 0.04 for dn in distinct_notes):
                                    distinct_notes.append(cur_p)
                            cur_p = p
                            cur_hold = 0.015
                    else:
                        if cur_hold >= 0.35:
                            total_held += cur_hold
                            if not any(abs(cur_p - dn) / dn < 0.04 for dn in distinct_notes):
                                distinct_notes.append(cur_p)
                        cur_p = 0.0
                        cur_hold = 0.0
                if cur_hold >= 0.35:
                    total_held += cur_hold
                    if not any(abs(cur_p - dn) / dn < 0.04 for dn in distinct_notes):
                        distinct_notes.append(cur_p)

                if len(distinct_notes) >= 3 and total_held / dur >= 0.30:
                    seg = VADSegment(
                        segment_id=0,
                        is_speech=False,
                        start_seconds=0.0,
                        end_seconds=dur,
                        duration_seconds=dur,
                        start_sample=0,
                        end_sample=len(audio),
                        mean_energy_db=float(round(rms_db_full, 2))
                    )
                    return [seg], 0.0, float(round(dur, 3)), 0.0

        # 5. Temporal Hangover Smoothing
        min_speech_frames = max(1, int(self.config.vad_min_speech_duration_ms / (self.config.vad_frame_duration_ms / 2)))
        min_silence_frames = max(1, int(self.config.vad_min_silence_duration_ms / (self.config.vad_frame_duration_ms / 2)))

        smoothed_flags = self._apply_hangover_smoothing(
            raw_speech_flags,
            min_speech_frames,
            min_silence_frames
        )

        # 6. Group consecutive frames into VADSegments
        segments: List[VADSegment] = []
        if len(smoothed_flags) == 0:
            return [], 0.0, 0.0, 0.0

        current_is_speech = bool(smoothed_flags[0])
        current_start_frame = 0
        seg_id = 0

        for f_idx in range(1, len(smoothed_flags)):
            if bool(smoothed_flags[f_idx]) != current_is_speech:
                start_sample = current_start_frame * hop_len
                end_sample = min(len(audio), f_idx * hop_len + frame_len)
                start_sec = start_sample / sample_rate
                end_sec = end_sample / sample_rate
                dur_sec = max(0.0, end_sec - start_sec)

                seg_frames = frame_rms_db[current_start_frame:f_idx]
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

        # Final segment
        start_sample = current_start_frame * hop_len
        end_sample = len(audio)
        start_sec = start_sample / sample_rate
        end_sec = end_sample / sample_rate
        dur_sec = max(0.0, end_sec - start_sec)
        seg_frames = frame_rms_db[current_start_frame:]
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
