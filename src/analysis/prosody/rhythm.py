"""
Speech Rhythm and Syllabic Timing Module.
Extracts Syllable Nuclei, Speaking Rate, Articulation Rate, and nPVI/rPVI Variability.
"""

from typing import List, Tuple
import numpy as np
from scipy import signal

from .config import ProsodyAnalysisConfig
from .models import RhythmMetrics


class RhythmAnalyzer:
    """Extracts temporal speech rhythm and syllabic duration variability."""

    def __init__(self, config: ProsodyAnalysisConfig):
        self.config = config

    def analyze(
        self,
        audio: np.ndarray,
        sample_rate: int,
        speech_duration_sec: float
    ) -> RhythmMetrics:
        """
        Analyze rhythm, speaking rate, and pairwise variability index.
        """
        if len(audio) == 0 or sample_rate <= 0:
            return RhythmMetrics()

        total_dur_sec = len(audio) / sample_rate

        # 1. Bandpass filter in vowel formant band (200 Hz - 3000 Hz) to isolate syllable nuclei
        nyq = sample_rate / 2.0
        low = max(0.01, 200.0 / nyq)
        high = min(0.95, 3000.0 / nyq)
        b, a = signal.butter(2, [low, high], btype='band')
        filtered = signal.lfilter(b, a, audio)

        # 2. Compute smooth intensity envelope (using moving RMS)
        frame_len = int(0.030 * sample_rate)  # 30ms envelope window
        hop_len = int(0.010 * sample_rate)    # 10ms hop
        num_frames = (len(filtered) - frame_len) // hop_len + 1

        if num_frames < 5:
            return RhythmMetrics()

        envelope = np.zeros(num_frames)
        for i in range(num_frames):
            frame = filtered[i * hop_len: i * hop_len + frame_len]
            envelope[i] = np.sqrt(np.mean(frame ** 2))

        max_env = np.max(envelope)
        if max_env <= 1e-6:
            return RhythmMetrics()

        env_db = 20.0 * np.log10(envelope / max_env + 1e-6)

        # 3. Peak picking for syllable nuclei with adaptive threshold
        min_distance_frames = max(2, int((self.config.min_syllable_duration_ms / 1000.0) / 0.010))
        peaks, _ = signal.find_peaks(
            env_db,
            distance=min_distance_frames,
            height=max(-35.0, self.config.syllable_energy_threshold_db),
            prominence=1.5
        )

        syllable_count = len(peaks)
        peak_times_sec = peaks * 0.010

        # Compute intervals between syllable peaks
        if len(peak_times_sec) > 1:
            durations_ms = np.diff(peak_times_sec) * 1000.0
            npvi, rpvi = self._compute_pvi(durations_ms)
        else:
            npvi, rpvi = 0.0, 0.0

        speaking_rate = syllable_count / total_dur_sec if total_dur_sec > 0 else 0.0
        artic_rate = syllable_count / speech_duration_sec if speech_duration_sec > 0 else speaking_rate

        # 4. Detect pauses from low-energy envelope intervals
        pause_frames = env_db < -30.0
        min_pause_frames = max(2, int((self.config.min_pause_duration_ms / 1000.0) / 0.010))
        pause_count, mean_pause_sec = self._count_pauses(pause_frames, min_pause_frames)

        return RhythmMetrics(
            syllable_count=syllable_count,
            speaking_rate_sps=float(round(speaking_rate, 2)),
            articulation_rate_sps=float(round(artic_rate, 2)),
            pause_count=pause_count,
            mean_pause_duration_sec=float(round(mean_pause_sec, 3)),
            npvi=float(round(npvi, 2)),
            rpvi=float(round(rpvi, 2))
        )

    @staticmethod
    def _compute_pvi(durations_ms: np.ndarray) -> Tuple[float, float]:
        """
        Compute Normalized Pairwise Variability Index (nPVI) and Raw PVI (rPVI).
        """
        m = len(durations_ms)
        if m < 2:
            return 0.0, 0.0

        diffs = np.abs(np.diff(durations_ms))
        rpvi = float(np.mean(diffs))

        npvi_terms = []
        for k in range(m - 1):
            pair_mean = (durations_ms[k] + durations_ms[k + 1]) / 2.0
            if pair_mean > 0:
                npvi_terms.append(abs(durations_ms[k] - durations_ms[k + 1]) / pair_mean)

        npvi = float(np.mean(npvi_terms) * 100.0) if npvi_terms else 0.0
        return npvi, rpvi

    @staticmethod
    def _count_pauses(pause_flags: np.ndarray, min_frames: int) -> Tuple[int, float]:
        """Count continuous silence pauses exceeding min_frames."""
        pauses = []
        curr_len = 0

        for is_pause in pause_flags:
            if is_pause:
                curr_len += 1
            else:
                if curr_len >= min_frames:
                    pauses.append(curr_len * 0.010)
                curr_len = 0

        if curr_len >= min_frames:
            pauses.append(curr_len * 0.010)

        count = len(pauses)
        mean_dur = float(np.mean(pauses)) if count > 0 else 0.0
        return count, mean_dur
