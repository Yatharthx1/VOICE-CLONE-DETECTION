from typing import List, Tuple
import numpy as np
from scipy import signal

from .config import AcousticAnalysisConfig
from .models import PitchAnalysisResult


class PitchAnalyzer:
    # Fundamental frequency (F0) tracker + robot pitch flatline detector

    def __init__(self, config: AcousticAnalysisConfig = AcousticAnalysisConfig()):
        self.config = config

    def analyze(self, audio: np.ndarray, sample_rate: int) -> PitchAnalysisResult:
        if len(audio) == 0 or sample_rate <= 0:
            return PitchAnalysisResult()

        frame_len = int(self.config.frame_length_ms * sample_rate / 1000)
        hop_len = int(self.config.hop_length_ms * sample_rate / 1000)

        num_frames = (len(audio) - frame_len) // hop_len + 1
        if num_frames <= 0:
            return PitchAnalysisResult()

        f0_contour = []
        voiced_frames = []
        cycle_periods = []
        cycle_amplitudes = []

        min_lag = int(sample_rate / self.config.pitch_fmax)
        max_lag = int(sample_rate / self.config.pitch_fmin)

        for i in range(num_frames):
            start = i * hop_len
            frame = audio[start: start + frame_len]
            windowed = frame * np.hanning(len(frame))

            corr = signal.correlate(windowed, windowed, mode='full')
            corr = corr[len(corr) // 2:]

            if max_lag >= len(corr):
                f0_contour.append(0.0)
                continue

            lag_region = corr[min_lag:max_lag]
            if len(lag_region) == 0:
                f0_contour.append(0.0)
                continue

            peak_lag = np.argmax(lag_region) + min_lag
            norm_peak = corr[peak_lag] / (corr[0] + 1e-8)

            if norm_peak >= self.config.voicing_threshold:
                # Sub-sample parabolic peak fitting for smooth pitch estimation
                if 0 < peak_lag < len(corr) - 1:
                    alpha = corr[peak_lag - 1]
                    beta = corr[peak_lag]
                    gamma = corr[peak_lag + 1]
                    denom = 2 * (2 * beta - alpha - gamma) + 1e-8
                    delta = (alpha - gamma) / denom
                    refined_lag = peak_lag + delta
                else:
                    refined_lag = float(peak_lag)

                pitch = sample_rate / refined_lag
                f0_contour.append(float(pitch))
                voiced_frames.append(float(pitch))
                cycle_periods.append(float(refined_lag / sample_rate))
                cycle_amplitudes.append(float(np.max(np.abs(frame))))
            else:
                f0_contour.append(0.0)

        voiced_ratio = len(voiced_frames) / num_frames if num_frames > 0 else 0.0

        if len(voiced_frames) >= 2:
            mean_f0 = float(np.mean(voiced_frames))
            std_f0 = float(np.std(voiced_frames))
            min_f0 = float(np.min(voiced_frames))
            max_f0 = float(np.max(voiced_frames))
        else:
            mean_f0 = std_f0 = min_f0 = max_f0 = 0.0

        jitters = self._compute_jitter(cycle_periods)
        shimmers = self._compute_shimmer(cycle_amplitudes)

        return PitchAnalysisResult(
            f0_contour=f0_contour,
            mean_f0_hz=round(mean_f0, 2),
            std_f0_hz=round(std_f0, 2),
            min_f0_hz=round(min_f0, 2),
            max_f0_hz=round(max_f0, 2),
            voiced_fraction=round(voiced_ratio, 3),
            jitter_local_pct=round(jitters['local'], 3),
            jitter_rap_pct=round(jitters['rap'], 3),
            jitter_ppq5_pct=round(jitters['ppq5'], 3),
            shimmer_local_pct=round(shimmers['local'], 3),
            shimmer_apq3_pct=round(shimmers['apq3'], 3),
            shimmer_apq5_pct=round(shimmers['apq5'], 3),
            shimmer_db=round(shimmers['db'], 3)
        )

    @staticmethod
    def _compute_jitter(periods: List[float]) -> dict:
        n = len(periods)
        if n < 5:
            return {'local': 0.0, 'rap': 0.0, 'ppq5': 0.0}

        p = np.array(periods)
        mean_p = np.mean(p)
        if mean_p <= 0:
            return {'local': 0.0, 'rap': 0.0, 'ppq5': 0.0}

        j_local = (np.mean(np.abs(np.diff(p))) / mean_p) * 100.0

        rap_terms = []
        for i in range(1, n - 1):
            smooth_3 = (p[i - 1] + p[i] + p[i + 1]) / 3.0
            rap_terms.append(abs(p[i] - smooth_3))
        j_rap = (np.mean(rap_terms) / mean_p) * 100.0 if rap_terms else 0.0

        ppq_terms = []
        for i in range(2, n - 2):
            smooth_5 = np.mean(p[i - 2: i + 3])
            ppq_terms.append(abs(p[i] - smooth_5))
        j_ppq5 = (np.mean(ppq_terms) / mean_p) * 100.0 if ppq_terms else 0.0

        return {
            'local': float(j_local),
            'rap': float(j_rap),
            'ppq5': float(j_ppq5)
        }

    @staticmethod
    def _compute_shimmer(amplitudes: List[float]) -> dict:
        n = len(amplitudes)
        if n < 5:
            return {'local': 0.0, 'apq3': 0.0, 'apq5': 0.0, 'db': 0.0}

        a = np.array(amplitudes)
        mean_a = np.mean(a)
        if mean_a <= 0:
            return {'local': 0.0, 'apq3': 0.0, 'apq5': 0.0, 'db': 0.0}

        s_local = (np.mean(np.abs(np.diff(a))) / mean_a) * 100.0

        apq3_terms = []
        for i in range(1, n - 1):
            smooth_3 = (a[i - 1] + a[i] + a[i + 1]) / 3.0
            apq3_terms.append(abs(a[i] - smooth_3))
        s_apq3 = (np.mean(apq3_terms) / mean_a) * 100.0 if apq3_terms else 0.0

        apq5_terms = []
        for i in range(2, n - 2):
            smooth_5 = np.mean(a[i - 2: i + 3])
            apq5_terms.append(abs(a[i] - smooth_5))
        s_apq5 = (np.mean(apq5_terms) / mean_a) * 100.0 if apq5_terms else 0.0

        ratio_terms = []
        for i in range(n - 1):
            if a[i] > 0 and a[i + 1] > 0:
                ratio_terms.append(abs(20.0 * np.log10(a[i + 1] / a[i])))
        s_db = float(np.mean(ratio_terms)) if ratio_terms else 0.0

        return {
            'local': float(s_local),
            'apq3': float(s_apq3),
            'apq5': float(s_apq5),
            'db': float(s_db)
        }
