from typing import List, Tuple
import numpy as np
from scipy import signal

from .config import AcousticAnalysisConfig
from .models import FormantResult


class FormantAnalyzer:
    # Levinson-Durbin root finding to measure the speaker's vocal tract length

    def __init__(self, config: AcousticAnalysisConfig = AcousticAnalysisConfig()):
        self.config = config

    def analyze(self, audio: np.ndarray, sample_rate: int) -> FormantResult:
        if len(audio) == 0 or sample_rate <= 0:
            return FormantResult()

        # Target 16kHz for clean formant root solving
        if sample_rate != 16000 and len(audio) > 100:
            target_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, target_samples)
            sample_rate = 16000

        # Pre-emphasis filter to boost higher frequencies
        pre_emph = np.append(audio[0], audio[1:] - self.config.pre_emphasis_coeff * audio[:-1])

        frame_len = int(self.config.frame_length_ms * sample_rate / 1000)
        hop_len = int(self.config.hop_length_ms * sample_rate / 1000)

        num_frames = (len(pre_emph) - frame_len) // hop_len + 1
        if num_frames <= 0:
            return FormantResult()

        f1_list, f2_list, f3_list, f4_list = [], [], [], []

        lpc_order = int(self.config.lpc_order_offset + sample_rate / 1000)

        for i in range(num_frames):
            start = i * hop_len
            frame = pre_emph[start: start + frame_len] * np.hamming(frame_len)

            # Levinson-Durbin LPC root extraction
            try:
                a = self._levinson_durbin(frame, lpc_order)
                roots = np.roots(a)
                # Keep roots in the upper half-plane with positive frequency
                roots = [r for r in roots if np.imag(r) > 0.01 and np.abs(r) < 0.999]

                angles = np.arctan2(np.imag(roots), np.real(roots))
                freqs = sorted(angles * (sample_rate / (2 * np.pi)))

                # Filter plausible human formant frequencies
                valid_formants = [f for f in freqs if 200 <= f <= 5000]

                if len(valid_formants) >= 1:
                    f1_list.append(valid_formants[0])
                if len(valid_formants) >= 2:
                    f2_list.append(valid_formants[1])
                if len(valid_formants) >= 3:
                    f3_list.append(valid_formants[2])
                if len(valid_formants) >= 4:
                    f4_list.append(valid_formants[3])
            except Exception:
                continue

        f1_mean = float(np.mean(f1_list)) if f1_list else 0.0
        f2_mean = float(np.mean(f2_list)) if f2_list else 0.0
        f3_mean = float(np.mean(f3_list)) if f3_list else 0.0
        f4_mean = float(np.mean(f4_list)) if f4_list else 0.0

        f1_std = float(np.std(f1_list)) if len(f1_list) > 1 else 0.0
        f2_std = float(np.std(f2_list)) if len(f2_list) > 1 else 0.0
        f3_std = float(np.std(f3_list)) if len(f3_list) > 1 else 0.0

        # Formant dispersion (delta F) and estimated vocal tract length in cm
        if f4_mean > 0 and f1_mean > 0:
            formant_dispersion = float((f4_mean - f1_mean) / 3.0)
        elif f3_mean > 0 and f1_mean > 0:
            formant_dispersion = float((f3_mean - f1_mean) / 2.0)
        else:
            formant_dispersion = 1000.0

        # Speed of sound c = 35000 cm/s
        vtl_cm = float(35000.0 / (2.0 * formant_dispersion)) if formant_dispersion > 0 else 17.5
        vtl_cm = float(np.clip(vtl_cm, 10.0, 25.0))

        return FormantResult(
            f1_mean_hz=round(f1_mean, 2) if f1_mean > 0 else None,
            f2_mean_hz=round(f2_mean, 2) if f2_mean > 0 else None,
            f3_mean_hz=round(f3_mean, 2) if f3_mean > 0 else None,
            f4_mean_hz=round(f4_mean, 2) if f4_mean > 0 else None,
            f1_std_hz=round(f1_std, 2) if f1_std > 0 else None,
            f2_std_hz=round(f2_std, 2) if f2_std > 0 else None,
            f3_std_hz=round(f3_std, 2) if f3_std > 0 else None,
            formant_dispersion_hz=round(formant_dispersion, 2),
            vocal_tract_length_cm=round(vtl_cm, 2)
        )

    @staticmethod
    def _levinson_durbin(frame: np.ndarray, order: int) -> np.ndarray:
        # Autocorrelation method for LPC coefficients
        r = signal.correlate(frame, frame, mode='full')
        r = r[len(r) // 2: len(r) // 2 + order + 1]

        if r[0] == 0:
            return np.ones(order + 1)

        a = np.zeros(order + 1)
        a[0] = 1.0
        e = r[0]

        for i in range(1, order + 1):
            gamma = -np.dot(a[:i], r[i:0:-1]) / (e + 1e-8)
            a_new = a.copy()
            a_new[1:i] = a[1:i] + gamma * a[i - 1:0:-1]
            a_new[i] = gamma
            a = a_new
            e *= (1.0 - gamma ** 2)
            if e <= 0:
                break

        return a
