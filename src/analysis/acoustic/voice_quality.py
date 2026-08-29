from typing import List, Tuple
import numpy as np
from scipy import signal

from .config import AcousticAnalysisConfig
from .models import VoiceQualityResult


class VoiceQualityAnalyzer:
    # Measures Harmonics-to-Noise Ratio (HNR) and Cepstral Peak Prominence (CPP)

    def __init__(self, config: AcousticAnalysisConfig = AcousticAnalysisConfig()):
        self.config = config

    def analyze(self, audio: np.ndarray, sample_rate: int) -> VoiceQualityResult:
        if len(audio) == 0 or sample_rate <= 0:
            return VoiceQualityResult()

        frame_len = int(self.config.frame_length_ms * sample_rate / 1000)
        hop_len = int(self.config.hop_length_ms * sample_rate / 1000)

        num_frames = (len(audio) - frame_len) // hop_len + 1
        if num_frames <= 0:
            return VoiceQualityResult()

        hnr_values = []
        cpp_values = []
        zcr_values = []

        for i in range(num_frames):
            start = i * hop_len
            frame = audio[start: start + frame_len] * np.hanning(frame_len)

            # 1. Harmonics-to-Noise Ratio via autocorrelation
            hnr = self._compute_frame_hnr(frame, sample_rate)
            if hnr is not None:
                hnr_values.append(hnr)

            # 2. Cepstral Peak Prominence (CPP)
            cpp = self._compute_frame_cpp(frame)
            if cpp is not None:
                cpp_values.append(cpp)

            # 3. Zero-Crossing Rate
            zcr = float(np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0)
            zcr_values.append(zcr)

        # 4. Energy Entropy
        energy_entropy = self._compute_energy_entropy(audio, frame_len, hop_len)

        mean_hnr = float(np.mean(hnr_values)) if hnr_values else 0.0
        std_hnr = float(np.std(hnr_values)) if hnr_values else 0.0
        mean_cpp = float(np.mean(cpp_values)) if cpp_values else 0.0
        std_cpp = float(np.std(cpp_values)) if cpp_values else 0.0
        mean_zcr = float(np.mean(zcr_values)) if zcr_values else 0.0

        return VoiceQualityResult(
            hnr_mean_db=round(mean_hnr, 2),
            hnr_std_db=round(std_hnr, 2),
            cpp_mean_db=round(mean_cpp, 2),
            cpp_std_db=round(std_cpp, 2),
            zero_crossing_rate_mean=round(mean_zcr, 4),
            energy_entropy=round(energy_entropy, 3)
        )

    def _compute_frame_hnr(self, frame: np.ndarray, sample_rate: int) -> float:
        # Autocorrelation peak ratio
        corr = signal.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]

        min_lag = int(sample_rate / self.config.pitch_fmax)
        max_lag = int(sample_rate / self.config.pitch_fmin)

        if max_lag >= len(corr):
            return None

        lag_region = corr[min_lag:max_lag]
        if len(lag_region) == 0:
            return None

        peak_val = np.max(lag_region)
        zero_lag = corr[0] + 1e-8

        norm_r = peak_val / zero_lag
        if norm_r >= 0.999:
            return 30.0
        elif norm_r <= 0.01:
            return 0.0
        else:
            return float(10.0 * np.log10(norm_r / (1.0 - norm_r)))

    @staticmethod
    def _compute_frame_cpp(frame: np.ndarray) -> float:
        # Real cepstrum: IFFT of log magnitude FFT
        spectrum = np.fft.rfft(frame, n=1024)
        log_mag = np.log(np.maximum(np.abs(spectrum), 1e-6))
        cepstrum = np.real(np.fft.irfft(log_mag))

        quef_min, quef_max = 20, 200
        if quef_max >= len(cepstrum):
            return None

        quef_region = np.abs(cepstrum[quef_min:quef_max])
        if len(quef_region) == 0:
            return None

        peak_val = np.max(quef_region)
        if peak_val <= 1e-12:
            return 0.0
        mean_val = np.mean(quef_region) + 1e-8

        return float(20.0 * np.log10(peak_val / mean_val))

    @staticmethod
    def _compute_energy_entropy(audio: np.ndarray, frame_len: int, hop_len: int, num_subframes: int = 10) -> float:
        if len(audio) < frame_len:
            return 0.0

        energies = []
        num_frames = (len(audio) - frame_len) // hop_len + 1
        for i in range(min(num_frames, 50)):
            frame = audio[i * hop_len: i * hop_len + frame_len]
            energies.append(np.sum(frame ** 2))

        total_energy = sum(energies) + 1e-8
        probs = [e / total_energy for e in energies if e > 0]
        entropy = -sum(p * np.log2(p + 1e-12) for p in probs)
        return float(entropy)
