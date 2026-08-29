"""
Audio Preprocessor.
Performs mono conversion, high-quality polyphase resampling (16kHz / 24kHz),
DC offset removal, and loudness/peak normalization for downstream feature extraction.
"""

from math import gcd
from typing import Optional, Union
import numpy as np
from scipy import signal

from .config import IngestionConfig


class AudioPreprocessor:
    """
    Standardizes raw audio into clean, normalized mono signals at the target sample rate.
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()

    def preprocess(
        self,
        audio: np.ndarray,
        source_sample_rate: int
    ) -> np.ndarray:
        """
        Full preprocessing pipeline:
        1. Downmix to mono (if enabled)
        2. Remove DC bias
        3. Resample to target_sample_rate
        4. Normalize amplitude
        
        Returns:
            Preprocessed 1D numpy array of float32 samples.
        """
        if audio is None or audio.size == 0:
            return np.zeros(0, dtype=np.float32)

        # 1. Convert to float32
        data = audio.astype(np.float32)

        # 2. Downmix multi-channel to mono
        if self.config.to_mono:
            data = self.to_mono(data)

        # 3. Remove DC Offset
        if self.config.remove_dc_offset:
            data = self.remove_dc_offset(data)

        # 4. Resample to target rate (e.g. 16kHz or 24kHz)
        if source_sample_rate != self.config.target_sample_rate:
            data = self.resample(data, source_sample_rate, self.config.target_sample_rate)

        # 5. Normalize
        if self.config.normalize_audio:
            data = self.normalize(data, target_dbfs=self.config.target_dbfs)

        return data.astype(np.float32)

    @staticmethod
    def to_mono(audio: np.ndarray) -> np.ndarray:
        """
        Downmix multi-channel audio to mono by taking channel-wise mean.
        """
        if audio.ndim == 1:
            return audio
        elif audio.ndim == 2:
            # If shape is (samples, channels), average along axis 1
            if audio.shape[1] > audio.shape[0] and audio.shape[0] in [1, 2, 4, 6, 8]:
                # Channel-first format (channels, samples)
                return np.mean(audio, axis=0)
            return np.mean(audio, axis=1)
        elif audio.ndim > 2:
            return np.mean(audio, axis=tuple(range(1, audio.ndim)))
        return audio

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        High-fidelity anti-aliased polyphase resampling using scipy.signal.resample_poly.
        Preserves pitch, duration, and frequency response without spectral distortion.
        """
        if orig_sr == target_sr or len(audio) == 0:
            return audio

        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g

        # Resample polyphase
        resampled = signal.resample_poly(audio, up, down, axis=0)
        return resampled.astype(np.float32)

    @staticmethod
    def remove_dc_offset(audio: np.ndarray, cutoff_hz: float = 20.0, sample_rate: int = 16000) -> np.ndarray:
        """
        Remove DC bias using mean subtraction and a highpass filter (> 20Hz).
        """
        if len(audio) == 0:
            return audio

        # Step 1: Subtract mean
        centered = audio - np.mean(audio)

        # Step 2: Gentle 1st order Butterworth highpass if audio is long enough
        if len(centered) > 64 and sample_rate > 2 * cutoff_hz:
            try:
                b, a = signal.butter(1, cutoff_hz / (0.5 * sample_rate), btype='high')
                filtered = signal.lfilter(b, a, centered)
                return filtered.astype(np.float32)
            except Exception:
                pass

        return centered.astype(np.float32)

    def normalize(
        self,
        audio: np.ndarray,
        target_dbfs: Optional[float] = None
    ) -> np.ndarray:
        """
        Normalize audio amplitude to target RMS dBFS while ensuring peak ceiling is respected.
        """
        if len(audio) == 0:
            return audio

        target_db = target_dbfs if target_dbfs is not None else self.config.target_dbfs
        peak_ceiling = self.config.peak_normalize_ceiling

        # Current RMS
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-7:
            return audio  # Pure silence, do not amplify noise

        current_dbfs = 20.0 * np.log10(rms)
        gain_db = target_db - current_dbfs
        gain_linear = 10.0 ** (gain_db / 20.0)

        scaled = audio * gain_linear

        # Peak limiter to avoid clipping beyond ceiling
        max_peak = np.max(np.abs(scaled))
        if max_peak > peak_ceiling:
            scaled = scaled * (peak_ceiling / max_peak)

        return np.clip(scaled, -1.0, 1.0).astype(np.float32)
