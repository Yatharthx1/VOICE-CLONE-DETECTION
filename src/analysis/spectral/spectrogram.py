"""
Spectrogram and Mel-Filterbank Computation Module.
"""

from typing import Tuple
import numpy as np


class SpectrogramExtractor:
    """Computes STFT magnitude spectrograms and log Mel-spectrograms."""

    def __init__(self, n_fft: int = 1024, hop_length: int = 256, n_mels: int = 80):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def compute_stft(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute STFT magnitude spectrogram and complex STFT.
        
        Returns:
            Tuple of (magnitude_spectrogram, complex_stft)
            Shape: (n_fft // 2 + 1, num_frames)
        """
        if len(audio) < self.n_fft:
            pad = np.zeros(self.n_fft, dtype=np.float32)
            pad[:len(audio)] = audio
            audio = pad

        window = np.hanning(self.n_fft)
        num_frames = (len(audio) - self.n_fft) // self.hop_length + 1
        
        # Frame extraction
        shape = (num_frames, self.n_fft)
        strides = (audio.strides[0] * self.hop_length, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides) * window

        complex_stft = np.fft.rfft(frames, n=self.n_fft, axis=-1).T
        magnitude = np.abs(complex_stft)

        return magnitude.astype(np.float32), complex_stft

    def compute_mel_spectrogram(
        self,
        magnitude: np.ndarray,
        sample_rate: int,
        fmin: float = 20.0,
        fmax: float = 8000.0
    ) -> np.ndarray:
        """
        Convert linear STFT magnitude spectrogram to log Mel-spectrogram.
        """
        n_freqs = magnitude.shape[0]
        mel_filters = self._create_mel_filterbank(
            n_freqs=n_freqs,
            sample_rate=sample_rate,
            n_mels=self.n_mels,
            fmin=fmin,
            fmax=min(fmax, sample_rate / 2.0)
        )
        mel_spec = np.dot(mel_filters, magnitude)
        log_mel_spec = np.log(np.maximum(mel_spec, 1e-6))
        return log_mel_spec.astype(np.float32)

    @staticmethod
    def _create_mel_filterbank(
        n_freqs: int,
        sample_rate: int,
        n_mels: int,
        fmin: float,
        fmax: float
    ) -> np.ndarray:
        """Create triangular Mel filterbank matrix."""
        # Convert Hz to Mel
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        min_mel = hz_to_mel(fmin)
        max_mel = hz_to_mel(fmax)
        mels = np.linspace(min_mel, max_mel, n_mels + 2)
        hz_points = mel_to_hz(mels)

        # Convert Hz to FFT bin indices
        fft_freqs = np.linspace(0, sample_rate / 2.0, n_freqs)
        filterbank = np.zeros((n_mels, n_freqs), dtype=np.float32)

        for m in range(1, n_mels + 1):
            f_left = hz_points[m - 1]
            f_center = hz_points[m]
            f_right = hz_points[m + 1]

            # Up-slope
            mask_up = (fft_freqs >= f_left) & (fft_freqs <= f_center)
            if f_center > f_left:
                filterbank[m - 1, mask_up] = (fft_freqs[mask_up] - f_left) / (f_center - f_left)

            # Down-slope
            mask_down = (fft_freqs >= f_center) & (fft_freqs <= f_right)
            if f_right > f_center:
                filterbank[m - 1, mask_down] = (f_right - fft_freqs[mask_down]) / (f_right - f_center)

        return filterbank
