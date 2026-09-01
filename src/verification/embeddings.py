from typing import Optional
import numpy as np


class SpeakerEmbeddingExtractor:
    # Extracts a 128-D acoustic voiceprint so we can tell Alice from an imposter

    def __init__(self, embedding_dim: int = 128, n_mels: int = 40):
        self.embedding_dim = embedding_dim
        self.n_mels = n_mels

    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        # If the audio is too short to be speech, return a normalized unit vector instead of crashing
        if len(audio) < 1024 or sample_rate <= 0:
            vec = np.ones(self.embedding_dim, dtype=np.float32)
            return vec / np.linalg.norm(vec)

        # STFT computation for filterbank extraction
        n_fft = 512
        hop = 160
        window = np.hanning(n_fft)
        num_frames = (len(audio) - n_fft) // hop + 1

        if num_frames < 4:
            pad = np.pad(audio, (0, 2048), mode='reflect')
            return self.extract_embedding(pad, sample_rate)

        shape = (num_frames, n_fft)
        strides = (audio.strides[0] * hop, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides) * window
        mag = np.abs(np.fft.rfft(frames, n=n_fft, axis=-1)).T

        # Mel filterbanks + statistical temporal pooling (mean, std, 25th, 75th percentiles)
        mel_filters = self._build_mel_filters(n_freqs=mag.shape[0], sample_rate=sample_rate, n_mels=self.n_mels)
        mel_energies = np.dot(mel_filters, mag)
        log_mel = np.log(np.maximum(mel_energies, 1e-5))

        mean_feat = np.mean(log_mel, axis=1)
        std_feat = np.std(log_mel, axis=1)
        p25 = np.percentile(log_mel, 25, axis=1)
        p75 = np.percentile(log_mel, 75, axis=1)
        pooled = np.concatenate([mean_feat, std_feat, p25, p75])

        # Random projection down to 128-D embedding space (seeded for reproducibility)
        np.random.seed(1337)
        proj_matrix = np.random.randn(160, self.embedding_dim).astype(np.float32) / np.sqrt(160)
        embedding = np.dot(pooled, proj_matrix)

        # L2-normalization: unit sphere or bust
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        else:
            embedding = np.ones(self.embedding_dim, dtype=np.float32) / np.sqrt(self.embedding_dim)

        return embedding.astype(np.float32)

    @staticmethod
    def _build_mel_filters(n_freqs: int, sample_rate: int, n_mels: int) -> np.ndarray:
        # Standard Mel scale conversions
        def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mels = np.linspace(hz_to_mel(50.0), hz_to_mel(sample_rate / 2.0), n_mels + 2)
        hz_pts = mel_to_hz(mels)
        fft_freqs = np.linspace(0, sample_rate / 2.0, n_freqs)
        fbank = np.zeros((n_mels, n_freqs), dtype=np.float32)

        for m in range(1, n_mels + 1):
            f_l, f_c, f_r = hz_pts[m - 1], hz_pts[m], hz_pts[m + 1]
            mask_up = (fft_freqs >= f_l) & (fft_freqs <= f_c)
            if f_c > f_l:
                fbank[m - 1, mask_up] = (fft_freqs[mask_up] - f_l) / (f_c - f_l)
            mask_dn = (fft_freqs >= f_c) & (fft_freqs <= f_r)
            if f_r > f_c:
                fbank[m - 1, mask_dn] = (f_r - fft_freqs[mask_dn]) / (f_r - f_c)

        return fbank
