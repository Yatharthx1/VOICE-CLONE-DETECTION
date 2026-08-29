"""
Audio Metadata Extractor.
Extracts container format, codec details, sample rate, channel count, duration,
bit depth, and acoustic signal metrics (Peak, RMS, Clipping, Dynamic Range, SNR).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import soundfile as sf

from .models import AudioMetadata


class MetadataExtractor:
    """
    Extracts deep file format, container, codec, and acoustic metrics from audio streams and files.
    """

    def __init__(self):
        self._has_ffprobe = shutil.which("ffprobe") is not None

    def extract_from_file(
        self,
        file_path: Union[str, Path],
        decoded_audio: Optional[np.ndarray] = None,
        sample_rate: Optional[int] = None
    ) -> AudioMetadata:
        """
        Extract complete container, codec, and acoustic metadata from an audio file.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size = path.stat().st_size

        # 1. Probe container and codec metadata
        probe_meta = self._probe_file_format(path)
        sr = probe_meta.get("sample_rate") or sample_rate or 16000
        channels = probe_meta.get("channels", 1)

        # 2. Extract acoustic metrics
        if decoded_audio is None:
            # Decode file locally to compute exact acoustic statistics
            try:
                decoded_audio, sr_read = sf.read(str(path), dtype="float32", always_2d=False)
                sr = sr_read
                channels = decoded_audio.shape[1] if decoded_audio.ndim > 1 else 1
            except Exception:
                try:
                    from .decoder import AudioDecoder
                    decoded_audio, sr, channels = AudioDecoder().decode_file(path)
                except Exception:
                    decoded_audio = None

        acoustic_stats = {}
        if decoded_audio is not None:
            acoustic_stats = self.compute_acoustic_statistics(decoded_audio, sr)
        elif probe_meta.get("duration", 0) > 0:
            dur = probe_meta.get("duration", 0.0)
            acoustic_stats["total_samples"] = int(dur * sr)
            acoustic_stats["duration_seconds"] = dur

        total_samples = acoustic_stats.get("total_samples", 0)
        duration_sec = acoustic_stats.get("duration_seconds", probe_meta.get("duration", 0.0))

        if total_samples == 0 and sr > 0 and duration_sec > 0:
            total_samples = int(duration_sec * sr)

        return AudioMetadata(
            file_path=str(path.resolve()),
            file_size_bytes=file_size,
            container_format=probe_meta.get("format_name", path.suffix.lstrip(".").lower() or "unknown"),
            codec=probe_meta.get("codec_name", "unknown"),
            sample_rate=sr,
            channels=channels,
            bit_depth=probe_meta.get("bit_depth"),
            bitrate_kbps=probe_meta.get("bitrate_kbps"),
            duration_seconds=float(duration_sec),
            total_samples=total_samples,
            peak_dbfs=acoustic_stats.get("peak_dbfs", 0.0),
            rms_dbfs=acoustic_stats.get("rms_dbfs", -100.0),
            is_clipped=acoustic_stats.get("is_clipped", False),
            dynamic_range_db=acoustic_stats.get("dynamic_range_db", 0.0),
            snr_estimate_db=acoustic_stats.get("snr_estimate_db")
        )

    def extract_from_array(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        container_format: str = "raw_pcm",
        codec: str = "pcm_f32le"
    ) -> AudioMetadata:
        """
        Extract metadata and acoustic statistics directly from a decoded in-memory array.
        """
        if audio_array.ndim == 1:
            channels = 1
            total_samples = len(audio_array)
        else:
            channels = audio_array.shape[1] if audio_array.ndim == 2 else 1
            total_samples = audio_array.shape[0]

        duration = total_samples / sample_rate if sample_rate > 0 else 0.0
        acoustic_stats = self.compute_acoustic_statistics(audio_array, sample_rate)

        return AudioMetadata(
            file_path=None,
            file_size_bytes=audio_array.nbytes,
            container_format=container_format,
            codec=codec,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=32,  # float32
            bitrate_kbps=float(sample_rate * channels * 32 / 1000.0),
            duration_seconds=float(duration),
            total_samples=total_samples,
            peak_dbfs=acoustic_stats.get("peak_dbfs", 0.0),
            rms_dbfs=acoustic_stats.get("rms_dbfs", -100.0),
            is_clipped=acoustic_stats.get("is_clipped", False),
            dynamic_range_db=acoustic_stats.get("dynamic_range_db", 0.0),
            snr_estimate_db=acoustic_stats.get("snr_estimate_db")
        )

    def _probe_file_format(self, file_path: Path) -> Dict[str, Any]:
        """Probe format and codec information using ffprobe or soundfile fallback."""
        if self._has_ffprobe:
            try:
                return self._ffprobe_inspect(file_path)
            except Exception:
                pass

        # Fallback to soundfile.info
        try:
            info = sf.info(str(file_path))
            return {
                "format_name": info.format.lower(),
                "codec_name": info.subtype.lower(),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration": info.duration,
                "bit_depth": self._subtype_to_bitdepth(info.subtype),
                "bitrate_kbps": None
            }
        except Exception:
            # Generic extension fallback
            ext = file_path.suffix.lstrip(".").lower()
            return {
                "format_name": ext or "unknown",
                "codec_name": ext or "unknown",
                "sample_rate": 16000,
                "channels": 1,
                "duration": 0.0,
                "bit_depth": 16,
                "bitrate_kbps": None
            }

    def _ffprobe_inspect(self, file_path: Path) -> Dict[str, Any]:
        """Run ffprobe in JSON output mode to extract full stream and format metadata."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-select_streams", "a:0",
            str(file_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)

        streams = data.get("streams", [])
        fmt = data.get("format", {})

        if not streams and not fmt:
            raise ValueError(f"No valid audio streams found in {file_path}")

        stream = streams[0] if streams else {}

        codec_name = stream.get("codec_name", fmt.get("format_name", "unknown"))
        format_name = fmt.get("format_name", file_path.suffix.lstrip(".")).split(",")[0]
        sample_rate = int(stream.get("sample_rate") or fmt.get("sample_rate") or 16000)
        channels = int(stream.get("channels") or 1)
        duration = float(stream.get("duration") or fmt.get("duration") or 0.0)

        # Bit depth estimation
        bits_raw = stream.get("bits_per_raw_sample") or stream.get("bits_per_sample")
        bit_depth = int(bits_raw) if bits_raw and str(bits_raw).isdigit() else None

        # Bitrate in kbps
        bitrate = stream.get("bit_rate") or fmt.get("bit_rate")
        bitrate_kbps = float(bitrate) / 1000.0 if bitrate and str(bitrate).replace(".", "").isdigit() else None

        return {
            "format_name": format_name,
            "codec_name": codec_name,
            "sample_rate": sample_rate,
            "channels": channels,
            "duration": duration,
            "bit_depth": bit_depth,
            "bitrate_kbps": bitrate_kbps
        }

    @staticmethod
    def compute_acoustic_statistics(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Compute Peak dBFS, RMS loudness, dynamic range, clipping flag, and SNR estimation.
        """
        if audio.size == 0:
            return {
                "total_samples": 0,
                "duration_seconds": 0.0,
                "peak_dbfs": -100.0,
                "rms_dbfs": -100.0,
                "is_clipped": False,
                "dynamic_range_db": 0.0,
                "snr_estimate_db": None
            }

        # Convert to 1D float array for unified statistical calculation
        flat = audio.astype(np.float64)
        if flat.ndim > 1:
            flat = np.mean(flat, axis=1)

        total_samples = len(flat)
        duration_sec = total_samples / sample_rate if sample_rate > 0 else 0.0

        # Peak calculation
        peak = np.max(np.abs(flat))
        peak_dbfs = 20.0 * np.log10(peak) if peak > 1e-7 else -100.0

        # Clipping detection: sample values at or very close to +/- 1.0 (>= 0.999)
        is_clipped = bool(peak >= 0.999)

        # RMS calculation
        rms = np.sqrt(np.mean(flat ** 2))
        rms_dbfs = 20.0 * np.log10(rms) if rms > 1e-7 else -100.0

        # Dynamic range estimation
        frame_size = int(0.030 * sample_rate)  # 30ms frames
        if frame_size > 0 and len(flat) >= frame_size:
            num_frames = len(flat) // frame_size
            frames = flat[: num_frames * frame_size].reshape(num_frames, frame_size)
            frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
            valid_rms = frame_rms[frame_rms > 1e-7]
            if len(valid_rms) > 0:
                frame_dbs = 20.0 * np.log10(valid_rms)
                signal_floor_db = np.percentile(frame_dbs, 10)
                signal_peak_db = np.percentile(frame_dbs, 95)
                dynamic_range_db = float(max(0.0, signal_peak_db - signal_floor_db))
                snr_estimate_db = float(max(0.0, signal_peak_db - signal_floor_db))
            else:
                dynamic_range_db = 0.0
                snr_estimate_db = None
        else:
            dynamic_range_db = 0.0
            snr_estimate_db = None

        return {
            "total_samples": total_samples,
            "duration_seconds": duration_sec,
            "peak_dbfs": float(round(peak_dbfs, 2)),
            "rms_dbfs": float(round(rms_dbfs, 2)),
            "is_clipped": is_clipped,
            "dynamic_range_db": float(round(dynamic_range_db, 2)),
            "snr_estimate_db": float(round(snr_estimate_db, 2)) if snr_estimate_db is not None else None
        }

    @staticmethod
    def _subtype_to_bitdepth(subtype: str) -> Optional[int]:
        sub = subtype.upper()
        if "PCM_16" in sub:
            return 16
        elif "PCM_24" in sub:
            return 24
        elif "PCM_32" in sub or "FLOAT" in sub:
            return 32
        elif "PCM_U8" in sub or "PCM_S8" in sub:
            return 8
        return None
