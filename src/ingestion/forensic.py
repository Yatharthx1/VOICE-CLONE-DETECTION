"""
Forensic Preservation Module for Voice Integrity Verification.
Preserves raw audio bitstreams, generates cryptographic checksums,
and enforces forensic chain of custody as mandated by the system architecture.
"""

import hashlib
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Union

from .models import ForensicRecord


class ForensicPreserver:
    """
    Manages forensic preservation, cryptographic hashing, and chain of custody
    verification for ingested voice samples.
    """

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None):
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path(tempfile.gettempdir()) / "voice_integrity_forensics"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_hashes_from_bytes(data: bytes) -> Tuple[str, str]:
        """Compute SHA-256 and MD5 hashes from raw bytes."""
        sha256 = hashlib.sha256(data).hexdigest()
        md5 = hashlib.md5(data).hexdigest()
        return sha256, md5

    @staticmethod
    def compute_hashes_from_file(file_path: Union[str, Path]) -> Tuple[str, str]:
        """Compute SHA-256 and MD5 hashes by streaming file chunks."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found for forensic hashing: {file_path}")

        sha256_h = hashlib.sha256()
        md5_h = hashlib.md5()

        with open(path, "rb") as f:
            while chunk := f.read(65536):
                sha256_h.update(chunk)
                md5_h.update(chunk)

        return sha256_h.hexdigest(), md5_h.hexdigest()

    def preserve_file(
        self,
        file_path: Union[str, Path],
        audio_id: Optional[str] = None
    ) -> ForensicRecord:
        """
        Archive an unmodified raw audio file into forensic storage and create its integrity record.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {file_path}")

        if audio_id is None:
            audio_id = str(uuid.uuid4())

        file_size = path.stat().st_size
        sha256, md5 = self.compute_hashes_from_file(path)

        # Store forensic copy preserving original extension
        suffix = path.suffix or ".bin"
        forensic_filename = f"{audio_id}_forensic{suffix}"
        forensic_dest = self.storage_dir / forensic_filename

        shutil.copy2(path, forensic_dest)

        return ForensicRecord(
            audio_id=audio_id,
            sha256_hash=sha256,
            md5_hash=md5,
            ingestion_timestamp=datetime.now(timezone.utc).isoformat(),
            original_filename=path.name,
            original_size_bytes=file_size,
            forensic_copy_path=str(forensic_dest),
            verified_integrity=True
        )

    def preserve_bytes(
        self,
        raw_bytes: bytes,
        original_filename: str = "stream_sample.wav",
        audio_id: Optional[str] = None
    ) -> ForensicRecord:
        """
        Archive raw bytes into forensic storage and create its integrity record.
        """
        if audio_id is None:
            audio_id = str(uuid.uuid4())

        file_size = len(raw_bytes)
        sha256, md5 = self.compute_hashes_from_bytes(raw_bytes)

        suffix = Path(original_filename).suffix or ".bin"
        forensic_filename = f"{audio_id}_forensic{suffix}"
        forensic_dest = self.storage_dir / forensic_filename

        with open(forensic_dest, "wb") as f:
            f.write(raw_bytes)

        return ForensicRecord(
            audio_id=audio_id,
            sha256_hash=sha256,
            md5_hash=md5,
            ingestion_timestamp=datetime.now(timezone.utc).isoformat(),
            original_filename=original_filename,
            original_size_bytes=file_size,
            forensic_copy_path=str(forensic_dest),
            verified_integrity=True
        )

    @staticmethod
    def verify_integrity(record: ForensicRecord) -> bool:
        """
        Recompute hashes from the forensic archive to verify it has not been modified or corrupted.
        """
        if not record.forensic_copy_path or not os.path.exists(record.forensic_copy_path):
            return False

        recomputed_sha256, recomputed_md5 = ForensicPreserver.compute_hashes_from_file(record.forensic_copy_path)
        return (recomputed_sha256 == record.sha256_hash) and (recomputed_md5 == record.md5_hash)
