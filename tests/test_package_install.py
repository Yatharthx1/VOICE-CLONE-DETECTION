"""
Unit tests verifying pip-installable package functionality and top-level exports.
"""

from pathlib import Path
import pytest


def test_top_level_package_exports():
    """Verify primary SDK classes and convenience functions are accessible at package top level."""
    import src as pkg

    assert hasattr(pkg, "VoiceIntegrityEngine")
    assert hasattr(pkg, "RiskScenario")
    assert hasattr(pkg, "Verdict")
    assert hasattr(pkg, "verify")
    assert hasattr(pkg, "__version__")
    assert pkg.__version__ == "1.0.0"


def test_convenience_verify_helper(speech_wav):
    """Verify top-level verify() function runs inference on an audio file."""
    import src as pkg

    result = pkg.verify(speech_wav)
    assert result.session_id is not None
    assert result.assessment.verdict is not None
    assert 0.0 <= result.assessment.dynamic_risk_score <= 100.0
    assert "REPORT" in result.summary()
