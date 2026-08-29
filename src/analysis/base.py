"""
Base abstractions for Parallel Analysis modules in the Voice Integrity Verification Framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from pydantic import BaseModel, ConfigDict


class BaseAnalysisResult(BaseModel):
    """Abstract base class for all parallel analysis output models."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    module_name: str
    anomaly_score: float  # 0.0 (genuine) to 1.0 (highly synthetic/cloned)
    confidence: float     # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to a dictionary."""
        return self.model_dump()


class BaseAnalysisModule(ABC):
    """Abstract base class for all parallel analysis engines."""

    @abstractmethod
    def analyze(self, *args: Any, **kwargs: Any) -> BaseAnalysisResult:
        """Run analysis on preprocessed audio or audio chunks."""
        pass
