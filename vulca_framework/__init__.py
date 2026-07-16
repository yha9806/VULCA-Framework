"""VULCA cultural-evaluation research prototype.

The public package currently exposes an uncalibrated Tier I + Tier II
experimental evaluator. It does not ship the paper's fitted Tier III sigmoid
calibrator or the private human-annotation pool used to fit it.
"""

from .trilayer_evaluator import TriLayerEvaluator, TriLayerResult, evaluate_critique
from .automated_metrics import AutomatedMetrics
from .checklist_judge import ChecklistJudge, ChecklistResult, JudgeBackendError
from .metrics import VULCAMetricsEvaluator

__version__ = "0.1.0"
__author__ = "VULCA Project Team"

__all__ = [
    "TriLayerEvaluator",
    "TriLayerResult",
    "evaluate_critique",
    "AutomatedMetrics",
    "ChecklistJudge",
    "ChecklistResult",
    "JudgeBackendError",
    "VULCAMetricsEvaluator",
]
