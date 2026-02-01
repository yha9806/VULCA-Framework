"""
VULCA-Framework: Tri-Tier Evaluation Framework for Cross-Cultural Art Critique

This framework provides a calibrated evaluation pipeline for assessing
Vision-Language Models (VLMs) on art critique tasks across multiple cultures.

Components:
- TriLayerEvaluator: Main evaluation class combining all tiers
- AutomatedMetrics: Tier I automated scoring (DCR, CSA, CDS, LQS)
- ChecklistJudge: Tier II LLM-as-Judge with 10-item rubric
- JudgeCalibration: Tier III isotonic regression calibration

Example:
    from vulca_framework import TriLayerEvaluator

    evaluator = TriLayerEvaluator(culture='chinese')
    result = evaluator.evaluate(
        vlm_critique="...",
        expert_critique="...",
        expert_dimensions=["CN_L1_D1", ...]
    )
    print(f"Score: {result.final_score:.2f}")

Author: VULCA Project Team
Version: 1.0.0
"""

from .trilayer_evaluator import TriLayerEvaluator, TriLayerResult, evaluate_critique
from .automated_metrics import AutomatedMetrics
from .checklist_judge import ChecklistJudge, ChecklistResult
from .judge_calibration import JudgeCalibrator
from .metrics import VULCAMetrics

__version__ = "1.0.0"
__author__ = "VULCA Project Team"

__all__ = [
    "TriLayerEvaluator",
    "TriLayerResult",
    "evaluate_critique",
    "AutomatedMetrics",
    "ChecklistJudge",
    "ChecklistResult",
    "JudgeCalibrator",
    "VULCAMetrics",
]
