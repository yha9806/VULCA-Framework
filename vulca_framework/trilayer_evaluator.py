"""
trilayer_evaluator.py - VULCA Tri-Layer Pyramid Evaluation Framework v2.0

Integrates Layer 1 (Automated Metrics) and Layer 2 (Checklist Judge) into
a unified evaluation pipeline with support for Mode A (reference-based) and
Mode B (reference-free).

Final Score = Layer1 (40%) + Layer2 (60%)

Usage:
    from scripts.evaluation.trilayer_evaluator import TriLayerEvaluator

    evaluator = TriLayerEvaluator(culture='chinese')

    # Mode A (with expert reference)
    result = evaluator.evaluate(
        vlm_critique=vlm_text,
        expert_critique=expert_text,
        expert_dimensions=dims,
        mode='A'
    )

    # Mode B (reference-free)
    result = evaluator.evaluate(
        vlm_critique=vlm_text,
        artwork_info="Title: xxx, Artist: yyy",
        mode='B'
    )

    print(f"Final Score: {result.final_score:.2f}")

Author: Claude Code
Version: 1.0 (2025-11-29)
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from .automated_metrics import AutomatedMetrics
    from .checklist_judge import ChecklistJudge, ChecklistResult
except ImportError:
    # For direct execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from automated_metrics import AutomatedMetrics
    from checklist_judge import ChecklistJudge, ChecklistResult


# =============================================================================
# Result Dataclass
# =============================================================================

@dataclass
class TriLayerResult:
    """Complete evaluation result from Tri-Layer framework."""

    # Scores
    layer1_score: float  # 0-1
    layer2_score: float  # 0-1
    final_score: float   # 0-1 (weighted combination)

    # Details
    layer1_metrics: Dict[str, float]
    layer2_result: ChecklistResult

    # Metadata
    mode: str
    culture: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Weights used
    layer1_weight: float = 0.4
    layer2_weight: float = 0.6

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'final_score': self.final_score,
            'layer1_score': self.layer1_score,
            'layer2_score': self.layer2_score,
            'layer1_metrics': self.layer1_metrics,
            'layer2_answers': self.layer2_result.answers,
            'layer2_yes_count': self.layer2_result.yes_count,
            'layer2_no_count': self.layer2_result.no_count,
            'mode': self.mode,
            'culture': self.culture,
            'weights': {
                'layer1': self.layer1_weight,
                'layer2': self.layer2_weight
            },
            'timestamp': self.timestamp
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"=== Tri-Layer Evaluation Summary (Mode {self.mode}) ===",
            f"Culture: {self.culture}",
            f"",
            f"Final Score: {self.final_score:.1%}",
            f"├── Layer 1 (Automated): {self.layer1_score:.1%} × {self.layer1_weight:.0%} = {self.layer1_score * self.layer1_weight:.1%}",
            f"└── Layer 2 (Checklist): {self.layer2_score:.1%} × {self.layer2_weight:.0%} = {self.layer2_score * self.layer2_weight:.1%}",
            f"",
            f"Layer 1 Breakdown:",
        ]

        for metric, value in self.layer1_metrics.items():
            if metric != 'mode':
                lines.append(f"  - {metric}: {value:.3f}")

        lines.append(f"")
        lines.append(f"Layer 2 Breakdown:")
        lines.append(f"  - Yes: {self.layer2_result.yes_count}")
        lines.append(f"  - No: {self.layer2_result.no_count}")
        lines.append(f"  - N/A: {self.layer2_result.na_count}")

        return '\n'.join(lines)


# =============================================================================
# Tri-Layer Evaluator Class
# =============================================================================

class TriLayerEvaluator:
    """
    Tri-Layer Pyramid Evaluation Framework.

    Combines:
    - Layer 1: Automated metrics (40% weight)
    - Layer 2: LLM checklist judge (60% weight)

    Supports:
    - Mode A: Reference-based (requires expert critique)
    - Mode B: Reference-free (no expert needed)
    """

    def __init__(
        self,
        culture: str,
        layer1_weight: float = 0.4,
        layer2_weight: float = 0.6,
        keywords_dir: str = None,
        judge_model: str = 'claude'
    ):
        """
        Initialize Tri-Layer evaluator.

        Args:
            culture: Culture name (chinese, western, japanese, korean, islamic, indian)
            layer1_weight: Weight for Layer 1 (default: 0.4)
            layer2_weight: Weight for Layer 2 (default: 0.6)
            keywords_dir: Path to dimension keywords directory
            judge_model: LLM to use for Layer 2 ('claude', 'gpt5', or 'fallback')
        """
        self.culture = culture.lower()
        self.layer1_weight = layer1_weight
        self.layer2_weight = layer2_weight

        # Initialize components
        self.layer1 = AutomatedMetrics(culture, keywords_dir)
        self.layer2 = ChecklistJudge(judge_model)

    def evaluate(
        self,
        vlm_critique: str,
        expert_critique: str = None,
        expert_dimensions: List[str] = None,
        artwork_info: str = None,
        mode: str = 'A'
    ) -> TriLayerResult:
        """
        Perform complete Tri-Layer evaluation.

        Args:
            vlm_critique: VLM-generated critique text
            expert_critique: Expert reference critique (Mode A only)
            expert_dimensions: Expert covered dimensions (Mode A only)
            artwork_info: Artwork metadata string (Mode B)
            mode: 'A' (reference-based) or 'B' (reference-free)

        Returns:
            TriLayerResult with all scores and details
        """
        mode = mode.upper()

        # Validate inputs
        if mode == 'A':
            if not expert_critique or not expert_dimensions:
                raise ValueError("Mode A requires expert_critique and expert_dimensions")

        # Layer 1: Automated Metrics
        layer1_total, layer1_metrics = self.layer1.compute_layer1_score(
            vlm_critique=vlm_critique,
            expert_critique=expert_critique,
            expert_dimensions=expert_dimensions,
            mode=mode
        )

        # Layer 2: Checklist Judge
        layer2_result = self.layer2.evaluate(
            vlm_critique=vlm_critique,
            expert_critique=expert_critique,
            culture=self.culture,
            artwork_info=artwork_info,
            mode=mode
        )

        # Calculate final score
        final_score = (
            layer1_total * self.layer1_weight +
            layer2_result.score * self.layer2_weight
        )

        return TriLayerResult(
            layer1_score=layer1_total,
            layer2_score=layer2_result.score,
            final_score=final_score,
            layer1_metrics=layer1_metrics,
            layer2_result=layer2_result,
            mode=mode,
            culture=self.culture,
            layer1_weight=self.layer1_weight,
            layer2_weight=self.layer2_weight
        )

    def evaluate_batch(
        self,
        samples: List[Dict],
        mode: str = 'A',
        progress_callback=None
    ) -> List[TriLayerResult]:
        """
        Evaluate a batch of samples.

        Args:
            samples: List of sample dictionaries with keys:
                - vlm_critique: VLM text
                - expert_critique: Expert text (Mode A)
                - expert_dimensions / covered_dimensions: Expert dims (Mode A)
                - artwork_info: Artwork metadata (Mode B)
            mode: 'A' or 'B'
            progress_callback: Optional callback(current, total)

        Returns:
            List of TriLayerResult objects
        """
        results = []
        total = len(samples)

        for i, sample in enumerate(samples):
            vlm_critique = sample.get('vlm_critique', '')

            if mode.upper() == 'A':
                expert_critique = sample.get('expert_critique') or sample.get('critique_zh') or sample.get('critique_en')
                expert_dimensions = sample.get('expert_dimensions') or sample.get('covered_dimensions', [])

                # Parse dimensions if string
                if isinstance(expert_dimensions, str):
                    try:
                        expert_dimensions = json.loads(expert_dimensions)
                    except:
                        expert_dimensions = []

                result = self.evaluate(
                    vlm_critique=vlm_critique,
                    expert_critique=expert_critique,
                    expert_dimensions=expert_dimensions,
                    mode='A'
                )
            else:
                artwork_info = sample.get('artwork_info', f"Culture: {self.culture}")
                result = self.evaluate(
                    vlm_critique=vlm_critique,
                    artwork_info=artwork_info,
                    mode='B'
                )

            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total)

        return results

    def get_batch_statistics(self, results: List[TriLayerResult]) -> Dict:
        """
        Calculate statistics from batch results.

        Args:
            results: List of TriLayerResult objects

        Returns:
            Dictionary with statistics
        """
        if not results:
            return {}

        final_scores = [r.final_score for r in results]
        layer1_scores = [r.layer1_score for r in results]
        layer2_scores = [r.layer2_score for r in results]

        return {
            'count': len(results),
            'final_score': {
                'mean': sum(final_scores) / len(final_scores),
                'min': min(final_scores),
                'max': max(final_scores)
            },
            'layer1_score': {
                'mean': sum(layer1_scores) / len(layer1_scores),
                'min': min(layer1_scores),
                'max': max(layer1_scores)
            },
            'layer2_score': {
                'mean': sum(layer2_scores) / len(layer2_scores),
                'min': min(layer2_scores),
                'max': max(layer2_scores)
            },
            'mode': results[0].mode,
            'culture': results[0].culture
        }


# =============================================================================
# Convenience Function
# =============================================================================

def evaluate_critique(
    vlm_critique: str,
    culture: str,
    expert_critique: str = None,
    expert_dimensions: List[str] = None,
    artwork_info: str = None,
    mode: str = 'A'
) -> TriLayerResult:
    """
    Convenience function for single critique evaluation.

    Args:
        vlm_critique: VLM-generated critique
        culture: Culture name
        expert_critique: Expert reference (Mode A)
        expert_dimensions: Expert dimensions (Mode A)
        artwork_info: Artwork info (Mode B)
        mode: 'A' or 'B'

    Returns:
        TriLayerResult
    """
    evaluator = TriLayerEvaluator(culture)
    return evaluator.evaluate(
        vlm_critique=vlm_critique,
        expert_critique=expert_critique,
        expert_dimensions=expert_dimensions,
        artwork_info=artwork_info,
        mode=mode
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    # Test with sample data
    test_vlm = """
    此幅山水画以浓墨淡彩描绘江南水乡风光，构图采用三远法中的平远视角，
    展现了文人画的典型意境。笔法流畅，墨色层次分明，气韵生动，
    体现了明代吴门画派的艺术特色。画面虚实相生，意境深远，
    充分展现了中国传统绘画的审美追求。作品在技法上运用了传统的皴擦点染，
    同时融入了文人画的诗意表达，体现了画家深厚的艺术修养。
    """

    test_expert = """
    此作为明代吴门画派代表作品，以工写结合的技法描绘江南山水。
    构图疏朗有致，笔墨苍润，气韵生动。画面呈现平远视角，
    山石采用披麻皴法，树木点叶精到。整体展现了文人画的审美理想，
    体现了画家对自然与人文的深刻理解。
    """

    test_dims = ['CN_L1_D1', 'CN_L1_D2', 'CN_L1_D3', 'CN_L2_D1', 'CN_L2_D2',
                 'CN_L3_D1', 'CN_L4_D1', 'CN_L5_D2', 'CN_L5_D3']

    print("=== Tri-Layer Evaluator Test ===\n")

    evaluator = TriLayerEvaluator('chinese', judge_model='fallback')

    # Mode A
    print("--- Mode A (Reference-Based) ---")
    result_a = evaluator.evaluate(
        vlm_critique=test_vlm,
        expert_critique=test_expert,
        expert_dimensions=test_dims,
        mode='A'
    )
    print(result_a.summary())
    print()

    # Mode B
    print("--- Mode B (Reference-Free) ---")
    result_b = evaluator.evaluate(
        vlm_critique=test_vlm,
        artwork_info="Title: 江南山水图, Artist: 佚名, Period: 明代",
        mode='B'
    )
    print(result_b.summary())
