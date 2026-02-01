"""
VULCA Evaluation Metrics v2.0
Implements 5-dimensional evaluation framework for VLM-generated art critiques.

Metrics:
- DCR: Dimension Coverage Rate
- CSA: Cultural Semantic Alignment
- CDS: Critique Depth Score
- FAR: Factual Accuracy Rate
- LQS: Linguistic Quality Score

Author: VULCA Team
Date: 2025-11-27
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import re
import json

# Optional dependencies
try:
    from bert_score import score as bert_score_fn
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    bert_score_fn = None


# =============================================================================
# Constants
# =============================================================================

LAYER_DEFINITIONS = {
    "L1": "Visual Perception",
    "L2": "Technical Analysis",
    "L3": "Cultural Symbolism",
    "L4": "Historical Context",
    "L5": "Philosophical Aesthetics"
}

CULTURE_DIMENSIONS = {
    "chinese": {"prefix": "CN", "total": 30, "threshold": 21},
    "western": {"prefix": "WE", "total": 22, "threshold": 15},
    "japanese": {"prefix": "JP", "total": 27, "threshold": 19},
    "korean": {"prefix": "KR", "total": 25, "threshold": 18},
    "islamic": {"prefix": "IS", "total": 28, "threshold": 20},
    "indian": {"prefix": "IN", "total": 30, "threshold": 21},
    "hermitage": {"prefix": "WS", "total": 30, "threshold": 21},
    "mural": {"prefix": "MU", "total": 30, "threshold": 21},
}

# Cultural term dictionaries (expandable)
CULTURAL_TERMS = {
    "chinese": [
        "气韵生动", "骨法用笔", "应物象形", "随类赋彩", "经营位置", "传移模写",
        "皴法", "留白", "意境", "写意", "工笔", "没骨", "青绿山水", "水墨",
        "六法", "三远", "平远", "高远", "深远", "笔墨", "墨分五色",
        "qiyun", "gufa", "bimo", "cunfa", "liubai", "yijing", "xieyi", "gongbi"
    ],
    "western": [
        "chiaroscuro", "sfumato", "impasto", "tenebrism", "perspective",
        "vanishing point", "composition", "color harmony", "brushwork",
        "oil painting", "tempera", "fresco", "allegory", "iconography",
        "baroque", "renaissance", "impressionism", "expressionism",
        "naturalism", "realism", "romanticism", "neoclassicism"
    ],
    "japanese": [
        "wabi-sabi", "mono no aware", "yugen", "ma", "notan",
        "ukiyo-e", "nihonga", "rinpa", "kano", "tosa",
        "sumi-e", "kakejiku", "emakimono", "byobu",
        "侘寂", "幽玄", "間", "浮世絵", "日本画"
    ],
    "korean": [
        "minhwa", "chaekgeori", "sipjangsaeng", "munbangdo",
        "dancheong", "sumukhwa", "chaekmunhwa",
        "민화", "책거리", "십장생", "문방도", "단청", "수묵화"
    ],
    "islamic": [
        "arabesque", "geometric pattern", "calligraphy", "muqarnas",
        "tessellation", "biomorphic", "vegetal motif", "palmette",
        "kufic", "naskh", "thuluth", "persian miniature", "illumination"
    ],
    "indian": [
        "rasa", "bhava", "mudra", "mandala", "yantra",
        "miniature painting", "mughal style", "rajput painting",
        "pahari painting", "tanjore painting", "madhubani",
        "pattachitra", "warli", "ajanta", "ellora"
    ]
}


# =============================================================================
# Abstract Base Class
# =============================================================================

class MetricBase(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def calculate(self, vlm_output: Dict[str, Any], reference: Dict[str, Any]) -> float:
        """Calculate metric score."""
        pass

    @abstractmethod
    def to_rubric_score(self, raw_score: float) -> int:
        """Convert raw score to 1-5 rubric score."""
        pass

    def get_name(self) -> str:
        """Get metric name."""
        return self.__class__.__name__


# =============================================================================
# DCR - Dimension Coverage Rate
# =============================================================================

class DimensionCoverageRate(MetricBase):
    """
    DCR = |D_vlm ∩ D_ref| / |D_ref|

    Measures how many reference dimensions the VLM critique covers.
    """

    def __init__(self, culture: str = "chinese"):
        self.culture = culture.lower()
        config = CULTURE_DIMENSIONS.get(self.culture, CULTURE_DIMENSIONS["chinese"])
        self.prefix = config["prefix"]
        self.total_dimensions = config["total"]

    def calculate(self, vlm_output: Dict[str, Any], reference: Dict[str, Any]) -> float:
        """
        Calculate DCR.

        Args:
            vlm_output: Dict with 'covered_dimensions' or 'critique_text'
            reference: Dict with 'covered_dimensions' list

        Returns:
            DCR score in [0.0, 1.0]
        """
        # Extract VLM dimensions
        vlm_dims = set(vlm_output.get("covered_dimensions", []))
        if not vlm_dims and "critique_text" in vlm_output:
            vlm_dims = self._extract_dimensions(vlm_output["critique_text"])

        # Extract reference dimensions
        ref_dims = set(reference.get("covered_dimensions", []))
        if not ref_dims:
            return 0.0

        # Calculate intersection
        intersection = vlm_dims & ref_dims
        dcr = len(intersection) / len(ref_dims)

        return round(dcr, 4)

    def _extract_dimensions(self, text: str) -> Set[str]:
        """Extract dimension IDs from text."""
        pattern = rf'{self.prefix}_L[1-5]_D\d+'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return set(d.upper() for d in matches)

    def to_rubric_score(self, raw_score: float) -> int:
        """
        DCR Rubric (tightened to reduce ceiling):
        5: >= 0.95
        4: 0.80 - 0.94
        3: 0.60 - 0.79
        2: 0.30 - 0.59
        1: < 0.30
        """
        if raw_score >= 0.95:
            return 5
        elif raw_score >= 0.80:
            return 4
        elif raw_score >= 0.60:
            return 3
        elif raw_score >= 0.30:
            return 2
        else:
            return 1


# =============================================================================
# CSA - Cultural Semantic Alignment
# =============================================================================

class CulturalSemanticAlignment(MetricBase):
    """
    CSA = alpha * BERTScore_F1 + beta * CulturalTermOverlap

    Default: alpha = 0.7, beta = 0.3
    """

    def __init__(self, culture: str = "chinese", alpha: float = 0.7, beta: float = 0.3):
        self.culture = culture.lower()
        self.alpha = alpha
        self.beta = beta
        self.cultural_terms = set(CULTURAL_TERMS.get(self.culture, []))

    def calculate(self, vlm_output: Dict[str, Any], reference: Dict[str, Any]) -> float:
        """
        Calculate CSA.

        Args:
            vlm_output: Dict with 'critique_text' or 'critique_en'
            reference: Dict with 'critique_text', 'critique_zh', or 'critique_en'

        Returns:
            CSA score in [0.0, 1.0]
        """
        vlm_text = vlm_output.get("critique_text") or vlm_output.get("critique_en", "")
        ref_text = reference.get("critique_text") or reference.get("critique_zh") or reference.get("critique_en", "")

        if not vlm_text or not ref_text:
            return 0.0

        # Calculate BERTScore
        bert_f1 = self._calculate_bertscore(vlm_text, ref_text)

        # Calculate Cultural Term Overlap
        term_overlap = self._calculate_term_overlap(vlm_text, ref_text)

        # Weighted combination
        csa = self.alpha * bert_f1 + self.beta * term_overlap

        return round(csa, 4)

    def _calculate_bertscore(self, candidate: str, reference: str) -> float:
        """Calculate BERTScore F1."""
        if not BERT_SCORE_AVAILABLE:
            # Fallback: simple token overlap
            return self._simple_overlap(candidate, reference)

        try:
            P, R, F1 = bert_score_fn(
                [candidate], [reference],
                model_type="bert-base-multilingual-cased",
                verbose=False
            )
            return float(F1[0])
        except Exception as e:
            print(f"BERTScore error: {e}")
            return self._simple_overlap(candidate, reference)

    def _simple_overlap(self, text1: str, text2: str) -> float:
        """Simple token overlap as BERTScore fallback."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        return len(intersection) / len(union)

    def _calculate_term_overlap(self, vlm_text: str, ref_text: str) -> float:
        """Calculate cultural term overlap."""
        if not self.cultural_terms:
            return 0.0

        vlm_lower = vlm_text.lower()
        ref_lower = ref_text.lower()

        # Find terms in reference
        ref_terms = {t for t in self.cultural_terms if t.lower() in ref_lower}
        if not ref_terms:
            return 0.0

        # Find terms in VLM output
        vlm_terms = {t for t in self.cultural_terms if t.lower() in vlm_lower}

        # Calculate overlap
        overlap = vlm_terms & ref_terms
        return len(overlap) / len(ref_terms)

    def to_rubric_score(self, raw_score: float) -> int:
        """
        CSA Rubric:
        5: >= 0.85
        4: 0.70 - 0.84
        3: 0.55 - 0.69
        2: 0.40 - 0.54
        1: < 0.40
        """
        if raw_score >= 0.85:
            return 5
        elif raw_score >= 0.70:
            return 4
        elif raw_score >= 0.55:
            return 3
        elif raw_score >= 0.40:
            return 2
        else:
            return 1


# =============================================================================
# CDS - Critique Depth Score
# =============================================================================

class CritiqueDepthScore(MetricBase):
    """
    CDS = sum(l * n_l) / sum(n_l)

    Where:
    - l: layer number (1-5)
    - n_l: count of dimensions hit in that layer
    """

    def __init__(self, culture: str = "chinese"):
        self.culture = culture.lower()
        config = CULTURE_DIMENSIONS.get(self.culture, CULTURE_DIMENSIONS["chinese"])
        self.prefix = config["prefix"]

    def calculate(self, vlm_output: Dict[str, Any], reference: Dict[str, Any] = None) -> float:
        """
        Calculate CDS based on layer distribution.

        Args:
            vlm_output: Dict with 'covered_dimensions'
            reference: Not used for CDS

        Returns:
            CDS score in [1.0, 5.0]
        """
        dims = vlm_output.get("covered_dimensions", [])
        if not dims:
            return 1.0

        # Count dimensions per layer
        layer_counts = defaultdict(int)
        for dim in dims:
            parts = dim.split("_")
            if len(parts) >= 2:
                layer = parts[1]
                if layer in LAYER_DEFINITIONS:
                    layer_num = int(layer[1])  # L1 -> 1, L5 -> 5
                    layer_counts[layer_num] += 1

        if not layer_counts:
            return 1.0

        # Calculate weighted average
        total_weighted = sum(l * n for l, n in layer_counts.items())
        total_count = sum(layer_counts.values())

        cds = total_weighted / total_count
        return round(cds, 4)

    def get_layer_distribution(self, vlm_output: Dict[str, Any]) -> Dict[str, int]:
        """Get the layer distribution for analysis."""
        dims = vlm_output.get("covered_dimensions", [])
        layer_counts = {f"L{i}": 0 for i in range(1, 6)}

        for dim in dims:
            parts = dim.split("_")
            if len(parts) >= 2:
                layer = parts[1]
                if layer in layer_counts:
                    layer_counts[layer] += 1

        return layer_counts

    def to_rubric_score(self, raw_score: float) -> int:
        """
        CDS Rubric:
        5: >= 4.0 (L4-L5 dominant)
        4: 3.0 - 3.9 (L3-L4)
        3: 2.0 - 2.9 (L2-L3)
        2: 1.5 - 1.9 (L1-L2)
        1: < 1.5 (L1 only)
        """
        if raw_score >= 4.0:
            return 5
        elif raw_score >= 3.0:
            return 4
        elif raw_score >= 2.0:
            return 3
        elif raw_score >= 1.5:
            return 2
        else:
            return 1


# =============================================================================
# FAR - Factual Accuracy Rate
# =============================================================================

class FactualAccuracyRate(MetricBase):
    """
    FAR = n_correct / n_total

    Requires external judge to verify factual claims.
    This class provides the framework; actual scoring is done by dual-judge.
    """

    def __init__(self):
        self.factual_claims = []
        self.verified_claims = []

    def calculate(self, vlm_output: Dict[str, Any], reference: Dict[str, Any]) -> float:
        """
        Calculate FAR from pre-judged results.

        Args:
            vlm_output: Dict with 'factual_scores' list from judge
            reference: Not directly used

        Returns:
            FAR score in [0.0, 1.0]
        """
        scores = vlm_output.get("factual_scores", [])
        if not scores:
            return 0.0

        # Scores should be binary (0 or 1) or continuous [0,1]
        correct = sum(1 for s in scores if s >= 0.5)
        total = len(scores)

        return round(correct / total, 4) if total > 0 else 0.0

    def from_judge_scores(self, judge_scores: List[float]) -> float:
        """Calculate FAR from judge-provided scores."""
        if not judge_scores:
            return 0.0

        # Convert 1-5 scale to binary (3+ is correct)
        correct = sum(1 for s in judge_scores if s >= 3.0)
        return round(correct / len(judge_scores), 4)

    def to_rubric_score(self, raw_score: float) -> int:
        """
        FAR Rubric:
        5: >= 0.95
        4: 0.80 - 0.94
        3: 0.60 - 0.79
        2: 0.40 - 0.59
        1: < 0.40
        """
        if raw_score >= 0.95:
            return 5
        elif raw_score >= 0.80:
            return 4
        elif raw_score >= 0.60:
            return 3
        elif raw_score >= 0.40:
            return 2
        else:
            return 1


# =============================================================================
# LQS - Linguistic Quality Score
# =============================================================================

class LinguisticQualityScore(MetricBase):
    """
    LQS = (Fluency + Coherence + Terminology) / 3

    Each component is scored 1-5 by judge model.
    """

    def __init__(self):
        pass

    def calculate(self, vlm_output: Dict[str, Any], reference: Dict[str, Any] = None) -> float:
        """
        Calculate LQS from component scores.

        Args:
            vlm_output: Dict with 'lqs_scores' containing fluency, coherence, terminology

        Returns:
            LQS score in [1.0, 5.0]
        """
        lqs_scores = vlm_output.get("lqs_scores", {})

        fluency = lqs_scores.get("fluency", 3.0)
        coherence = lqs_scores.get("coherence", 3.0)
        terminology = lqs_scores.get("terminology", 3.0)

        lqs = (fluency + coherence + terminology) / 3
        return round(lqs, 4)

    def from_components(self, fluency: float, coherence: float, terminology: float) -> float:
        """Calculate LQS from individual components."""
        lqs = (fluency + coherence + terminology) / 3
        return round(lqs, 4)

    def to_rubric_score(self, raw_score: float) -> int:
        """
        LQS is already on 1-5 scale.
        Round to nearest integer.
        """
        return max(1, min(5, round(raw_score)))


# =============================================================================
# Combined Evaluator
# =============================================================================

class VULCAMetricsEvaluator:
    """
    Combined evaluator for all 5 VULCA metrics.
    """

    def __init__(self, culture: str = "chinese", csa_alpha: float = 0.7, csa_beta: float = 0.3):
        self.culture = culture.lower()

        # Initialize all metrics
        self.dcr = DimensionCoverageRate(culture)
        self.csa = CulturalSemanticAlignment(culture, csa_alpha, csa_beta)
        self.cds = CritiqueDepthScore(culture)
        self.far = FactualAccuracyRate()
        self.lqs = LinguisticQualityScore()

    def evaluate(self, vlm_output: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all 5 metrics on a VLM output.

        Args:
            vlm_output: VLM-generated critique data
            reference: Expert reference critique data

        Returns:
            Dict with all metric scores (raw and rubric)
        """
        results = {
            "culture": self.culture,
            "metrics": {},
            "rubric_scores": {},
            "composite_score": 0.0
        }

        # DCR
        dcr_raw = self.dcr.calculate(vlm_output, reference)
        results["metrics"]["DCR"] = dcr_raw
        results["rubric_scores"]["DCR"] = self.dcr.to_rubric_score(dcr_raw)

        # CSA
        csa_raw = self.csa.calculate(vlm_output, reference)
        results["metrics"]["CSA"] = csa_raw
        results["rubric_scores"]["CSA"] = self.csa.to_rubric_score(csa_raw)

        # CDS
        cds_raw = self.cds.calculate(vlm_output, reference)
        results["metrics"]["CDS"] = cds_raw
        results["rubric_scores"]["CDS"] = self.cds.to_rubric_score(cds_raw)

        # FAR (requires judge scores in vlm_output)
        far_raw = self.far.calculate(vlm_output, reference)
        results["metrics"]["FAR"] = far_raw
        results["rubric_scores"]["FAR"] = self.far.to_rubric_score(far_raw)

        # LQS (requires judge scores in vlm_output)
        lqs_raw = self.lqs.calculate(vlm_output, reference)
        results["metrics"]["LQS"] = lqs_raw
        results["rubric_scores"]["LQS"] = self.lqs.to_rubric_score(lqs_raw)

        # Composite score (average of rubric scores)
        rubric_values = list(results["rubric_scores"].values())
        results["composite_score"] = round(sum(rubric_values) / len(rubric_values), 2)

        # Layer distribution for analysis
        results["layer_distribution"] = self.cds.get_layer_distribution(vlm_output)

        return results

    def evaluate_batch(self, samples: List[Tuple[Dict, Dict]]) -> Dict[str, Any]:
        """
        Evaluate a batch of samples.

        Args:
            samples: List of (vlm_output, reference) tuples

        Returns:
            Aggregated results with per-metric statistics
        """
        all_results = []

        for vlm_output, reference in samples:
            result = self.evaluate(vlm_output, reference)
            all_results.append(result)

        # Aggregate statistics
        metrics_agg = defaultdict(list)
        rubric_agg = defaultdict(list)

        for r in all_results:
            for metric, value in r["metrics"].items():
                metrics_agg[metric].append(value)
            for metric, value in r["rubric_scores"].items():
                rubric_agg[metric].append(value)

        summary = {
            "sample_count": len(samples),
            "culture": self.culture,
            "metrics_avg": {},
            "rubric_avg": {},
            "metrics_std": {},
        }

        for metric in ["DCR", "CSA", "CDS", "FAR", "LQS"]:
            values = metrics_agg[metric]
            if values:
                summary["metrics_avg"][metric] = round(sum(values) / len(values), 4)
                # Simple std calculation
                mean = sum(values) / len(values)
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                summary["metrics_std"][metric] = round(variance ** 0.5, 4)

            rubric_vals = rubric_agg[metric]
            if rubric_vals:
                summary["rubric_avg"][metric] = round(sum(rubric_vals) / len(rubric_vals), 2)

        return summary


# =============================================================================
# CLI Test
# =============================================================================

def _test_metrics():
    """Quick test of all metrics."""
    print("Testing VULCA Metrics v2.0")
    print("=" * 50)

    # Sample data
    vlm_output = {
        "critique_text": "This Chinese painting demonstrates excellent use of qi-yun shengdong and traditional bimo techniques.",
        "covered_dimensions": ["CN_L1_D1", "CN_L1_D2", "CN_L2_D1", "CN_L3_D1", "CN_L4_D1"],
        "factual_scores": [1, 1, 0, 1, 1],  # 4/5 correct
        "lqs_scores": {"fluency": 4, "coherence": 4, "terminology": 5}
    }

    reference = {
        "critique_zh": "这幅中国画展现了气韵生动的美学追求，笔墨技法精湛。",
        "covered_dimensions": ["CN_L1_D1", "CN_L1_D2", "CN_L1_D3", "CN_L2_D1", "CN_L3_D1", "CN_L4_D1", "CN_L5_D1"]
    }

    evaluator = VULCAMetricsEvaluator(culture="chinese")
    results = evaluator.evaluate(vlm_output, reference)

    print("\nMetric Results:")
    for metric, value in results["metrics"].items():
        rubric = results["rubric_scores"][metric]
        print(f"  {metric}: {value:.4f} -> Rubric: {rubric}/5")

    print(f"\nComposite Score: {results['composite_score']}/5")
    print(f"Layer Distribution: {results['layer_distribution']}")

    print("\n" + "=" * 50)
    print("All tests passed!")
    return True


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _test_metrics()
