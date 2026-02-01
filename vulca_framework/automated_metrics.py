"""
automated_metrics.py - Layer 1 Automated Metrics for VULCA Evaluation Framework v2.0

Implements 6 automated metrics for Tri-Layer Pyramid evaluation:
- DMR: Dimension Match Rate (Mode A - Reference-Based)
- DCI: Dimension Coverage Index (Mode B - Reference-Free)
- LDS: Length Deviation Score (Mode A)
- LAS: Length Adequacy Score (Mode B)
- CTP: Cultural Term Presence (Both modes)
- LAR: Layer Analysis Richness (Both modes)

Usage:
    from scripts.evaluation.automated_metrics import AutomatedMetrics

    metrics = AutomatedMetrics(culture='chinese')

    # Mode A (with expert reference)
    scores_a = metrics.compute_all(vlm_critique, expert_critique, expert_dimensions, mode='A')

    # Mode B (reference-free)
    scores_b = metrics.compute_all(vlm_critique, mode='B')

Author: Claude Code
Version: 1.0 (2025-11-29)
"""

import json
import re
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import Counter


class AutomatedMetrics:
    """Layer 1 automated metrics calculator for VULCA evaluation."""

    # Expected critique length ranges by culture (Chinese characters or English words)
    CULTURE_LENGTH_RANGES = {
        'chinese': {'zh': (200, 800), 'en': (100, 400)},
        'western': {'zh': (150, 600), 'en': (100, 350)},
        'japanese': {'zh': (200, 700), 'en': (100, 350)},
        'korean': {'zh': (200, 700), 'en': (100, 350)},
        'islamic': {'zh': (250, 600), 'en': (120, 350)},
        'indian': {'zh': (300, 800), 'en': (150, 400)},
    }

    # Layer prefixes by culture
    CULTURE_PREFIXES = {
        'chinese': 'CN_',
        'western': 'WE_',
        'japanese': 'JP_',
        'korean': 'KR_',
        'islamic': 'IS_',
        'indian': 'IN_',
    }

    def __init__(self, culture: str, keywords_dir: str = None):
        """
        Initialize metrics calculator.

        Args:
            culture: One of 'chinese', 'western', 'japanese', 'korean', 'islamic', 'indian'
            keywords_dir: Path to dimension keywords directory.
                         Defaults to experiments/dimension_keywords/
        """
        self.culture = culture.lower()
        if self.culture not in self.CULTURE_PREFIXES:
            raise ValueError(f"Unknown culture: {culture}. Must be one of {list(self.CULTURE_PREFIXES.keys())}")

        self.prefix = self.CULTURE_PREFIXES[self.culture]
        self.length_ranges = self.CULTURE_LENGTH_RANGES.get(self.culture, {'zh': (200, 600), 'en': (100, 300)})

        # Load keywords
        if keywords_dir is None:
            keywords_dir = Path(__file__).parent.parent.parent / 'dimension_keywords'
        else:
            keywords_dir = Path(keywords_dir)

        self.keywords_zh: Set[str] = set()
        self.keywords_en: Set[str] = set()
        self.dimension_keywords: Dict[str, Dict] = {}

        self._load_keywords(keywords_dir)

    def _load_keywords(self, keywords_dir: Path):
        """Load keywords from JSON file."""
        keywords_file = keywords_dir / f"{self.culture}.json"

        if keywords_file.exists():
            with open(keywords_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.keywords_zh = set(data.get('all_keywords_zh', []))
            self.keywords_en = set(data.get('all_keywords_en', []))

            # Extract dimension-level keywords
            for layer_id, layer_data in data.get('layers', {}).items():
                for dim_id, dim_data in layer_data.get('dimensions', {}).items():
                    self.dimension_keywords[dim_id] = {
                        'zh': set(dim_data.get('keywords_zh', [])),
                        'en': set(dim_data.get('keywords_en', []))
                    }
        else:
            print(f"Warning: Keywords file not found: {keywords_file}")

    # =========================================================================
    # DMR: Dimension Match Rate (Mode A - Reference-Based)
    # =========================================================================

    def compute_dmr(self, vlm_critique: str, expert_dimensions: List[str]) -> float:
        """
        Compute Dimension Match Rate - how well VLM covers expert dimensions.

        Args:
            vlm_critique: VLM-generated critique text
            expert_dimensions: List of dimension IDs from expert critique (e.g., ['CN_L1_D1', 'CN_L2_D2'])

        Returns:
            Score between 0 and 1
        """
        if not expert_dimensions:
            return 0.0

        vlm_text = vlm_critique.lower() if vlm_critique else ""
        matched_dimensions = 0

        for dim_id in expert_dimensions:
            if dim_id in self.dimension_keywords:
                dim_kw = self.dimension_keywords[dim_id]
                # Check if any keyword matches
                zh_match = any(kw in vlm_critique for kw in dim_kw.get('zh', []))
                en_match = any(kw.lower() in vlm_text for kw in dim_kw.get('en', []))

                if zh_match or en_match:
                    matched_dimensions += 1

        return matched_dimensions / len(expert_dimensions)

    # =========================================================================
    # DCI: Dimension Coverage Index (Mode B - Reference-Free)
    # =========================================================================

    def compute_dci(self, vlm_critique: str) -> float:
        """
        Compute Dimension Coverage Index - how many culture dimensions are covered.

        Args:
            vlm_critique: VLM-generated critique text

        Returns:
            Score between 0 and 1
        """
        if not vlm_critique:
            return 0.0

        vlm_text = vlm_critique.lower()
        covered_dimensions = set()

        for dim_id, dim_kw in self.dimension_keywords.items():
            # Check if any keyword matches
            zh_match = any(kw in vlm_critique for kw in dim_kw.get('zh', []))
            en_match = any(kw.lower() in vlm_text for kw in dim_kw.get('en', []))

            if zh_match or en_match:
                covered_dimensions.add(dim_id)

        # Total dimensions for culture (from dimension_keywords)
        total_dims = len(self.dimension_keywords)
        if total_dims == 0:
            total_dims = 30  # Default fallback

        return len(covered_dimensions) / total_dims

    # =========================================================================
    # LDS: Length Deviation Score (Mode A - Reference-Based)
    # =========================================================================

    def compute_lds(self, vlm_critique: str, expert_critique: str) -> float:
        """
        Compute Length Deviation Score - how close VLM length is to expert length.

        Args:
            vlm_critique: VLM-generated critique text
            expert_critique: Expert reference critique text

        Returns:
            Score between 0 and 1 (1 = same length, lower for deviation)
        """
        vlm_len = self._get_text_length(vlm_critique)
        expert_len = self._get_text_length(expert_critique)

        if expert_len == 0:
            return 0.0

        # Calculate deviation ratio
        ratio = vlm_len / expert_len

        # Score based on deviation (0.5-2.0 range gets partial credit)
        if 0.8 <= ratio <= 1.2:
            return 1.0  # Perfect range
        elif 0.5 <= ratio <= 2.0:
            # Linear decay outside 0.8-1.2
            if ratio < 0.8:
                return 0.5 + (ratio - 0.5) / 0.6  # 0.5 at ratio=0.5, 1.0 at ratio=0.8
            else:
                return 0.5 + (2.0 - ratio) / 1.6  # 1.0 at ratio=1.2, 0.5 at ratio=2.0
        else:
            return max(0.0, 0.5 - abs(1 - ratio) * 0.25)  # Rapid decay outside

    # =========================================================================
    # LAS: Length Adequacy Score (Mode B - Reference-Free)
    # =========================================================================

    def compute_las(self, vlm_critique: str) -> float:
        """
        Compute Length Adequacy Score - whether VLM meets culture-specific length expectations.

        Args:
            vlm_critique: VLM-generated critique text

        Returns:
            Score between 0 and 1
        """
        text_len = self._get_text_length(vlm_critique)

        # Detect language and get appropriate range
        if self._is_chinese(vlm_critique):
            min_len, max_len = self.length_ranges.get('zh', (200, 600))
        else:
            min_len, max_len = self.length_ranges.get('en', (100, 300))

        # Score based on range
        if min_len <= text_len <= max_len:
            return 1.0
        elif text_len < min_len:
            return max(0.0, text_len / min_len)
        else:
            # Slight penalty for being too long
            over_ratio = (text_len - max_len) / max_len
            return max(0.5, 1.0 - over_ratio * 0.3)

    # =========================================================================
    # CTP: Cultural Term Presence (Both Modes)
    # =========================================================================

    def compute_ctp(self, vlm_critique: str) -> float:
        """
        Compute Cultural Term Presence - ratio of cultural keywords found.

        Args:
            vlm_critique: VLM-generated critique text

        Returns:
            Score between 0 and 1
        """
        if not vlm_critique:
            return 0.0

        vlm_text = vlm_critique.lower()

        # Count matched keywords
        zh_matches = sum(1 for kw in self.keywords_zh if kw in vlm_critique)
        en_matches = sum(1 for kw in self.keywords_en if kw.lower() in vlm_text)

        total_matches = zh_matches + en_matches

        # Normalize - expect around 10-30 keyword matches for good coverage
        # Score caps at 1.0 for 20+ matches
        if total_matches >= 20:
            return 1.0
        elif total_matches >= 10:
            return 0.5 + (total_matches - 10) / 20
        else:
            return total_matches / 20

    # =========================================================================
    # LAR: Layer Analysis Richness (Both Modes)
    # =========================================================================

    def compute_lar(self, vlm_critique: str) -> float:
        """
        Compute Layer Analysis Richness - how evenly distributed are covered layers.

        Uses entropy-based measure of layer coverage distribution.

        Args:
            vlm_critique: VLM-generated critique text

        Returns:
            Score between 0 and 1
        """
        if not vlm_critique:
            return 0.0

        vlm_text = vlm_critique.lower()
        layer_counts = Counter()

        # Count matches per layer
        for dim_id, dim_kw in self.dimension_keywords.items():
            # Extract layer from dimension ID (e.g., CN_L1_D1 -> L1)
            parts = dim_id.split('_')
            if len(parts) >= 2:
                layer = parts[1]  # L1, L2, etc.

                zh_match = any(kw in vlm_critique for kw in dim_kw.get('zh', []))
                en_match = any(kw.lower() in vlm_text for kw in dim_kw.get('en', []))

                if zh_match or en_match:
                    layer_counts[layer] += 1

        if not layer_counts:
            return 0.0

        # Calculate entropy
        total = sum(layer_counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in layer_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # Normalize by max entropy (log2(5) for 5 layers)
        max_entropy = math.log2(5)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Also factor in number of layers covered
        layers_covered = len(layer_counts)
        coverage_bonus = layers_covered / 5

        # Combined score: 60% entropy + 40% coverage
        return 0.6 * normalized_entropy + 0.4 * coverage_bonus

    # =========================================================================
    # Combined Metrics
    # =========================================================================

    def compute_all(
        self,
        vlm_critique: str,
        expert_critique: str = None,
        expert_dimensions: List[str] = None,
        mode: str = 'A'
    ) -> Dict[str, float]:
        """
        Compute all Layer 1 metrics.

        Args:
            vlm_critique: VLM-generated critique text
            expert_critique: Expert reference critique (required for Mode A)
            expert_dimensions: Expert covered dimensions (required for Mode A)
            mode: 'A' for reference-based, 'B' for reference-free

        Returns:
            Dictionary with all metric scores
        """
        mode = mode.upper()

        if mode == 'A':
            if expert_critique is None or expert_dimensions is None:
                raise ValueError("Mode A requires expert_critique and expert_dimensions")

            return {
                'DMR': self.compute_dmr(vlm_critique, expert_dimensions),
                'LDS': self.compute_lds(vlm_critique, expert_critique),
                'CTP': self.compute_ctp(vlm_critique),
                'LAR': self.compute_lar(vlm_critique),
                'mode': 'A'
            }
        else:  # Mode B
            return {
                'DCI': self.compute_dci(vlm_critique),
                'LAS': self.compute_las(vlm_critique),
                'CTP': self.compute_ctp(vlm_critique),
                'LAR': self.compute_lar(vlm_critique),
                'mode': 'B'
            }

    def compute_layer1_score(
        self,
        vlm_critique: str,
        expert_critique: str = None,
        expert_dimensions: List[str] = None,
        mode: str = 'A',
        weights: Dict[str, float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute weighted Layer 1 total score.

        Args:
            vlm_critique: VLM-generated critique
            expert_critique: Expert reference (Mode A)
            expert_dimensions: Expert dimensions (Mode A)
            mode: 'A' or 'B'
            weights: Custom weights (default: equal weights)

        Returns:
            Tuple of (total_score, individual_scores)
        """
        scores = self.compute_all(vlm_critique, expert_critique, expert_dimensions, mode)

        # Default weights
        if weights is None:
            if mode.upper() == 'A':
                weights = {'DMR': 0.35, 'LDS': 0.15, 'CTP': 0.25, 'LAR': 0.25}
            else:
                weights = {'DCI': 0.35, 'LAS': 0.15, 'CTP': 0.25, 'LAR': 0.25}

        # Calculate weighted sum
        total = 0.0
        for metric, weight in weights.items():
            if metric in scores:
                total += scores[metric] * weight

        return total, scores

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_text_length(self, text: str) -> int:
        """Get text length (characters for Chinese, words for English)."""
        if not text:
            return 0

        if self._is_chinese(text):
            # Count Chinese characters
            return len(re.findall(r'[\u4e00-\u9fff]', text))
        else:
            # Count words
            return len(text.split())

    def _is_chinese(self, text: str) -> bool:
        """Check if text is primarily Chinese."""
        if not text:
            return False
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        return chinese_chars > len(text) * 0.3


# =============================================================================
# Convenience Functions
# =============================================================================

def compute_layer1_metrics(
    vlm_critique: str,
    culture: str,
    expert_critique: str = None,
    expert_dimensions: List[str] = None,
    mode: str = 'A',
    keywords_dir: str = None
) -> Tuple[float, Dict[str, float]]:
    """
    Convenience function to compute Layer 1 metrics.

    Args:
        vlm_critique: VLM-generated critique
        culture: Culture name
        expert_critique: Expert reference (Mode A)
        expert_dimensions: Expert dimensions (Mode A)
        mode: 'A' or 'B'
        keywords_dir: Path to keywords directory

    Returns:
        Tuple of (total_score, individual_scores)
    """
    metrics = AutomatedMetrics(culture, keywords_dir)
    return metrics.compute_layer1_score(
        vlm_critique, expert_critique, expert_dimensions, mode
    )


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    # Quick test
    test_critique_zh = """
    此幅山水画以浓墨淡彩描绘江南水乡风光，构图采用三远法中的平远视角，
    展现了文人画的典型意境。笔法流畅，墨色层次分明，气韵生动，
    体现了明代吴门画派的艺术特色。画面虚实相生，意境深远，
    充分展现了中国传统绘画的审美追求。
    """

    test_dimensions = ['CN_L1_D1', 'CN_L1_D2', 'CN_L1_D3', 'CN_L2_D1', 'CN_L2_D2',
                       'CN_L3_D1', 'CN_L4_D1', 'CN_L5_D2', 'CN_L5_D3']

    metrics = AutomatedMetrics('chinese')

    print("=== Layer 1 Automated Metrics Test ===\n")

    # Mode A
    score_a, scores_a = metrics.compute_layer1_score(
        test_critique_zh,
        test_critique_zh,  # Using same as expert for test
        test_dimensions,
        mode='A'
    )
    print(f"Mode A Total Score: {score_a:.3f}")
    for k, v in scores_a.items():
        if k != 'mode':
            print(f"  {k}: {v:.3f}")

    print()

    # Mode B
    score_b, scores_b = metrics.compute_layer1_score(
        test_critique_zh,
        mode='B'
    )
    print(f"Mode B Total Score: {score_b:.3f}")
    for k, v in scores_b.items():
        if k != 'mode':
            print(f"  {k}: {v:.3f}")
