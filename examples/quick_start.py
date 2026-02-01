"""
VULCA-Framework Quick Start Example
Demonstrates the tri-tier evaluation framework.
"""

import sys
sys.path.append('..')

from vulca_framework import TriLayerEvaluator


def main():
    print("=== VULCA-Framework Quick Start ===\n")

    # Sample VLM-generated critique
    vlm_critique = """
    This Yuan dynasty landscape painting demonstrates masterful brushwork
    in the literati tradition. The composition follows the "level distance"
    perspective, with misty mountains receding into the background. The
    artist employs dry-brush techniques characteristic of Ni Zan's style,
    with sparse, economical strokes that create a sense of noble solitude.
    The empty space in the painting reflects the Daoist concept of wu (emptiness),
    inviting contemplation and spiritual engagement.
    """

    # Expert reference critique
    expert_critique = """
    此作为元代文人山水画精品，运用平远构图法，展现江南水乡意境。
    画家采用渴笔皴法，笔墨苍润而不失清逸，深得倪瓒高士风范。
    构图疏朗，意境空灵，大量留白体现了道家"无"之哲学思想。
    山石树木皆以写意笔法出之，气韵生动，格调高雅。
    """

    # Expert-annotated dimensions covered
    expert_dimensions = [
        "CN_L1_D1", "CN_L1_D2", "CN_L1_D3",  # Visual perception
        "CN_L2_D1", "CN_L2_D2", "CN_L2_D3",  # Technical analysis
        "CN_L3_D1", "CN_L3_D2",               # Cultural symbolism
        "CN_L4_D1", "CN_L4_D2",               # Historical context
        "CN_L5_D1", "CN_L5_D2", "CN_L5_D3",  # Philosophical aesthetics
    ]

    # Initialize evaluator
    # Note: Use 'fallback' for testing without API keys
    evaluator = TriLayerEvaluator(
        culture='chinese',
        judge_model='fallback'  # Change to 'claude' for production
    )

    # Mode A: Reference-based evaluation
    print("--- Mode A: Reference-Based Evaluation ---")
    result = evaluator.evaluate(
        vlm_critique=vlm_critique,
        expert_critique=expert_critique,
        expert_dimensions=expert_dimensions,
        mode='A'
    )

    print(result.summary())
    print()

    # Show structured output
    print("--- Structured Result ---")
    result_dict = result.to_dict()
    print(f"Final Score: {result_dict['final_score']:.1%}")
    print(f"Layer 1 (Automated): {result_dict['layer1_score']:.1%}")
    print(f"Layer 2 (Checklist): {result_dict['layer2_score']:.1%}")
    print()

    # Mode B: Reference-free evaluation (no expert needed)
    print("--- Mode B: Reference-Free Evaluation ---")
    result_b = evaluator.evaluate(
        vlm_critique=vlm_critique,
        artwork_info="Title: Landscape in Yuan Style, Artist: Anonymous, Period: Yuan Dynasty (1271-1368)",
        mode='B'
    )
    print(f"Final Score: {result_b.final_score:.1%}")


if __name__ == "__main__":
    main()
