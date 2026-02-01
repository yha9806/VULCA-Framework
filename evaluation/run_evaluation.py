"""
VULCA-Framework Batch Evaluation Runner
Evaluates VLM-generated critiques using the tri-tier framework.

Usage:
    python run_evaluation.py --input critiques.jsonl --output results/ --judge claude
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from vulca_framework import TriLayerEvaluator


def load_samples(filepath: str):
    """Load samples from JSONL file."""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def save_results(results, output_dir: str, model_name: str = "evaluation"):
    """Save evaluation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"{model_name}_results_{timestamp}.json"

    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "total_samples": len(results),
            "model": model_name
        },
        "results": [r.to_dict() for r in results]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="VULCA-Framework Batch Evaluation")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file with VLM critiques")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    parser.add_argument("--judge", type=str, default="claude", choices=["claude", "gpt5", "fallback"],
                        help="Judge model for Tier II")
    parser.add_argument("--culture", type=str, default="chinese",
                        help="Culture for evaluation (chinese, western, japanese, korean, islamic, indian)")
    parser.add_argument("--mode", type=str, default="A", choices=["A", "B"],
                        help="Evaluation mode: A (reference-based) or B (reference-free)")

    args = parser.parse_args()

    print(f"=== VULCA-Framework Batch Evaluation ===")
    print(f"Input: {args.input}")
    print(f"Judge: {args.judge}")
    print(f"Culture: {args.culture}")
    print(f"Mode: {args.mode}")
    print()

    # Load samples
    samples = load_samples(args.input)
    print(f"Loaded {len(samples)} samples")

    # Initialize evaluator
    evaluator = TriLayerEvaluator(
        culture=args.culture,
        judge_model=args.judge
    )

    # Run evaluation
    def progress_callback(current, total):
        print(f"\rEvaluating: {current}/{total}", end="", flush=True)

    results = evaluator.evaluate_batch(
        samples=samples,
        mode=args.mode,
        progress_callback=progress_callback
    )
    print()

    # Calculate statistics
    stats = evaluator.get_batch_statistics(results)
    print(f"\n=== Statistics ===")
    print(f"Samples: {stats['count']}")
    print(f"Final Score: {stats['final_score']['mean']:.1%} (min: {stats['final_score']['min']:.1%}, max: {stats['final_score']['max']:.1%})")
    print(f"Layer 1: {stats['layer1_score']['mean']:.1%}")
    print(f"Layer 2: {stats['layer2_score']['mean']:.1%}")

    # Save results
    save_results(results, args.output, f"{args.culture}_{args.judge}")


if __name__ == "__main__":
    main()
