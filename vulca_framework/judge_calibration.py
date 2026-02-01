#!/usr/bin/env python3
"""
Judge-Human Calibration Module (A1)
===================================
Implements isotonic regression calibration to align judge scores with human expert scores.

Part of: strengthen-acl-submission-robustness OpenSpec change
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from scipy import stats
import pickle

# Dimensions to calibrate
DIMENSIONS = ['DCR', 'CSA', 'CDS', 'FAR', 'LQS']


@dataclass
class CalibrationMetrics:
    """Metrics for evaluating calibration quality."""
    mae_before: float
    mae_after: float
    mae_improvement: float
    spearman_before: float
    spearman_after: float
    spearman_improvement: float
    kappa_before: float
    kappa_after: float
    kappa_improvement: float


@dataclass
class DimensionCalibrationResult:
    """Result for a single dimension calibration."""
    dimension: str
    n_samples: int
    metrics: CalibrationMetrics
    cv_mae_mean: float
    cv_mae_std: float
    judge_score_range: Tuple[float, float]
    human_score_range: Tuple[float, float]


def load_human_annotations(filepath: str) -> Dict:
    """Load human annotations from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_scores(annotations: List[Dict]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Extract judge scores and human average scores from annotations.

    Returns:
        judge_scores: Dict mapping dimension -> array of judge scores
        human_scores: Dict mapping dimension -> array of human average scores
    """
    judge_scores = {dim: [] for dim in DIMENSIONS}
    human_scores = {dim: [] for dim in DIMENSIONS}

    for ann in annotations:
        judge = ann.get('layer2_judge_scores', {})
        human1 = ann.get('human_scores_annotator1', {})
        human2 = ann.get('human_scores_annotator2', {})

        for dim in DIMENSIONS:
            if dim in judge and dim in human1 and dim in human2:
                judge_scores[dim].append(judge[dim])
                # Human average
                human_avg = (human1[dim] + human2[dim]) / 2
                human_scores[dim].append(human_avg)

    # Convert to numpy arrays
    judge_scores = {dim: np.array(scores) for dim, scores in judge_scores.items()}
    human_scores = {dim: np.array(scores) for dim, scores in human_scores.items()}

    return judge_scores, human_scores


def compute_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute weighted Cohen's kappa for ordinal data.
    Discretizes continuous scores to nearest 0.5 for kappa calculation.
    """
    # Discretize to 0.5 increments (1, 1.5, 2, ..., 5)
    y_true_disc = np.round(y_true * 2) / 2
    y_pred_disc = np.round(y_pred * 2) / 2

    # Use quadratic weights for ordinal kappa
    try:
        from sklearn.metrics import cohen_kappa_score
        # Map to integers for sklearn
        labels = np.arange(1, 5.5, 0.5)
        y_true_int = [np.argmin(np.abs(labels - v)) for v in y_true_disc]
        y_pred_int = [np.argmin(np.abs(labels - v)) for v in y_pred_disc]
        return cohen_kappa_score(y_true_int, y_pred_int, weights='quadratic')
    except:
        return 0.0


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Spearman correlation."""
    rho, _ = stats.spearmanr(y_true, y_pred)
    return rho if not np.isnan(rho) else 0.0


def calibrate_dimension(
    judge_scores: np.ndarray,
    human_scores: np.ndarray,
    n_folds: int = 5
) -> Tuple[IsotonicRegression, DimensionCalibrationResult, str]:
    """
    Calibrate a single dimension using isotonic regression with cross-validation.

    Args:
        judge_scores: Array of judge scores
        human_scores: Array of human average scores
        n_folds: Number of CV folds

    Returns:
        model: Fitted IsotonicRegression model (on full data)
        result: Calibration result with metrics
        dimension: Dimension name (placeholder, set by caller)
    """
    n_samples = len(judge_scores)

    # Before calibration metrics (full dataset)
    mae_before = compute_mae(human_scores, judge_scores)
    spearman_before = compute_spearman(human_scores, judge_scores)
    kappa_before = compute_weighted_kappa(human_scores, judge_scores)

    # Cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_maes = []
    cv_calibrated_scores = np.zeros_like(judge_scores)

    for train_idx, test_idx in kf.split(judge_scores):
        # Fit on train
        model_cv = IsotonicRegression(y_min=1.0, y_max=5.0, out_of_bounds='clip')
        model_cv.fit(judge_scores[train_idx], human_scores[train_idx])

        # Predict on test
        calibrated = model_cv.predict(judge_scores[test_idx])
        cv_calibrated_scores[test_idx] = calibrated

        # Compute test MAE
        cv_mae = compute_mae(human_scores[test_idx], calibrated)
        cv_maes.append(cv_mae)

    # After calibration metrics (using CV predictions)
    mae_after = compute_mae(human_scores, cv_calibrated_scores)
    spearman_after = compute_spearman(human_scores, cv_calibrated_scores)
    kappa_after = compute_weighted_kappa(human_scores, cv_calibrated_scores)

    # Fit final model on all data
    final_model = IsotonicRegression(y_min=1.0, y_max=5.0, out_of_bounds='clip')
    final_model.fit(judge_scores, human_scores)

    metrics = CalibrationMetrics(
        mae_before=mae_before,
        mae_after=mae_after,
        mae_improvement=mae_before - mae_after,
        spearman_before=spearman_before,
        spearman_after=spearman_after,
        spearman_improvement=spearman_after - spearman_before,
        kappa_before=kappa_before,
        kappa_after=kappa_after,
        kappa_improvement=kappa_after - kappa_before
    )

    result = DimensionCalibrationResult(
        dimension="",  # Set by caller
        n_samples=n_samples,
        metrics=metrics,
        cv_mae_mean=np.mean(cv_maes),
        cv_mae_std=np.std(cv_maes),
        judge_score_range=(float(judge_scores.min()), float(judge_scores.max())),
        human_score_range=(float(human_scores.min()), float(human_scores.max()))
    )

    return final_model, result


def run_calibration_experiment(
    annotations_path: str,
    output_dir: str,
    n_folds: int = 5
) -> Dict:
    """
    Run the full calibration experiment.

    Args:
        annotations_path: Path to human annotations JSON
        output_dir: Directory to save results
        n_folds: Number of CV folds

    Returns:
        Full calibration report as dict
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    data = load_human_annotations(annotations_path)
    annotations = data['annotations']

    print(f"Loaded {len(annotations)} annotations")

    # Extract scores
    judge_scores, human_scores = extract_scores(annotations)

    # Calibrate each dimension
    models = {}
    results = {}

    for dim in DIMENSIONS:
        print(f"\nCalibrating {dim}...")
        js = judge_scores[dim]
        hs = human_scores[dim]

        if len(js) == 0:
            print(f"  WARNING: No data for {dim}")
            continue

        model, result = calibrate_dimension(js, hs, n_folds)
        result.dimension = dim

        models[dim] = model
        results[dim] = result

        print(f"  MAE: {result.metrics.mae_before:.3f} → {result.metrics.mae_after:.3f} "
              f"(Δ={result.metrics.mae_improvement:+.3f})")
        print(f"  Spearman: {result.metrics.spearman_before:.3f} → {result.metrics.spearman_after:.3f} "
              f"(Δ={result.metrics.spearman_improvement:+.3f})")
        print(f"  κw: {result.metrics.kappa_before:.3f} → {result.metrics.kappa_after:.3f} "
              f"(Δ={result.metrics.kappa_improvement:+.3f})")

    # Compute overall metrics
    overall_mae_before = np.mean([r.metrics.mae_before for r in results.values()])
    overall_mae_after = np.mean([r.metrics.mae_after for r in results.values()])
    overall_spearman_before = np.mean([r.metrics.spearman_before for r in results.values()])
    overall_spearman_after = np.mean([r.metrics.spearman_after for r in results.values()])
    overall_kappa_before = np.mean([r.metrics.kappa_before for r in results.values()])
    overall_kappa_after = np.mean([r.metrics.kappa_after for r in results.values()])

    # Count improvements
    dims_mae_improved = sum(1 for r in results.values() if r.metrics.mae_improvement > 0)
    dims_spearman_improved = sum(1 for r in results.values() if r.metrics.spearman_improvement > 0)
    dims_kappa_improved = sum(1 for r in results.values() if r.metrics.kappa_improvement > 0)

    # Build report
    report = {
        "experiment": "Judge-Human Calibration (A1)",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_samples": len(annotations),
            "n_folds": n_folds,
            "calibration_method": "IsotonicRegression",
            "score_bounds": [1.0, 5.0]
        },
        "per_dimension_results": {
            dim: {
                "n_samples": r.n_samples,
                "mae_before": round(r.metrics.mae_before, 4),
                "mae_after": round(r.metrics.mae_after, 4),
                "mae_improvement": round(r.metrics.mae_improvement, 4),
                "spearman_before": round(r.metrics.spearman_before, 4),
                "spearman_after": round(r.metrics.spearman_after, 4),
                "spearman_improvement": round(r.metrics.spearman_improvement, 4),
                "kappa_before": round(r.metrics.kappa_before, 4),
                "kappa_after": round(r.metrics.kappa_after, 4),
                "kappa_improvement": round(r.metrics.kappa_improvement, 4),
                "cv_mae_mean": round(r.cv_mae_mean, 4),
                "cv_mae_std": round(r.cv_mae_std, 4)
            }
            for dim, r in results.items()
        },
        "overall_summary": {
            "mae_before": round(overall_mae_before, 4),
            "mae_after": round(overall_mae_after, 4),
            "mae_improvement": round(overall_mae_before - overall_mae_after, 4),
            "spearman_before": round(overall_spearman_before, 4),
            "spearman_after": round(overall_spearman_after, 4),
            "spearman_improvement": round(overall_spearman_after - overall_spearman_before, 4),
            "kappa_before": round(overall_kappa_before, 4),
            "kappa_after": round(overall_kappa_after, 4),
            "kappa_improvement": round(overall_kappa_after - overall_kappa_before, 4),
            "dims_mae_improved": dims_mae_improved,
            "dims_spearman_improved": dims_spearman_improved,
            "dims_kappa_improved": dims_kappa_improved,
            "total_dimensions": len(results)
        },
        "conclusion": {
            "calibration_effective": dims_mae_improved >= 3,
            "interpretation": (
                f"Calibration improved MAE in {dims_mae_improved}/5 dimensions. "
                f"Average MAE reduced from {overall_mae_before:.3f} to {overall_mae_after:.3f}. "
                + ("This demonstrates calibration can partially align judge scores with human experts, "
                   "but fundamental disagreement patterns remain." if dims_mae_improved >= 3 else
                   "Limited improvement suggests intrinsic scale differences between judge and human experts.")
            )
        }
    }

    # Save report
    report_path = output_path / "calibration_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_path}")

    # Save models
    models_path = output_path / "calibration_models.pkl"
    with open(models_path, 'wb') as f:
        pickle.dump(models, f)
    print(f"Models saved to: {models_path}")

    # Generate visualization data
    viz_data = {
        "dimensions": list(results.keys()),
        "before": {
            "mae": [r.metrics.mae_before for r in results.values()],
            "spearman": [r.metrics.spearman_before for r in results.values()],
            "kappa": [r.metrics.kappa_before for r in results.values()]
        },
        "after": {
            "mae": [r.metrics.mae_after for r in results.values()],
            "spearman": [r.metrics.spearman_after for r in results.values()],
            "kappa": [r.metrics.kappa_after for r in results.values()]
        }
    }

    # Try to generate violin plot
    try:
        generate_violin_plot(judge_scores, human_scores, models, output_path)
    except ImportError:
        print("matplotlib not available, skipping violin plot")

    return report


def generate_violin_plot(
    judge_scores: Dict[str, np.ndarray],
    human_scores: Dict[str, np.ndarray],
    models: Dict[str, IsotonicRegression],
    output_path: Path
):
    """Generate violin plot comparing before/after calibration distributions."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))

    for idx, dim in enumerate(DIMENSIONS):
        ax = axes[idx]
        js = judge_scores[dim]
        hs = human_scores[dim]

        if dim in models:
            calibrated = models[dim].predict(js)
        else:
            calibrated = js

        # Compute residuals
        residuals_before = hs - js
        residuals_after = hs - calibrated

        # Violin plot
        parts = ax.violinplot([residuals_before, residuals_after], positions=[1, 2],
                              showmeans=True, showmedians=True)

        # Styling
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
        parts['bodies'][0].set_facecolor('coral')
        parts['bodies'][1].set_facecolor('steelblue')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Before', 'After'])
        ax.set_title(dim)
        ax.set_ylabel('Residual (Human - Judge)')
        ax.set_ylim(-2.5, 2.5)

    plt.suptitle('Judge-Human Calibration: Residual Distribution by Dimension', fontsize=12)
    plt.tight_layout()

    plot_path = output_path / "calibration_violin.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Violin plot saved to: {plot_path}")


def apply_calibration(
    scores: Dict[str, float],
    models_path: str
) -> Dict[str, float]:
    """
    Apply saved calibration models to new scores.

    Args:
        scores: Dict of dimension -> score
        models_path: Path to saved models pickle

    Returns:
        Calibrated scores dict
    """
    with open(models_path, 'rb') as f:
        models = pickle.load(f)

    calibrated = {}
    for dim, score in scores.items():
        if dim in models:
            calibrated[dim] = float(models[dim].predict([[score]])[0])
        else:
            calibrated[dim] = score

    return calibrated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Judge-Human Calibration Experiment")
    parser.add_argument("--annotations", type=str,
                        default="checkpoints/p11_human_annotations_150.json",
                        help="Path to human annotations JSON")
    parser.add_argument("--output", type=str,
                        default="results/calibration",
                        help="Output directory for results")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of CV folds")

    args = parser.parse_args()

    report = run_calibration_experiment(
        annotations_path=args.annotations,
        output_dir=args.output,
        n_folds=args.folds
    )

    print("\n" + "="*60)
    print("CALIBRATION EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Conclusion: {report['conclusion']['interpretation']}")
