---
license: mit
tags:
- vlm-evaluation
- art-critique
- cultural-ai
- benchmark-framework
---

# VULCA-Framework

**VULCA-Framework** is a tri-tier evaluation framework for assessing Vision-Language Models (VLMs) on cross-cultural art critique tasks. It provides calibrated scoring with human-aligned metrics.

📄 **Paper**: Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models (ACL 2026)

🔗 **Related**: [VULCA-Bench Dataset](https://github.com/yha9806/VULCA-Bench) | [HuggingFace Dataset](https://huggingface.co/datasets/harryHURRY/vulca-bench)

## Framework Overview

### Tri-Tier Evaluation Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Tier I: Automated Metrics                │
│   DCR (Dimension Coverage) + CSA (Cultural Alignment) +    │
│   CDS (Critique Depth) + LQS (Linguistic Quality)          │
│                  (Risk indicators, not primary score)       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Tier II: LLM-as-Judge                      │
│   Single Primary Judge (Claude-Opus-4.5) with 5-dimension  │
│   rubric (Coverage, Alignment, Depth, Accuracy, Quality)   │
│                   (Primary evaluation score)                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                Tier III: Human Calibration                  │
│   Sigmoid calibration to human ratings (n=450)             │
│   Yields 1.7% MAE reduction on held-out set (n=155)       │
└─────────────────────────────────────────────────────────────┘
```

### 5-Dimension Evaluation Metrics

| Dimension | Full Name | Description | Scale |
|-----------|-----------|-------------|-------|
| **DCR** | Dimension Coverage Rate | Coverage of expert-annotated dimensions | 1-5 |
| **CSA** | Cultural Semantic Alignment | Cultural term overlap + BERTScore | 1-5 |
| **CDS** | Critique Depth Score | Weighted L1-L5 layer coverage | 1-5 |
| **FAR** | Factual Accuracy Rate | Factual correctness of statements | 1-5 |
| **LQS** | Linguistic Quality Score | Fluency + coherence + terminology | 1-5 |

### L1-L5 Cultural Understanding Layers

| Layer | Name | What it Measures |
|-------|------|------------------|
| **L1** | Visual Perception | Color, line, composition, visual elements |
| **L2** | Technical Analysis | Medium, technique, materials, craftsmanship |
| **L3** | Cultural Symbolism | Motifs, iconography, symbolic meanings |
| **L4** | Historical Context | Period, artist, provenance, art movements |
| **L5** | Philosophical Aesthetics | Aesthetic theory, cultural values, philosophy |

## Quick Start

### 1. Install Dependencies

```bash
pip install openai anthropic numpy scikit-learn
```

### 2. Set API Keys

```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # For Claude judge
export OPENAI_API_KEY="sk-..."         # Optional: for GPT judge
```

### 3. Run Evaluation

```python
from vulca_framework import TriLayerEvaluator

evaluator = TriLayerEvaluator(judge_model="claude-opus-4.5")

# Evaluate a VLM-generated critique
result = evaluator.evaluate(
    vlm_critique="The painting shows masterful brushwork...",
    expert_anchor="This Yuan dynasty landscape embodies...",
    culture="chinese"
)

print(f"Calibrated Score: {result['calibrated_score']:.2f}/5")
print(f"Layer Scores: {result['layer_scores']}")
print(f"Risk Indicators: {result['risk_indicators']}")
```

### 4. Batch Evaluation

```bash
python evaluation/run_evaluation.py \
    --input vlm_critiques.jsonl \
    --output results/ \
    --judge claude-opus-4.5
```

## Key Findings

1. **Automated metrics are unreliable proxies** for cultural depth (Tier I alone insufficient)
2. **Western samples score higher** than non-Western samples under current rubrics
3. **Cross-judge scale mismatch** makes dual-judge averaging unreliable (ICC = -0.50)
4. **Single calibrated judge** with explicit human calibration yields best results

## Recommended Judge Configuration

| Priority | Judge Model | Mean Score | Characteristics |
|----------|-------------|------------|-----------------|
| **Primary** | Claude-Opus-4.5 | 3.42 | Most discriminative, strictest |
| Backup | GPT-5 | 4.09 | Moderate strictness |
| Free tier | GLM-4.5V | - | Chinese-friendly |

## Repository Structure

```
VULCA-Framework/
├── vulca_framework/
│   ├── __init__.py
│   ├── trilayer_evaluator.py    # Main evaluation framework
│   ├── automated_metrics.py     # Tier I metrics (DCR, CSA, CDS, LQS)
│   ├── checklist_judge.py       # Tier II LLM-as-Judge
│   ├── judge_calibration.py     # Tier III isotonic calibration
│   └── metrics.py               # 5-dimension metrics
├── evaluation/
│   ├── run_evaluation.py        # Batch evaluation runner
│   ├── run_ablation.py          # Ablation experiments
│   └── analyze_results.py       # Results analysis
├── data/
│   ├── human_annotations.json   # 450 human-rated samples
│   └── calibration_params.json  # Pre-trained calibration
├── examples/
│   └── quick_start.py
└── README.md
```

## Citation

```bibtex
@inproceedings{yu2026vulcaframework,
  title={Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models},
  author={Yu, Haorui and Wen, Xuehang and Zhang, Fengrui and Yi, Qiufeng},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
```

## License

This framework is released under the MIT License.

## Acknowledgments

We thank the annotators who provided human ratings for calibration, and the cultural institutions that contributed to the VULCA-Bench dataset.
