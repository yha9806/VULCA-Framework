# VULCA Cultural Evaluation Framework

Research prototype for evaluating vision-language-model art critiques across
cultural traditions.

## Status

This repository is **pre-release software (`v0.1.0`)**, not a complete
reproduction package for the paper.

The public runtime currently provides:

- Tier I automated indicators;
- a Tier II ten-item checklist judge;
- an explicit rule-based backend for local smoke tests.

It does **not** include the paper's fitted Tier III sigmoid calibrator, the
450-sample human-annotation pool, or the full culture-specific keyword
resources. The runtime's `final_score` is therefore labelled
`experimental_uncalibrated_composite` and must not be reported as the paper's
calibrated score.

The file `vulca_framework/judge_calibration.py` is a retained legacy isotonic
experiment. It is not part of the public package API and does not implement the
paper's v3 sigmoid method.

## Install

From a clone of this repository:

```bash
python -m pip install .
```

External judge clients are optional:

```bash
python -m pip install '.[anthropic]'
python -m pip install '.[openai]'
```

Legacy calibration experiments require:

```bash
python -m pip install '.[calibration]'
```

## Local smoke test

The fallback backend is deterministic and makes no external API call. It is
only a plumbing test; it is not an LLM judge and does not reproduce research
scores.

```python
from vulca_framework import TriLayerEvaluator

evaluator = TriLayerEvaluator(
    culture="chinese",
    judge_model="fallback",
)

result = evaluator.evaluate(
    vlm_critique="The landscape uses sparse ink and extensive negative space.",
    artwork_info="Chinese landscape painting",
    mode="B",
)

print(result.final_score)
print(result.to_dict()["score_kind"])
print(result.to_dict()["judge_backend"])
```

Run the complete example with:

```bash
python examples/quick_start.py
```

## External judge backends

Set the relevant API key and select the backend explicitly:

```bash
export ANTHROPIC_API_KEY="..."
# or: export OPENAI_API_KEY="..."
```

```python
evaluator = TriLayerEvaluator(
    culture="chinese",
    judge_model="claude",
    judge_model_name="your-approved-provider-model-id",
)
```

Missing credentials, missing client libraries, and provider request failures
raise `JudgeBackendError`. They never silently change the evaluation to the
rule-based fallback.

Every result records both `judge_backend` and `judge_model_name`. Pass an
explicit provider model identifier when reproducing an experiment instead of
depending on the prototype default.

### Data boundary

The external backends send the VLM critique and, depending on the selected
mode, the expert critique or artwork metadata to the chosen provider. Do not
submit private, embargoed, customer, or rights-restricted material unless that
transfer is authorized and complies with the provider and project data policy.

## Evaluation modes

- **Mode A** requires `expert_critique` and `expert_dimensions`.
- **Mode B** accepts `artwork_info` without an expert critique.

Culture-specific Tier I metrics look for
`vulca_framework/dimension_keywords/<culture>.json`. These resources are not
included in the current prototype, so missing-resource warnings mean that
Tier I results are incomplete.

## Command line

Batch evaluation accepts JSON Lines input:

```bash
python evaluation/run_evaluation.py \
  --input vlm_critiques.jsonl \
  --output results/ \
  --culture chinese \
  --mode B \
  --judge fallback
```

Use `--judge claude` or `--judge gpt5` only after installing the matching
optional dependency and setting its API key.

## Paper relationship

The associated manuscript is:

> Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language
> Models, arXiv:2601.07984, v3. The arXiv record describes it as submitted to
> ACL 2026.

The paper reports a five-dimension Tier II judge and aggregate sigmoid
calibration on a 450-sample human-scored pool with a 155-sample held-out set.
Those fitted artifacts are not present here, so this repository must not be
used to claim exact paper reproduction.

```bibtex
@article{yu2026crosscultural,
  title={Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models},
  author={Yu, Haorui and Wen, Xuehang and Zhang, Fengrui and Yi, Qiufeng},
  journal={arXiv preprint arXiv:2601.07984},
  year={2026}
}
```

## Repository boundary

The canonical [VULCA Cultural Visual Benchmark](https://github.com/vulca-org/vulca-cultural-visual-benchmark)
owns public benchmark metadata, release manifests, dataset documentation, and
reference evaluation tooling. This repository owns reusable evaluation-method
code and experiments. Metric implementations here are experimental consumers
of the benchmark construct; they do not redefine the benchmark's canonical
schema or release scores.

## Repository contents

```text
vulca_framework/
  automated_metrics.py     Tier I experimental indicators
  checklist_judge.py       Tier II ten-item checklist
  trilayer_evaluator.py    Uncalibrated prototype orchestration
  judge_calibration.py     Legacy, non-integrated isotonic experiment
  metrics.py               Alternative metric implementations
evaluation/
  run_evaluation.py        Batch runner
examples/
  quick_start.py           Local fallback example
vulca_framework/dimension_keywords/
  README.md                Missing-resource disclosure
tests/
  test_public_api.py       Public API and safety smoke tests
```

## License

MIT. See [LICENSE](LICENSE).
