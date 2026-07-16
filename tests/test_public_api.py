import os
import unittest
from unittest.mock import patch

import vulca_framework
from vulca_framework import (
    ChecklistJudge,
    JudgeBackendError,
    TriLayerEvaluator,
    VULCAMetricsEvaluator,
)


class PublicApiTests(unittest.TestCase):
    def test_public_exports_are_importable(self):
        self.assertEqual(vulca_framework.__version__, "0.1.0")
        self.assertTrue(callable(TriLayerEvaluator))
        self.assertTrue(callable(VULCAMetricsEvaluator))

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            ChecklistJudge("unknown")

    def test_missing_provider_key_does_not_silently_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            judge = ChecklistJudge("claude")
            with self.assertRaises(JudgeBackendError):
                judge.evaluate(
                    vlm_critique="A description of a painting.",
                    artwork_info="Unknown painting",
                    mode="B",
                )

    def test_fallback_mode_is_explicit_in_result(self):
        evaluator = TriLayerEvaluator(
            culture="chinese",
            judge_model="fallback",
        )
        result = evaluator.evaluate(
            vlm_critique=(
                "The composition uses sparse ink brushwork and negative space "
                "to express literati ideals and Daoist emptiness."
            ),
            artwork_info="Chinese landscape painting",
            mode="B",
        )
        payload = result.to_dict()

        self.assertEqual(payload["judge_backend"], "fallback")
        self.assertEqual(
            payload["judge_model_name"],
            "rule-based-fallback-v0.1.0",
        )
        self.assertFalse(payload["calibrated"])
        self.assertEqual(
            payload["score_kind"],
            "experimental_uncalibrated_composite",
        )
        self.assertGreaterEqual(payload["final_score"], 0.0)
        self.assertLessEqual(payload["final_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
