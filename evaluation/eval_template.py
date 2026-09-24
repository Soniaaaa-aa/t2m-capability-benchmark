"""
eval_template.py — copy this file to start a new evaluator
===========================================================

1. Copy to evaluation/eval_<name>.py (e.g. eval_rotation.py for turn_direction).
   Any evaluation/eval_*.py file is imported automatically by
   run_benchmark.ipynb — no notebook edit is needed.
2. Rename the class, implement calculate_evidence / decide, set
   CURRENT_THRESHOLDS once you have calibrated them.
3. Uncomment register_evaluator(...) at the bottom. The name must match
   EVALUATION_CONFIG in evaluation/common.py (change common.py only via a
   reviewed PR).
4. Optional: your own calibration notebook evaluation/analysis/analysis_<name>.ipynb
   (copy analysis_body_side.ipynb) and tests/test_<name>.py.

This template file itself is skipped by the automatic import.
"""

import numpy as np

from common import BaseEvaluator, register_evaluator


class TemplateEvaluator(BaseEvaluator):
    """One line: what this evaluator decides."""

    EVALUATOR_NAME = "TemplateEvaluator"

    # Current rule used by run_benchmark.ipynb. None = evidence only
    # (pass_fail = None) until thresholds are calibrated.
    CURRENT_THRESHOLDS = None
    CURRENT_THRESHOLD_STATUS = "not_set"

    def __init__(self, thresholds=None, threshold_status=None):
        self.thresholds = thresholds
        self.threshold_status = threshold_status or ("not_set" if thresholds is None else "custom")

    @classmethod
    def for_benchmark(cls):
        """Instance used by run_benchmark.ipynb (the current rule)."""
        return cls(thresholds=cls.CURRENT_THRESHOLDS, threshold_status=cls.CURRENT_THRESHOLD_STATUS)

    def calculate_evidence(self, motion, requirement, evaluation_case):
        """Return a dict of numbers measured from motion [T, 22, 3]."""
        raise NotImplementedError

    def decide(self, evidence, expected):
        """Return (pass_fail, reason) using self.thresholds."""
        raise NotImplementedError

    def evaluate(self, motion, requirement, evaluation_case=None):
        expected = requirement.get("value", requirement.get("expected"))
        evidence = self.calculate_evidence(np.asarray(motion), requirement, evaluation_case)

        if self.thresholds is None:
            pass_fail, reason = None, "Thresholds not set — evidence only."
        else:
            pass_fail, reason = self.decide(evidence, expected)

        # Common result fields (keep these names so results can be merged).
        return {
            "requirement_id": requirement.get("requirement_id", requirement.get("id")),
            "requirement_type": requirement.get("type"),
            "expected_value": expected,
            "evaluator_name": self.EVALUATOR_NAME,
            "pass_fail": pass_fail,            # "PASS" / "FAIL" / None
            "score": None,                     # optional continuous score
            "threshold_status": self.threshold_status,
            "evidence": evidence,
            "reason": reason,
        }


# In your own eval_<name>.py, uncomment (name must match EVALUATION_CONFIG):
# register_evaluator("TemplateEvaluator", TemplateEvaluator)
_ = register_evaluator  # keeps linters quiet in the template
