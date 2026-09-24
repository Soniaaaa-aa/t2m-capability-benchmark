"""
eval_template.py — copy this file to start a new evaluator
===========================================================

1. Copy to eval_<name>.py (e.g. eval_rotation.py for turn_direction).
2. Rename the class, set REQUIREMENT_TYPES, implement calculate_evidence/decide.
3. Make sure common.EVALUATION_CONFIG (evaluation/common.py) maps your requirement type(s) to the
   name you register below (change common.py only via a reviewed PR).
4. Add `import eval_<name>` to STEP 0 of run_benchmark.ipynb and a
   STEP 1x section for your evaluator.
5. Add tests/test_<name>.py at the repository root (see tests/test_body_side.py
   and tests/synth_motion.py for synthetic test motions).

This template file is NOT imported by the notebook.
"""

import numpy as np

from common import BaseEvaluator, register_evaluator


class TemplateEvaluator(BaseEvaluator):
    """One line: what this evaluator decides."""

    EVALUATOR_NAME = "TemplateEvaluator"
    REQUIREMENT_TYPES = {"template"}          # e.g. {"turn_direction"}

    def __init__(self, thresholds=None, threshold_status=None):
        # Convention used by the other evaluators: thresholds=None means
        # "evidence only" (pass_fail = None) until they are calibrated.
        self.thresholds = thresholds
        self.threshold_status = threshold_status or ("not_set" if thresholds is None else "custom")

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


# Uncomment in your own eval_<name>.py:
# register_evaluator("TemplateEvaluator", TemplateEvaluator)
_ = register_evaluator  # keeps linters quiet in the template
