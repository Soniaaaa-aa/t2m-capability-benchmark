"""
rule_base.py — shared plumbing for the rule-based evaluators (plan §3 ③)
=======================================================================

RuleEvaluator gives every new evaluator the same behaviour as the existing
BodySideEvaluator / TrajectoryEvaluator:

  * PROVISIONAL_THRESHOLDS  starting values from the plan (§4)
  * CURRENT_THRESHOLDS      the rule run_benchmark.ipynb uses; copy calibrated
                            values here, set CURRENT_THRESHOLD_STATUS = "frozen_v1.0"
  * thresholds=None         evidence only, pass_fail = None
  * decide(evidence, expected) is pure → thresholds can be re-fitted from saved
    evidence without re-running the motion analysis (validation.grid_search).

A subclass only writes calculate_evidence() and decide(). The existing
evaluators are not changed and do not depend on this file.
"""

from __future__ import annotations

import numpy as np

import events as E
from common import BaseEvaluator


class RuleEvaluator(BaseEvaluator):
    EVALUATOR_NAME = "RuleEvaluator"
    PROVISIONAL_THRESHOLDS: dict = {}
    CURRENT_THRESHOLDS: dict | None = None
    CURRENT_THRESHOLD_STATUS = "provisional_not_frozen"

    def __init__(self, thresholds=None, threshold_status=None, event_params=None):
        if thresholds is None:
            self.thresholds = None
            self.threshold_status = "not_set"
        else:
            self.thresholds = {**self.PROVISIONAL_THRESHOLDS, **thresholds}
            self.threshold_status = threshold_status or "custom"
        self.event_params = event_params

    @classmethod
    def for_benchmark(cls):
        return cls(thresholds=cls.CURRENT_THRESHOLDS, threshold_status=cls.CURRENT_THRESHOLD_STATUS)

    # ------------------------------------------------------------------ helpers
    def analyse(self, motion):
        return E.analyse(motion, self.event_params)

    @staticmethod
    def expected_of(requirement):
        return requirement.get("value", requirement.get("expected"))

    # ------------------------------------------------------------------ to implement
    def calculate_evidence(self, motion, requirement, evaluation_case):
        """Return a JSON-safe dict. Put "unsupported": "<why>" to return pass_fail=None."""
        raise NotImplementedError

    def decide(self, evidence, expected):
        """Return (pass_fail, reason) using self.thresholds. Must not look at the motion."""
        raise NotImplementedError

    def decide_from_evidence(self, evidence, expected):
        """Unified calibration hook (same name on BodySideEvaluator)."""
        return self.decide(evidence, expected)

    # ------------------------------------------------------------------ common interface
    def evaluate(self, motion, requirement, evaluation_case=None):
        expected = self.expected_of(requirement)
        motion = np.asarray(motion, dtype=np.float64)
        evidence = self.calculate_evidence(motion, requirement, evaluation_case or {})
        evidence.setdefault("judge", "rule")

        if evidence.get("unsupported"):
            pass_fail, reason = None, f"Not decidable by rule: {evidence['unsupported']}"
        elif self.thresholds is None:
            pass_fail, reason = None, "Thresholds not set — evidence only."
        else:
            pass_fail, reason = self.decide(evidence, expected)

        return {
            "requirement_id": requirement.get("requirement_id", requirement.get("id")),
            "requirement_type": requirement.get("type"),
            "expected_value": expected,
            "evaluator_name": self.EVALUATOR_NAME,
            "pass_fail": pass_fail,
            "score": evidence.get("score"),
            "threshold_status": self.threshold_status,
            "evidence": evidence,
            "reason": reason,
        }


def current():
    """
    RuleEvaluator bound to the CURRENTLY loaded common.BaseEvaluator.

    run_benchmark.ipynb does importlib.reload(common) after a git pull; that
    creates a new BaseEvaluator class, and a cached rule_base would still hold the
    old one (register_evaluator then rejects the subclass). Every eval_*.py
    therefore takes its base class through this function.
    """
    import importlib
    import sys

    import common as _common
    mod = sys.modules[__name__]
    if mod.BaseEvaluator is not _common.BaseEvaluator:
        mod = importlib.reload(mod)
    return mod.RuleEvaluator


def verdict(ok):
    return "PASS" if ok else "FAIL"


def events_json(evs, limit=10):
    return [e.to_dict() for e in list(evs)[:limit]]
