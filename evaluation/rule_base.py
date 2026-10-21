"""
Shared base class for the rule-based evaluators.

RuleEvaluator gives every evaluator built on top of it the same behaviour:
PROVISIONAL_THRESHOLDS are the starting values, CURRENT_THRESHOLDS are what
run_benchmark.ipynb actually uses (copy the calibrated numbers here and flip
CURRENT_THRESHOLD_STATUS to "frozen_v1.0" once they're locked in). Passing
thresholds=None switches to evidence-only mode (pass_fail stays None).
decide() is pure, so thresholds can be refit later from saved evidence
without rerunning the motion analysis - see validation.grid_search.

A subclass just needs calculate_evidence() and decide(). BodySideEvaluator
and TrajectoryEvaluator predate this file and don't use it.
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
            
            # evidence-only mode: calculate_evidence still runs, evaluate() below just skips decide()
            self.thresholds = None
            self.threshold_status = "not_set"
        else:
            self.thresholds = {**self.PROVISIONAL_THRESHOLDS, **thresholds}       # caller's values win on overlap
            self.threshold_status = threshold_status or "custom"
        self.event_params = event_params

    @classmethod
    def for_benchmark(cls):
        
        # the instance actually used for scoring, wired to this evaluator's current calibrated thresholds
        return cls(thresholds=cls.CURRENT_THRESHOLDS, threshold_status=cls.CURRENT_THRESHOLD_STATUS)

    # helpers
    def analyse(self, motion):
        return E.analyse(motion, self.event_params)

    @staticmethod
    def expected_of(requirement):
        return requirement.get("value", requirement.get("expected"))        # definitions have used both key names

    # to implement
    def calculate_evidence(self, motion, requirement, evaluation_case):
        """Return a JSON-safe dict. Put "unsupported": "<why>" to return pass_fail=None."""
        raise NotImplementedError

    def decide(self, evidence, expected):
        """Return (pass_fail, reason) using self.thresholds. Must not look at the motion."""
        raise NotImplementedError

    def decide_from_evidence(self, evidence, expected):
        """Unified calibration hook (same name on BodySideEvaluator)."""
        return self.decide(evidence, expected)

    # common interface
    def evaluate(self, motion, requirement, evaluation_case=None):
        expected = self.expected_of(requirement)
        motion = np.asarray(motion, dtype=np.float64)
        evidence = self.calculate_evidence(motion, requirement, evaluation_case or {})
        evidence.setdefault("judge", "rule")             # marks this result as rule-judged, vs. human or VLM

        if evidence.get("unsupported"):
            
            # calculate_evidence itself said it can't handle this requirement/value
            pass_fail, reason = None, f"Not decidable by rule: {evidence['unsupported']}"
        elif self.thresholds is None:
            # evidence-only mode (see __init__) - don't even attempt a decision
            pass_fail, reason = None, "Thresholds not set — evidence only."
        else:
            pass_fail, reason = self.decide(evidence, expected)

        # same shape every evaluator returns, rule-based or not, so callers never need to know which kind ran
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
    Returns RuleEvaluator, rebound to whichever common.BaseEvaluator is loaded now.

    run_benchmark.ipynb reloads common.py after a git pull, which creates a
    fresh BaseEvaluator class. Without this, a stale rule_base module would
    still subclass the old one and register_evaluator would reject it. So
    every eval_*.py calls this instead of importing RuleEvaluator directly.
    """
    import importlib
    import sys

    import common as _common
    mod = sys.modules[__name__]
    if mod.BaseEvaluator is not _common.BaseEvaluator:
        mod = importlib.reload(mod)            # stale reference - reload so RuleEvaluator subclasses the fresh BaseEvaluator
    return mod.RuleEvaluator


def verdict(ok):
    return "PASS" if ok else "FAIL"


def events_json(evs, limit=10):
    return [e.to_dict() for e in list(evs)[:limit]]
