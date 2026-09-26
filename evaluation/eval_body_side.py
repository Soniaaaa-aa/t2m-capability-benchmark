"""
eval_body_side.py — body_side requirements (BodySideEvaluator)
===============================================================

Owner: <your name>   (edit this file only via your own branch + Pull Request)

Which side of the body — left / right / both — performs the action.

Pilot usage
-----------
C2-09 / C2-10  kicks the left / right leg out to the side   -> leg
C2-19 / C2-20  moves the left / right hand across the body  -> arm
C4-19          walks forward while raising the right hand   -> arm
C5-16          ... jumps once with both hands raised         -> arm (both)

Method
------
* Left / right come from the anatomical joint labels of the HumanML3D skeleton
  (joint 20 is always the left wrist), not from the world X axis, so the result
  does not depend on facing direction, walking or turning.

      arm: left 16/18/20 (shoulder, elbow, wrist)   right 17/19/21
      leg: left 1/4/7    (hip, knee, ankle)         right 2/5/8

* Limb activity: per frame, the end effector relative to its limb root,
  compared with the neutral hanging position (0, -L, 0), divided by the limb
  length L. activity = 95th percentile over the clip.
      ~0 hanging, ~0.5 walking arm swing, ~1.1 side kick, ~1.5 hand to the
      opposite shoulder, ~2.0 arm overhead.
  Rotation about Y and body size do not change it.

* Laterality index  LI = (A_left - A_right) / (A_left + A_right).

* Limb (arm / leg): explicit requirement field -> applies_to (v1.2) -> nearest preceding action
  requirement (kick -> leg, reach / raise_hand -> arm) -> sibling
  leg_direction / arm_direction -> words in the prompt -> "auto".

Decision (thresholds calibrated against Human Gold in analysis/analysis_rules.ipynb)
--------
  left : A_left  >= min_activity and LI >=  side_margin
  right: A_right >= min_activity and LI <= -side_margin
  both : both    >= min_activity and |LI| <= both_max_imbalance

Without thresholds the evaluator returns pass_fail = None plus the evidence,
consistent with TrajectoryEvaluator.

Scope: which side moved — not whether the action itself is correct
(ActionEvaluator), the kick direction (LimbGeometryEvaluator) or the target
shoulder (SpatialRelationEvaluator).
"""

import numpy as np

from common import (
    BaseEvaluator,
    get_human_label,
    iter_requirements,
    load_motion,
    register_evaluator,
)


class BodySideEvaluator(BaseEvaluator):
    """
    Evaluates body_side Requirements (left / right / both).

    Left and right are taken from the anatomical joint labels of the
    HumanML3D skeleton, not from the world X axis, so the result does not
    depend on the facing direction or on walking / turning.

    Input Motion : Standardised Motion [T, 22, 3]
    Output       : Requirement-level result (common Evaluator Interface)
    """

    EVALUATOR_NAME = "BodySideEvaluator"
    SUPPORTED_SIDES = {"left", "right", "both"}

    # (root, middle, end effector) — anatomical labels
    LIMB_JOINTS = {
        "arm": {"left": (16, 18, 20), "right": (17, 19, 21)},  # shoulder, elbow, wrist
        "leg": {"left": (1, 4, 7), "right": (2, 5, 8)},        # hip, knee, ankle
    }

    # Body-part nouns are checked before verbs ("raise_leg" -> leg).
    LEG_NOUNS = {"leg", "legs", "foot", "feet", "knee", "knees", "ankle", "toe", "toes"}
    ARM_NOUNS = {"hand", "hands", "arm", "arms", "wrist", "wrists", "elbow", "elbows", "fist"}
    LEG_VERBS = {"kick", "kicks", "step", "steps", "stomp", "stomps", "hop", "hops", "lunge"}
    ARM_VERBS = {"reach", "reaches", "wave", "waves", "punch", "punches", "clap", "claps",
                 "throw", "throws", "point", "points", "grab", "grabs", "raise", "raises"}

    PROVISIONAL_THRESHOLDS = {
        "min_activity": 0.7,         # limb must clearly leave the neutral pose
        "side_margin": 0.25,         # |LI| needed to call one side dominant
        "both_max_imbalance": 0.5,   # |LI| allowed for "both"
    }

    # Current rule used by run_benchmark.ipynb (run_all_evaluators).
    # After calibrating in evaluation/analysis/analysis_rules.ipynb, copy the
    # selected values here and update the status (e.g. "frozen_v1.0").
    CURRENT_THRESHOLDS = dict(PROVISIONAL_THRESHOLDS)
    CURRENT_THRESHOLD_STATUS = "provisional_not_frozen"

    @classmethod
    def for_benchmark(cls):
        return cls(thresholds=cls.CURRENT_THRESHOLDS, threshold_status=cls.CURRENT_THRESHOLD_STATUS)

    def __init__(self, thresholds=None, threshold_status=None, percentile=95.0):
        """
        thresholds       : None -> evidence only (pass_fail = None), as in
                           TrajectoryEvaluator. A dict -> PASS / FAIL decision;
                           missing keys fall back to PROVISIONAL_THRESHOLDS.
        threshold_status : label stored in the result, e.g. "calibrated".
        percentile       : percentile of the per-frame deviation used as activity.
        """
        if thresholds is None:
            self.thresholds = None
            self.threshold_status = "not_set"
        else:
            self.thresholds = {**self.PROVISIONAL_THRESHOLDS, **thresholds}
            self.threshold_status = threshold_status or "custom"
        self.percentile = float(percentile)

    # --------------------------------------------------------
    # 1. Which limb does the Requirement refer to?
    # --------------------------------------------------------
    @classmethod
    def _limb_from_words(cls, text):
        words = set(
            str(text).lower().replace("_", " ").replace("-", " ")
            .replace(",", " ").replace(".", " ").split()
        )
        leg_n, arm_n = bool(words & cls.LEG_NOUNS), bool(words & cls.ARM_NOUNS)
        if leg_n != arm_n:
            return "leg" if leg_n else "arm"
        if leg_n and arm_n:
            return None  # e.g. "hand touches foot": ambiguous
        leg_v, arm_v = bool(words & cls.LEG_VERBS), bool(words & cls.ARM_VERBS)
        if leg_v != arm_v:
            return "leg" if leg_v else "arm"
        return None

    def infer_limb(self, requirement, evaluation_case=None):
        """Return (limb, source) with limb in {"arm", "leg", "auto"}."""
        # (a) explicit field on the Requirement
        for key in ("limb", "body_part", "effector"):
            if requirement.get(key):
                limb = self._limb_from_words(requirement[key])
                if limb:
                    return limb, f"requirement.{key}"

        requirements = (evaluation_case or {}).get("requirements", [])

        # (a2) explicit applies_to (definition v1.2+): the action(s) it refers to
        targets = requirement.get("applies_to") or []
        if targets:
            by_id = {r.get("id"): r for r in requirements}
            for rid in targets:
                r = by_id.get(rid)
                if r and r.get("type") == "action":
                    limb = self._limb_from_words(r.get("value", r.get("expected", "")))
                    if limb:
                        return limb, f"applies_to:{rid}={r.get('value')}"

        # (b) nearest preceding action Requirement
        index = next((i for i, r in enumerate(requirements) if r is requirement), None)
        if index is None:
            index = next((i for i, r in enumerate(requirements) if r == requirement), None)
        if index is not None:
            for r in reversed(requirements[:index]):
                if r.get("type") == "action":
                    limb = self._limb_from_words(r.get("value", r.get("expected", "")))
                    if limb:
                        return limb, f"action:{r.get('value', r.get('expected'))}"
                    break

        # (c) sibling limb-specific Requirements
        types = {r.get("type") for r in requirements}
        if ("leg_direction" in types) != ("arm_direction" in types):
            return ("leg" if "leg_direction" in types else "arm"), "sibling_requirement"

        # (d) prompt text
        limb = self._limb_from_words((evaluation_case or {}).get("prompt", ""))
        if limb:
            return limb, "prompt_text"

        return "auto", "auto"

    # --------------------------------------------------------
    # 2. Limb activity evidence
    # --------------------------------------------------------
    def limb_activity(self, motion, limb):
        """Activity of the left and right limb of one type."""
        joints = self.LIMB_JOINTS[limb]

        # One limb length for both sides, so the normalisation is symmetric.
        lengths = []
        for side in ("left", "right"):
            root, middle, end = joints[side]
            segment = (
                np.linalg.norm(motion[:, middle] - motion[:, root], axis=1)
                + np.linalg.norm(motion[:, end] - motion[:, middle], axis=1)
            )
            lengths.append(float(np.median(segment)))
        limb_length = float(np.mean(lengths))
        if limb_length < 1e-6:
            raise ValueError(f"Degenerate {limb} length: {limb_length}")

        neutral = np.array([0.0, -limb_length, 0.0])
        result = {"limb": limb, "limb_length_m": limb_length}

        for side in ("left", "right"):
            root, _, end = joints[side]
            rel = motion[:, end] - motion[:, root]          # effector relative to its root
            deviation = np.linalg.norm(rel - neutral, axis=1) / limb_length
            rise = (rel[:, 1] + limb_length) / limb_length  # height gained vs hanging
            horizontal = np.linalg.norm(rel[:, [0, 2]], axis=1) / limb_length

            result[side] = {
                "activity": float(np.percentile(deviation, self.percentile)),
                "peak_deviation": float(deviation.max()),
                "peak_rise": float(np.percentile(rise, self.percentile)),
                "peak_horizontal": float(np.percentile(horizontal, self.percentile)),
            }

        a_left = result["left"]["activity"]
        a_right = result["right"]["activity"]
        result["laterality_index"] = float((a_left - a_right) / (a_left + a_right + 1e-8))
        return result

    def calculate_evidence(self, motion, limb):
        motion = np.asarray(motion, dtype=np.float64)
        if motion.ndim != 3 or motion.shape[1:] != (22, 3):
            raise ValueError(f"Expected [T, 22, 3], got {motion.shape}")
        if not np.isfinite(motion).all():
            raise ValueError("Motion contains NaN or Inf values.")

        per_limb = {name: self.limb_activity(motion, name) for name in ("arm", "leg")}

        used = limb
        if limb == "auto":
            used = max(per_limb, key=lambda n: max(per_limb[n]["left"]["activity"],
                                                   per_limb[n]["right"]["activity"]))
        chosen = per_limb[used]

        li = chosen["laterality_index"]
        raw_side = "left" if li > 0 else "right" if li < 0 else "undetermined"

        return {
            "limb_used": used,
            "left_activity": chosen["left"]["activity"],
            "right_activity": chosen["right"]["activity"],
            "laterality_index": li,
            "raw_side": raw_side,
            "limb_length_m": chosen["limb_length_m"],
            "left_detail": chosen["left"],
            "right_detail": chosen["right"],
            # Diagnostic: all four limbs, e.g. to spot an arm moving for a kick prompt.
            "all_limbs": {
                f"{side}_{name}": per_limb[name][side]["activity"]
                for name in ("arm", "leg") for side in ("left", "right")
            },
        }

    # --------------------------------------------------------
    # 3. Decision
    # --------------------------------------------------------
    def decide(self, required_side, evidence, thresholds):
        a_l, a_r = evidence["left_activity"], evidence["right_activity"]
        li = evidence["laterality_index"]
        t_min = thresholds["min_activity"]
        t_side = thresholds["side_margin"]
        t_both = thresholds["both_max_imbalance"]

        left_active, right_active = a_l >= t_min, a_r >= t_min

        if not (left_active or right_active):
            predicted = "none"
        elif left_active and li >= t_side:
            predicted = "left"
        elif right_active and li <= -t_side:
            predicted = "right"
        elif left_active and right_active and abs(li) <= t_both:
            predicted = "both"
        else:
            predicted = "undetermined"

        if required_side == "left":
            passed = left_active and li >= t_side
        elif required_side == "right":
            passed = right_active and li <= -t_side
        else:
            passed = left_active and right_active and abs(li) <= t_both

        reason = (
            f"{evidence['limb_used']}: A_left={a_l:.2f}, A_right={a_r:.2f}, LI={li:+.2f} "
            f"(min_activity={t_min}, side_margin={t_side}, both_max_imbalance={t_both}) "
            f"-> predicted {predicted}, required {required_side}"
        )
        return ("PASS" if passed else "FAIL"), predicted, reason

    def decide_from_evidence(self, evidence, expected):
        """Unified calibration hook (validation.grid_search, finalize.redecide):
        decision from saved evidence with self.thresholds. Returns (pass_fail, reason)."""
        pass_fail, _, reason = self.decide(str(expected).lower().strip(), evidence, self.thresholds)
        return pass_fail, reason

    # --------------------------------------------------------
    # 4. Common Evaluator Interface
    # --------------------------------------------------------
    def evaluate(self, motion, requirement, evaluation_case=None):
        required = requirement.get("value", requirement.get("expected"))
        required_side = str(required).lower().strip()
        if required_side not in self.SUPPORTED_SIDES:
            raise ValueError(f"Unsupported body_side: {required!r}")

        limb, limb_source = self.infer_limb(requirement, evaluation_case)
        evidence = self.calculate_evidence(motion, limb)

        li = evidence["laterality_index"]
        if required_side == "left":
            score = li
        elif required_side == "right":
            score = -li
        else:  # balance of the two sides: 1 = equal, 0 = one side only
            hi = max(evidence["left_activity"], evidence["right_activity"])
            lo = min(evidence["left_activity"], evidence["right_activity"])
            score = lo / hi if hi > 1e-8 else 0.0

        if self.thresholds is None:
            pass_fail, predicted = None, None
            reason = (
                f"Thresholds not set — evidence only. {evidence['limb_used']}: "
                f"A_left={evidence['left_activity']:.2f}, "
                f"A_right={evidence['right_activity']:.2f}, LI={li:+.2f}"
            )
        else:
            pass_fail, predicted, reason = self.decide(required_side, evidence, self.thresholds)

        return {
            "requirement_id": requirement.get("requirement_id", requirement.get("id")),
            "requirement_type": "body_side",
            "expected_value": required_side,
            "evaluator_name": self.EVALUATOR_NAME,
            "pass_fail": pass_fail,
            "score": float(score),
            "predicted_side": predicted,
            "predicted_side_raw": evidence["raw_side"],
            "limb": evidence["limb_used"],
            "limb_source": limb_source,
            "thresholds": self.thresholds,
            "threshold_status": self.threshold_status,
            "evidence": evidence,
            "reason": reason,
        }


register_evaluator("BodySideEvaluator", BodySideEvaluator)


# ============================================================
# Pilot workflow helpers (used by run_benchmark.ipynb, STEP 11)
# ============================================================

def _label_or_none(human_gold_labels, case, req_index):
    """Human Gold Label, or None when no labels were loaded for this prompt."""
    if not human_gold_labels or case["prompt_id"] not in human_gold_labels:
        return None
    return get_human_label(human_gold_labels, case, req_index)


def collect_body_side_evidence(evaluation_cases, human_gold_labels=None):
    """
    STEP 11A — run BodySideEvaluator WITHOUT thresholds on every body_side
    Requirement and attach the Human Gold Label (None if not available).
    """
    evaluator = BodySideEvaluator()  # thresholds not set -> evidence only
    rows = []
    for case, req_index, req in iter_requirements(evaluation_cases, "body_side"):
        motion = load_motion(case["motion_path"])
        result = evaluator.evaluate(motion, req, case)
        ev = result["evidence"]
        rows.append({
            "prompt_id": case["prompt_id"],
            "requirement_index": req_index,
            "prompt": case["prompt"],
            "motion_path": case["motion_path"],
            "required_side": result["expected_value"],
            "limb": result["limb"],
            "limb_source": result["limb_source"],
            "left_activity": ev["left_activity"],
            "right_activity": ev["right_activity"],
            "laterality_index": ev["laterality_index"],
            "score": result["score"],
            "predicted_side_raw": result["predicted_side_raw"],
            "human_label": _label_or_none(human_gold_labels, case, req_index),
            "all_limbs": ev["all_limbs"],
        })
    return rows


def predict_body_side(row, thresholds):
    """PASS / FAIL for one evidence row (same rule as BodySideEvaluator.decide)."""
    a_l, a_r, li = row["left_activity"], row["right_activity"], row["laterality_index"]
    if row["required_side"] == "left":
        ok = a_l >= thresholds["min_activity"] and li >= thresholds["side_margin"]
    elif row["required_side"] == "right":
        ok = a_r >= thresholds["min_activity"] and li <= -thresholds["side_margin"]
    else:
        ok = (a_l >= thresholds["min_activity"] and a_r >= thresholds["min_activity"]
              and abs(li) <= thresholds["both_max_imbalance"])
    return "PASS" if ok else "FAIL"


CANDIDATE_MIN_ACTIVITY = [round(0.3 + 0.1 * i, 2) for i in range(10)]      # 0.3 … 1.2
CANDIDATE_SIDE_MARGIN = [round(0.05 * i, 2) for i in range(1, 13)]         # 0.05 … 0.6
CANDIDATE_BOTH_IMBALANCE = [0.3, 0.5, 0.7]


def calibrate_body_side_thresholds(rows, min_labelled_for_calibrated=10):
    """
    STEP 11B — grid search against Human Gold (UNCERTAIN / missing excluded).

    Best = highest accuracy, then fewest false PASS, then closest to the
    provisional thresholds. Returns a dict with thresholds, status and details.
    """
    provisional = BodySideEvaluator.PROVISIONAL_THRESHOLDS
    usable = [r for r in rows if r["human_label"] in {"PASS", "FAIL"}]

    if not usable:
        return {"thresholds": dict(provisional), "status": "provisional",
                "usable": 0, "correct": None, "false_pass": None,
                "n_best": None, "grid_size": 0, "cases": []}

    has_both = any(r["required_side"] == "both" for r in usable)
    both_grid = CANDIDATE_BOTH_IMBALANCE if has_both else [provisional["both_max_imbalance"]]

    grid = []
    for t_min in CANDIDATE_MIN_ACTIVITY:
        for t_side in CANDIDATE_SIDE_MARGIN:
            for t_both in both_grid:
                t = {"min_activity": t_min, "side_margin": t_side, "both_max_imbalance": t_both}
                preds = [predict_body_side(r, t) for r in usable]
                correct = sum(p == r["human_label"] for p, r in zip(preds, usable))
                false_pass = sum(p == "PASS" and r["human_label"] == "FAIL"
                                 for p, r in zip(preds, usable))
                distance = sum(abs(t[k] - provisional[k]) / provisional[k] for k in t)
                grid.append((correct, -false_pass, -distance, t))

    grid.sort(key=lambda g: g[:3], reverse=True)
    best_correct, neg_fp, _, best = grid[0]
    status = ("calibrated" if len(usable) >= min_labelled_for_calibrated
              else "calibrated_small_sample")

    return {
        "thresholds": best,
        "status": status,
        "usable": len(usable),
        "correct": best_correct,
        "false_pass": -neg_fp,
        "n_best": sum(1 for g in grid if g[0] == best_correct),
        "grid_size": len(grid),
        "cases": [dict(r, predicted=predict_body_side(r, best)) for r in usable],
    }


def evaluate_body_side(evaluation_cases, thresholds, threshold_status, human_gold_labels=None):
    """STEP 11C — final PASS / FAIL for every body_side Requirement."""
    evaluator = BodySideEvaluator(thresholds=thresholds, threshold_status=threshold_status)
    results = []
    for case, req_index, req in iter_requirements(evaluation_cases, "body_side"):
        result = evaluator.evaluate(load_motion(case["motion_path"]), req, case)
        result.update({
            "model": case["model"],
            "prompt_id": case["prompt_id"],
            "requirement_index": req_index,
            "human_label": _label_or_none(human_gold_labels, case, req_index),
        })
        results.append(result)
    return results
