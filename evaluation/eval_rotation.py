"""
eval_rotation.py — turn_direction requirements (RotationEvaluator)
==================================================================

Owner: A (plan §8)   — new file; eval_trajectory.py is not touched.

Pilot: C1-06 / C1-07 (turn left / right), C4-01, C4-02, C5-16 (turn right).

Method (plan §4)
  heading_yaw: facing angle from the hip + shoulder line, + = turned LEFT.
  Turn events (events.py) = runs of yaw rate >= 25 deg/s, net change >= 30 deg.
  The dominant turn = the turn event with the largest |Δyaw|.

Decision
  PASS  dominant turn has the requested sign and |Δyaw| >= min_turn_deg
  score = Δyaw signed toward the requested side (deg; negative = wrong way)
"""

from common import register_evaluator
from rule_base import current, events_json, verdict

RuleEvaluator = current()   # base class bound to the loaded common.py (safe after reload)

SIGN = {"left": 1.0, "right": -1.0, "counterclockwise": 1.0, "clockwise": -1.0}


class RotationEvaluator(RuleEvaluator):
    EVALUATOR_NAME = "RotationEvaluator"
    PROVISIONAL_THRESHOLDS = {"min_turn_deg": 45.0}
    CURRENT_THRESHOLDS = dict(PROVISIONAL_THRESHOLDS)
    CURRENT_THRESHOLD_STATUS = "provisional_not_frozen"

    def calculate_evidence(self, motion, requirement, case):
        expected = str(self.expected_of(requirement)).lower()
        if expected not in SIGN:
            return {"unsupported": f"turn_direction value {expected!r}"}
        a = self.analyse(motion)
        turns = a.get("turn")
        net = float(a.yaw[-1] - a.yaw[0])
        dom = max(turns, key=lambda e: abs(e.attrs["yaw_change_deg"]), default=None)
        dyaw = dom.attrs["yaw_change_deg"] if dom else 0.0
        return {
            "n_turn_events": len(turns),
            "dominant_yaw_change_deg": float(dyaw),       # + = left
            "dominant_turn": dom.to_dict() if dom else None,
            "net_yaw_change_deg": net,
            "turn_events": events_json(turns),
            "score": float(SIGN[expected] * dyaw),
        }

    def decide(self, ev, expected):
        need = self.thresholds["min_turn_deg"]
        ok = ev["score"] >= need
        side = "left" if ev["dominant_yaw_change_deg"] > 0 else "right"
        if ev["n_turn_events"] == 0:
            return "FAIL", f"No turn detected (net yaw {ev['net_yaw_change_deg']:+.0f} deg)."
        return verdict(ok), (f"dominant turn {ev['dominant_yaw_change_deg']:+.0f} deg ({side}); "
                             f"required {expected} >= {need:.0f} deg")


register_evaluator("RotationEvaluator", RotationEvaluator)
