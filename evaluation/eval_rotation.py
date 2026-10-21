"""
turn_direction requirements -> RotationEvaluator.

heading_yaw comes from the hip+shoulder line (positive = turned left). A turn
event (see events.py) is a run of yaw rate >= 25 deg/s with a net change of
>= 30 deg; the "dominant" turn is whichever event has the biggest |yaw change|.

PASS if the dominant turn goes the requested way and |yaw change| >= min_turn_deg.
score is the signed yaw change toward the requested side (negative = wrong way).

"""


from common import register_evaluator
from rule_base import current, events_json, verdict
 
RuleEvaluator = current()  # rebind to whatever common.py is loaded right now
 
SIGN = {"left": 1.0, "right": -1.0, "counterclockwise": 1.0, "clockwise": -1.0}
 
 
class RotationEvaluator(RuleEvaluator):
    EVALUATOR_NAME = "RotationEvaluator"
    PROVISIONAL_THRESHOLDS = {"min_turn_deg": 45.0}
 
    CURRENT_THRESHOLDS = {"min_turn_deg": 45.0}
    CURRENT_THRESHOLD_STATUS = "frozen_v1.0"
 
    def calculate_evidence(self, motion, requirement, case):
        expected = str(self.expected_of(requirement)).lower()
        if expected not in SIGN:
            return {"unsupported": f"turn_direction value {expected!r}"}
        a = self.analyse(motion)
        turns = a.get("turn")     # every detected turn event in the clip
        net = float(a.yaw[-1] - a.yaw[0])     # start-to-end yaw drift, just for the message/debugging

        # the biggest single turn is what actually gets judged, not the net drift -
        # a clip that turns left hard then drifts back a little should still pass as "turn left"
        dom = max(turns, key=lambda e: abs(e.attrs["yaw_change_deg"]), default=None)
        dyaw = dom.attrs["yaw_change_deg"] if dom else 0.0
        return {
            "n_turn_events": len(turns),
            "dominant_yaw_change_deg": float(dyaw),       # + = left
            "dominant_turn": dom.to_dict() if dom else None,
            "net_yaw_change_deg": net,
            "turn_events": events_json(turns),
            "score": float(SIGN[expected] * dyaw),     # flips sign so "right" turns score positive too
        }
 
    def decide(self, ev, expected):
        need = self.thresholds["min_turn_deg"]
        ok = ev["score"] >= need
        side = "left" if ev["dominant_yaw_change_deg"] > 0 else "right"     # for the message text only
        if ev["n_turn_events"] == 0:
            # nothing to compare against - fail outright instead of letting score=0 pass some edge case
            return "FAIL", f"No turn detected (net yaw {ev['net_yaw_change_deg']:+.0f} deg)."
        return verdict(ok), (f"dominant turn {ev['dominant_yaw_change_deg']:+.0f} deg ({side}); "
                             f"required {expected} >= {need:.0f} deg")
 
 
register_evaluator("RotationEvaluator", RotationEvaluator)
 
