"""
target / relation requirements -> SpatialRelationEvaluator.

Example from the Pilot: "moves the left hand across the body toward the right
shoulder" -> target=right_shoulder, relation=cross_body.

Which hand/foot gets checked: the body_side applying to the same action if
there is one, else "left/right hand" mentioned in the prompt text, else just
pick whichever effector comes closest (for target) or crosses furthest (for
relation).

target: minimum effector-to-target-joint distance over the whole clip,
normalised by shoulder width. PASS when that's <= target_max_dist_sw.

relation, each normalised by shoulder width unless noted:
  cross_body      lateral position past the midline, on the OPPOSITE side      PASS max >= cross_min_sw
  above_head      wrist height above the head top (metres)                     PASS max >= above_head_min_m
  in_front        wrist forward of the chest                                   PASS max >= front_min_sw
  behind_back     wrist behind the chest                                       PASS max >= behind_min_sw
  hands_together  wrist-to-wrist distance                                      PASS min <= together_max_sw
Anything else returns pass_fail=None - left for a human or a model judge.

"""


import numpy as np
 
import events as E
import primitives as P
from common import register_evaluator
from rule_base import current, verdict
 
RuleEvaluator = current()      # rebind to whatever common.py is loaded right now
 
TARGET_JOINTS = {
    "left_shoulder": 16, "right_shoulder": 17, "left_elbow": 18, "right_elbow": 19,
    "left_hand": 20, "right_hand": 21, "left_wrist": 20, "right_wrist": 21,
    "left_hip": 1, "right_hip": 2, "left_knee": 4, "right_knee": 5,
    "left_foot": 10, "right_foot": 11, "left_ankle": 7, "right_ankle": 8,
    "head": 15, "neck": 12, "chest": 9, "pelvis": 0, "waist": 0,
}
EFFECTOR = {("arm", "left"): 20, ("arm", "right"): 21, ("leg", "left"): 7, ("leg", "right"): 8}
LEG_WORDS = {"kick", "step", "foot", "leg", "knee"}
 
 
def _side_from_text(text, limb):
    
    # last-resort side lookup: scan the raw prompt for "left/right" directly before a body-part noun
    words = str(text).lower().replace(".", " ").replace(",", " ").split()
    nouns = {"hand", "arm", "wrist"} if limb == "arm" else {"foot", "leg", "knee"}
    for i, w in enumerate(words[:-1]):
        if w in ("left", "right") and words[i + 1] in nouns:
            return w
    return None
 
 
class SpatialRelationEvaluator(RuleEvaluator):
    EVALUATOR_NAME = "SpatialRelationEvaluator"
    PROVISIONAL_THRESHOLDS = {
        "target_max_dist_sw": 0.75,     # k   (× shoulder width)
        "cross_min_sw": 0.25,           # m   (× shoulder width past the midline)
        "above_head_min_m": 0.05,
        "front_min_sw": 1.0,
        "behind_min_sw": 0.2,
        "together_max_sw": 0.4,
    }
 
    CURRENT_THRESHOLDS = {"target_max_dist_sw": 0.75, "cross_min_sw": 0.1, "above_head_min_m": 0.05, "front_min_sw": 1.0, "behind_min_sw": 0.2, "together_max_sw": 0.4}
    CURRENT_THRESHOLD_STATUS = "frozen_v1.0"
 
    def _effectors(self, requirement, case):
        
        # work out which limb (arm/leg) and which side(s) to check, in order of how much we trust the source
        ctx = E.requirement_context(requirement, case)
        action = str(ctx["action"] or "").lower()
        limb = "leg" if any(w in action for w in LEG_WORDS) else "arm"  # guess limb from the action word
        side, source = ctx["body_side"], "body_side"
        if side not in ("left", "right"):
            
            # no explicit body_side requirement - fall back to scanning the prompt text itself
            side, source = _side_from_text(case.get("prompt", ""), limb), "prompt_text"
        if side not in ("left", "right"):
            
            # still unknown - check both sides and let the caller (below) pick whichever fits best
            return limb, ["left", "right"], "auto"
        return limb, [side], source
 
    def calculate_evidence(self, motion, requirement, case):
        rtype = requirement.get("type")
        value = str(self.expected_of(requirement)).lower()
        m = P.as_motion(motion)
        sw = P.body_scale(m)["shoulder_width"]
        limb, sides, source = self._effectors(requirement, case)
        ev = {"requirement_type": rtype, "limb": limb, "side_source": source, "shoulder_width_m": sw}
 
        if rtype == "target":
            if value not in TARGET_JOINTS:
                return {**ev, "unsupported": f"target {value!r}"}
            tj = TARGET_JOINTS[value]
            per = {}
            for s in sides:
                
                # distance from this side's effector to the target joint, at every frame
                d = np.linalg.norm(m[:, EFFECTOR[(limb, s)]] - m[:, tj], axis=1)
                per[s] = {"min_dist_m": float(d.min()), "frame": int(d.argmin()), "start_dist_m": float(d[0])}
            side = min(per, key=lambda s: per[s]["min_dist_m"])      # whichever side actually got closer wins
            ev.update({"side": side, "per_side": per, "target_joint": tj,
                       "min_dist_sw": per[side]["min_dist_m"] / sw, "score": per[side]["min_dist_m"] / sw})
            return ev
 
        if rtype != "relation":
            return {**ev, "unsupported": f"type {rtype!r}"}
 
        right, up, forward = P.body_frame(m)
        mid = 0.5 * (m[:, P.L_SHOULDER] + m[:, P.R_SHOULDER])      # reference point for all the relations below
        per = {}
        for s in sides:
            w = m[:, EFFECTOR[(limb, s)]]
            rel = w - mid
            lateral = (rel * right).sum(1)     # signed sideways offset from the body midline
            if value == "cross_body":
                series = lateral if s == "left" else -lateral           # + = on the opposite side
                per[s] = float(series.max() / sw)
            elif value == "above_head":
                per[s] = float((w[:, 1] - m[:, P.HEAD, 1]).max())     # wrist height minus head height
            elif value in ("in_front", "in_front_of_body", "front"):
                per[s] = float((rel * forward).sum(1).max() / sw)
            elif value in ("behind_back", "behind", "behind_body"):
                per[s] = float((-(rel * forward).sum(1)).max() / sw)
            elif value in ("hands_together", "together"):
                
                # symmetric by definition， just one wrist-to-wrist distance, no per-side split needed
                d = np.linalg.norm(m[:, P.L_WRIST] - m[:, P.R_WRIST], axis=1)
                per = {"both": float(d.min() / sw)}
                break
            else:
                return {**ev, "unsupported": f"relation {value!r}"}
            
        # closer is better for hands_together, farther is better for every other relation
        best = min(per, key=per.get) if value in ("hands_together", "together") else max(per, key=per.get)
        ev.update({"relation": value, "side": best, "per_side": per, "score": per[best]})
        return ev
 
    def decide(self, ev, expected):
        t = self.thresholds
        v = ev["score"]
        if ev["requirement_type"] == "target":
            need = t["target_max_dist_sw"]
            return verdict(v <= need), (f"{ev['side']} {ev['limb']} closest to {expected}: "
                                        f"{v:.2f} × shoulder width (need <= {need})")
        rel = ev["relation"]

    
        # each relation has its own threshold and its own pass direction (>= vs <=)
        if rel == "cross_body":
            need, ok = t["cross_min_sw"], v >= t["cross_min_sw"]
        elif rel == "above_head":
            need, ok = t["above_head_min_m"], v >= t["above_head_min_m"]
        elif rel in ("in_front", "in_front_of_body", "front"):
            need, ok = t["front_min_sw"], v >= t["front_min_sw"]
        elif rel in ("behind_back", "behind", "behind_body"):
            need, ok = t["behind_min_sw"], v >= t["behind_min_sw"]
        else:
            need, ok = t["together_max_sw"], v <= t["together_max_sw"]
        return verdict(ok), f"{rel}: {ev['side']} {ev['limb']} = {v:.2f} (threshold {need})"
 
 
register_evaluator("SpatialRelationEvaluator", SpatialRelationEvaluator)
 
