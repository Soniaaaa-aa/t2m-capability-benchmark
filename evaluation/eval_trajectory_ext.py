"""
AttributeEvaluator (attribute: slow/fast) and BodyFrameDirectionEvaluator,
which is the default "direction" evaluator as of 2026-09-26. Additive file -
eval_trajectory.py and its world-frame TrajectoryEvaluator are untouched.

AttributeEvaluator compares motions within the same pair_id: speed is the
mean root speed over the walk events of the linked action. fast passes when
v_self >= speed_ratio * v_partner and v_self clears min_walk_speed; slow
passes when v_partner >= speed_ratio * v_self, v_partner clears
min_walk_speed, and the slow motion actually walks at all. The partner file
is <same folder>/<partner_prompt_id>.npy, with the pairing read from the
benchmark definition (or case["pair_id"] if that's already set). If there's
no partner to compare against, it falls back to absolute thresholds
(abs_slow_max / abs_fast_min) - evidence["mode"] records which rule fired.

BodyFrameDirectionEvaluator projects the displacement during the action onto
the body's own facing direction rather than world axes: for the first step
of a prompt (or when there's no order requirement) that's the facing at clip
start, so "walks to the left" means left of wherever the person started
facing, side-step or turn included; for a later step in an ordered prompt
it's the facing at the start of that step instead. To compare against the
older world-frame rule: run_all_evaluators(cases, config={**EVALUATION_CONFIG,
"direction": "TrajectoryEvaluator"}). Same starting threshold either way:
0.50 m of required displacement.

"""




import json
from functools import lru_cache
from pathlib import Path
 
import numpy as np
 
import events as E
import primitives as P
from common import load_motion, register_evaluator
from rule_base import current, verdict
 
RuleEvaluator = current()              # rebind to whatever common.py is loaded right now
 
DEFAULT_DEFINITION = Path(__file__).resolve().parent.parent / "benchmark" / "Benchmark_18pilot_78main.json"

 
# NEW: lets the notebook tell AttributeEvaluator which benchmark file
# is actually in use for this run, instead of relying on a fixed
# filename inside the repo that can drift out of sync.
_ACTIVE_DEFINITION_PATH = None
 
def set_active_benchmark_path(path):
    
    # called once by the notebook at the start of a run, before any AttributeEvaluator is built
    global _ACTIVE_DEFINITION_PATH
    _ACTIVE_DEFINITION_PATH = str(path)
OPPOSITE = {"slow": "fast", "fast": "slow", "slowly": "fast", "quickly": "slow", "quick": "slow"}           # what the partner should be
SPEED_WORDS = {"slow": "slow", "slowly": "slow", "fast": "fast", "quickly": "fast", "quick": "fast"}        # normalize prompt wording
 
 
@lru_cache(maxsize=4)
def _pairs(definition_path):
    
    # prompt_id -> (pair_id, that prompt's requirements), read once per definition file and cached
    p = Path(definition_path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for pr in data.get("prompts", []):
        if pr.get("pair_id"):
            out[pr["prompt_id"]] = (pr["pair_id"], [r for r in pr.get("requirements", [])])
    return out
 
 
def walk_speed(analysis, action_req=None, case=None):
    """Mean root speed (m/s) over the walk events; falls back to the whole clip."""
    evs = analysis.get("walk")
    if evs:
        frames = np.concatenate([np.arange(e.start, e.end) for e in evs])
        return float(analysis.root["speed"][frames].mean()), True
    return float(analysis.root["speed"].mean()), False
 
 
class AttributeEvaluator(RuleEvaluator):
    EVALUATOR_NAME = "AttributeEvaluator"
    PROVISIONAL_THRESHOLDS = {
        "speed_ratio": 1.2,         # r(plan §4)
        "min_walk_speed": 0.3,      # v_min  m/s
        "abs_slow_max": 0.9,        # fallback without a partner, m/s
        "abs_fast_min": 1.4,
    }
 
    CURRENT_THRESHOLDS = {"speed_ratio": 1.2, "min_walk_speed": 0.3, "abs_slow_max": 0.9, "abs_fast_min": 1.4}
    CURRENT_THRESHOLD_STATUS = "frozen_v1.0"
 
    def __init__(self, *args, definition_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.definition_path = str(definition_path or DEFAULT_DEFINITION)  # which benchmark file to read pairs from
 
    @classmethod
    def for_benchmark(cls):
        
        # the instance actually used for scoring - wired to whichever definition file the run set as active
        return cls(
            thresholds=cls.CURRENT_THRESHOLDS,
            threshold_status=cls.CURRENT_THRESHOLD_STATUS,
            definition_path=_ACTIVE_DEFINITION_PATH,
        )
 
    def find_partner(self, case, value):
        pairs = _pairs(self.definition_path)
        pid = case.get("prompt_id")
        # prefer an explicit pair_id already on the case, else look it up from the benchmark definition
        pair_id = case.get("pair_id") or (pairs.get(pid) or (None,))[0]
        if not pair_id:
            return None, None                      # unpaired - caller falls back to the absolute thresholds
        want = OPPOSITE.get(value)                 # we're "slow", so the partner should be the one required to be "fast" (or vice versa)
        for other, (pair, reqs) in pairs.items():
            
            # another prompt in the same pair whose own requirements ask for the opposite speed word
            if other != pid and pair == pair_id and any(
                    r.get("type") == "attribute" and SPEED_WORDS.get(str(r.get("value")).lower()) == want
                    for r in reqs):
                return other, pair_id
        return None, pair_id              # pair exists, but nothing in it actually requires the opposite speed
 
    def calculate_evidence(self, motion, requirement, case):
        value = SPEED_WORDS.get(str(self.expected_of(requirement)).lower())       # normalize to "slow" or "fast"
        if value is None:
            return {"unsupported": f"attribute {self.expected_of(requirement)!r} (only slow / fast)"}
        a = self.analyse(motion)
        v_self, walked = walk_speed(a)
        
        # default to the no-partner case; overwritten below once we know whether one exists
        ev = {"attribute": value, "self_speed_mps": v_self, "self_walk_detected": walked,
              "mode": "absolute_fallback", "partner_prompt_id": None}
        partner, pair_id = self.find_partner(case, value)
        ev["pair_id"] = pair_id
        path = Path(case.get("motion_path", "")).parent / f"{partner}.npy" if partner else None
        if path is not None and path.exists():
            
            # partner's motion is actually on disk， score the two against each other directly
            pa = self.analyse(load_motion(path))
            v_partner, partner_walked = walk_speed(pa)
            ratio = v_self / max(v_partner, 1e-6)
            ev.update({"mode": "paired", "partner_prompt_id": partner, "partner_speed_mps": v_partner,
                       "partner_walk_detected": partner_walked, "speed_ratio_self_over_partner": ratio,
                       "score": ratio if value == "fast" else 1.0 / max(ratio, 1e-6)})
        else:
            ev["score"] = v_self
            if partner:
                ev["partner_missing_file"] = str(path)           # we know who it should be, just can't find the file
        return ev
 
    def decide(self, ev, expected):
        t = self.thresholds
        if ev["mode"] == "paired":
            r = ev["speed_ratio_self_over_partner"]
            if ev["attribute"] == "fast":
                ok = r >= t["speed_ratio"] and ev["self_speed_mps"] >= t["min_walk_speed"]
            else:
                
                # slow: partner must be clearly faster by the same ratio, AND this side must actually be
                # walking (not just standing still, which would trivially count as "slow")
                ok = (1 / max(r, 1e-6) >= t["speed_ratio"] and ev["partner_speed_mps"] >= t["min_walk_speed"]
                      and ev["self_walk_detected"])
            return verdict(ok), (f"{ev['attribute']}: {ev['self_speed_mps']:.2f} m/s vs "
                                 f"{ev['partner_prompt_id']} {ev['partner_speed_mps']:.2f} m/s "
                                 f"(ratio {r:.2f}, need ×{t['speed_ratio']})")
        
        # no partner available - judge against a fixed speed instead of a relative ratio
        v = ev["self_speed_mps"]
        if ev["attribute"] == "fast":
            ok = v >= t["abs_fast_min"]
        else:
            ok = ev["self_walk_detected"] and v <= t["abs_slow_max"]
        return verdict(ok), f"{ev['attribute']} (no partner, absolute rule): {v:.2f} m/s"
 
 
class BodyFrameDirectionEvaluator(RuleEvaluator):
    EVALUATOR_NAME = "BodyFrameDirectionEvaluator"
    PROVISIONAL_THRESHOLDS = {"min_displacement": 0.50}      # same d_min as TrajectoryEvaluator
 
    CURRENT_THRESHOLDS = {"min_displacement": 0.50}
    CURRENT_THRESHOLD_STATUS = "frozen_v1.0"
    AXES = {"forward": (1, +1), "backward": (1, -1), "right": (0, +1), "left": (0, -1)}
 
    def calculate_evidence(self, motion, requirement, case):
        want = str(self.expected_of(requirement)).lower()   # forward / backward / left / right
        if want not in self.AXES:
            return {"unsupported": f"direction {want!r}"}
        m = P.as_motion(motion)
        a = self.analyse(m)
        ctx = E.requirement_context(requirement, case)      # tells us the step index and any linked action
        window, source = (0, len(m)), "whole_clip"          # default: the whole clip, unless narrowed below
        if ctx["action_req"] is not None:
            
            # an attached action requirement narrows the window to just while that action happens
            _, evs = E.events_for_action(a, ctx["action_req"], case)
            if evs:
                window, source = (min(e.start for e in evs), max(e.end for e in evs)), "action_events"
        ref = window[0] if ctx["step"] else 0          # first step: facing at the clip start
        
        # smoothed motion (events.EVENT_PARAMS["smooth_window"]) so one jittery frame
        # cannot tilt the reference facing
        lat, fwd = P.displacement_in_frame(a.motion, window[0], window[1], frame_index=ref)
        axis, sign = self.AXES[want]
        req = (lat, fwd)[axis] * sign                    # just the requested axis, signed so positive means "the right way"
        d = m[window[1] - 1, 0] - m[window[0], 0]        # raw world-space displacement, kept for comparison/debugging
        return {"window": list(window), "window_source": source, "reference_frame": int(ref),
                "reference_yaw_deg": float(a.yaw[ref]), "step": ctx["step"],
                "lateral_right_m": lat, "forward_m": fwd, "world_dx_m": float(d[0]), "world_dz_m": float(d[2]),
                "required_displacement_m": float(req), "score": float(req)}
 
    def decide(self, ev, expected):
        
        # the signed displacement in the requested direction just needs to clear the minimum
        need = self.thresholds["min_displacement"]
        return verdict(ev["required_displacement_m"] >= need), (
            f"body frame (ref frame {ev['reference_frame']}): right {ev['lateral_right_m']:+.2f} m, "
            f"forward {ev['forward_m']:+.2f} m; required {expected} >= {need} m")
 
 
register_evaluator("AttributeEvaluator", AttributeEvaluator)
register_evaluator("BodyFrameDirectionEvaluator", BodyFrameDirectionEvaluator)
