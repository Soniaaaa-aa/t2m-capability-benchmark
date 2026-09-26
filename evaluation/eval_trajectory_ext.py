"""
eval_trajectory_ext.py — attribute (AttributeEvaluator) + body-frame direction (default)
=======================================================================================

Owner: A (plan §8). New file: eval_trajectory.py (A's TrajectoryEvaluator) is
NOT modified. Everything here is additive.

1. AttributeEvaluator  — attribute slow / fast   (Pilot: C3-01 slow, C3-02 fast)
   Plan §4: compare inside the same pair_id.
     speed = mean root speed over the walk events of the linked action
     fast  PASS  v_self >= speed_ratio × v_partner  and  v_self >= min_walk_speed
     slow  PASS  v_partner >= speed_ratio × v_self  and  v_partner >= min_walk_speed
                 and the slow motion itself walks (a walk event exists)
   The partner motion is <same folder>/<partner prompt_id>.npy; the pair is read
   from the benchmark definition in benchmark/ (or case["pair_id"] if present).
   Without a partner: absolute fallback  slow <= abs_slow_max,  fast >= abs_fast_min
   (evidence["mode"] says which rule was used).

2. BodyFrameDirectionEvaluator — the DEFAULT direction evaluator
   (EVALUATION_CONFIG["direction"], team decision 2026-09-26).
   Plan §4: displacement during the action, projected onto the body's facing:
     * first step of a prompt (or no order requirement): facing at the clip start,
       so "walks to the left" is left of where the person started facing, whether
       they side-step or turn and walk;
     * later step of an ordered prompt (C5-16 "turns right, walks forward"): facing
       at the start of that action, i.e. after the previous step.
   A's world-frame TrajectoryEvaluator (eval_trajectory.py) is unchanged and still
   registered; to compare:
       run_all_evaluators(cases, config={**EVALUATION_CONFIG, "direction": "TrajectoryEvaluator"})
   Same starting threshold as A's rule: required displacement >= 0.50 m.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import events as E
import primitives as P
from common import load_motion, register_evaluator
from rule_base import current, verdict

RuleEvaluator = current()   # base class bound to the loaded common.py (safe after reload)

DEFAULT_DEFINITION = Path(__file__).resolve().parent.parent / "benchmark" / "pilot_benchmark_definition.json"
OPPOSITE = {"slow": "fast", "fast": "slow", "slowly": "fast", "quickly": "slow", "quick": "slow"}
SPEED_WORDS = {"slow": "slow", "slowly": "slow", "fast": "fast", "quickly": "fast", "quick": "fast"}


@lru_cache(maxsize=4)
def _pairs(definition_path):
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
        "speed_ratio": 1.2,        # r      (plan §4)
        "min_walk_speed": 0.3,     # v_min  m/s
        "abs_slow_max": 0.9,       # fallback without a partner, m/s
        "abs_fast_min": 1.4,
    }
    CURRENT_THRESHOLDS = dict(PROVISIONAL_THRESHOLDS)
    CURRENT_THRESHOLD_STATUS = "provisional_not_frozen"

    def __init__(self, *args, definition_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.definition_path = str(definition_path or DEFAULT_DEFINITION)

    def find_partner(self, case, value):
        pairs = _pairs(self.definition_path)
        pid = case.get("prompt_id")
        pair_id = case.get("pair_id") or (pairs.get(pid) or (None,))[0]
        if not pair_id:
            return None, None
        want = OPPOSITE.get(value)
        for other, (pair, reqs) in pairs.items():
            if other != pid and pair == pair_id and any(
                    r.get("type") == "attribute" and SPEED_WORDS.get(str(r.get("value")).lower()) == want
                    for r in reqs):
                return other, pair_id
        return None, pair_id

    def calculate_evidence(self, motion, requirement, case):
        value = SPEED_WORDS.get(str(self.expected_of(requirement)).lower())
        if value is None:
            return {"unsupported": f"attribute {self.expected_of(requirement)!r} (only slow / fast)"}
        a = self.analyse(motion)
        v_self, walked = walk_speed(a)
        ev = {"attribute": value, "self_speed_mps": v_self, "self_walk_detected": walked,
              "mode": "absolute_fallback", "partner_prompt_id": None}
        partner, pair_id = self.find_partner(case, value)
        ev["pair_id"] = pair_id
        path = Path(case.get("motion_path", "")).parent / f"{partner}.npy" if partner else None
        if path is not None and path.exists():
            pa = self.analyse(load_motion(path))
            v_partner, partner_walked = walk_speed(pa)
            ratio = v_self / max(v_partner, 1e-6)
            ev.update({"mode": "paired", "partner_prompt_id": partner, "partner_speed_mps": v_partner,
                       "partner_walk_detected": partner_walked, "speed_ratio_self_over_partner": ratio,
                       "score": ratio if value == "fast" else 1.0 / max(ratio, 1e-6)})
        else:
            ev["score"] = v_self
            if partner:
                ev["partner_missing_file"] = str(path)
        return ev

    def decide(self, ev, expected):
        t = self.thresholds
        if ev["mode"] == "paired":
            r = ev["speed_ratio_self_over_partner"]
            if ev["attribute"] == "fast":
                ok = r >= t["speed_ratio"] and ev["self_speed_mps"] >= t["min_walk_speed"]
            else:
                ok = (1 / max(r, 1e-6) >= t["speed_ratio"] and ev["partner_speed_mps"] >= t["min_walk_speed"]
                      and ev["self_walk_detected"])
            return verdict(ok), (f"{ev['attribute']}: {ev['self_speed_mps']:.2f} m/s vs "
                                 f"{ev['partner_prompt_id']} {ev['partner_speed_mps']:.2f} m/s "
                                 f"(ratio {r:.2f}, need ×{t['speed_ratio']})")
        v = ev["self_speed_mps"]
        if ev["attribute"] == "fast":
            ok = v >= t["abs_fast_min"]
        else:
            ok = ev["self_walk_detected"] and v <= t["abs_slow_max"]
        return verdict(ok), f"{ev['attribute']} (no partner, absolute rule): {v:.2f} m/s"


class BodyFrameDirectionEvaluator(RuleEvaluator):
    EVALUATOR_NAME = "BodyFrameDirectionEvaluator"
    PROVISIONAL_THRESHOLDS = {"min_displacement": 0.50}     # same d_min as TrajectoryEvaluator
    CURRENT_THRESHOLDS = dict(PROVISIONAL_THRESHOLDS)
    CURRENT_THRESHOLD_STATUS = "provisional_not_frozen"
    AXES = {"forward": (1, +1), "backward": (1, -1), "right": (0, +1), "left": (0, -1)}

    def calculate_evidence(self, motion, requirement, case):
        want = str(self.expected_of(requirement)).lower()
        if want not in self.AXES:
            return {"unsupported": f"direction {want!r}"}
        m = P.as_motion(motion)
        a = self.analyse(m)
        ctx = E.requirement_context(requirement, case)
        window, source = (0, len(m)), "whole_clip"
        if ctx["action_req"] is not None:
            _, evs = E.events_for_action(a, ctx["action_req"], case)
            if evs:
                window, source = (min(e.start for e in evs), max(e.end for e in evs)), "action_events"
        ref = window[0] if ctx["step"] else 0          # first step: facing at the clip start
        # smoothed motion (events.EVENT_PARAMS["smooth_window"]) so one jittery frame
        # cannot tilt the reference facing
        lat, fwd = P.displacement_in_frame(a.motion, window[0], window[1], frame_index=ref)
        axis, sign = self.AXES[want]
        req = (lat, fwd)[axis] * sign
        d = m[window[1] - 1, 0] - m[window[0], 0]
        return {"window": list(window), "window_source": source, "reference_frame": int(ref),
                "reference_yaw_deg": float(a.yaw[ref]), "step": ctx["step"],
                "lateral_right_m": lat, "forward_m": fwd, "world_dx_m": float(d[0]), "world_dz_m": float(d[2]),
                "required_displacement_m": float(req), "score": float(req)}

    def decide(self, ev, expected):
        need = self.thresholds["min_displacement"]
        return verdict(ev["required_displacement_m"] >= need), (
            f"body frame (ref frame {ev['reference_frame']}): right {ev['lateral_right_m']:+.2f} m, "
            f"forward {ev['forward_m']:+.2f} m; required {expected} >= {need} m")


register_evaluator("AttributeEvaluator", AttributeEvaluator)
register_evaluator("BodyFrameDirectionEvaluator", BodyFrameDirectionEvaluator)
