"""
eval_limb.py — leg_direction / arm_direction (LimbGeometryEvaluator)
                torso_direction (TorsoGeometryEvaluator)
====================================================================

Owner: B (plan §8)

Pilot: C2-09 / C2-10 "kicks the left / right leg out to the side" -> leg_direction = side.
arm_direction and torso_direction do not occur in the Pilot; they are implemented
and unit-tested on synthetic motions but cannot be calibrated until Main.

LimbGeometry (plan §4)
  Limb: leg for leg_direction, arm for arm_direction.
  Side: body_side applying to the same action; else the more active limb.
  Peak frame: largest deviation from the hanging pose (inside the action's
  events, e.g. the kick, when they exist).
  At the peak, the effector displacement from the hanging pose (0, -L, 0),
  in the body frame, / limb length:
      outward (away from the midline), inward, forward, backward, up, down
  PASS  requested component >= min_extent  and
        requested >= min_dominance × largest competing HORIZONTAL component
        (for up / down: × the largest horizontal component)

TorsoGeometry
  pelvis -> neck vector vs vertical; lean direction in the hip frame.
  PASS  tilt >= min_tilt_deg in the requested direction, dominant by min_dominance.
"""

import numpy as np

import events as E
import primitives as P
from common import register_evaluator
from rule_base import current, verdict

RuleEvaluator = current()   # base class bound to the loaded common.py (safe after reload)

DIRECTION_ALIASES = {
    "side": "outward", "sideways": "outward", "out": "outward", "outward": "outward", "lateral": "outward",
    "inward": "inward", "across": "inward",
    "forward": "forward", "front": "forward", "forwards": "forward",
    "backward": "backward", "back": "backward", "backwards": "backward", "behind": "backward",
    "up": "up", "upward": "up", "upwards": "up", "overhead": "up",
    "down": "down", "downward": "down",
}
HORIZONTAL = ("outward", "inward", "forward", "backward")


class LimbGeometryEvaluator(RuleEvaluator):
    EVALUATOR_NAME = "LimbGeometryEvaluator"
    PROVISIONAL_THRESHOLDS = {"min_extent": 0.35, "min_dominance": 1.2}
    CURRENT_THRESHOLDS = dict(PROVISIONAL_THRESHOLDS)
    CURRENT_THRESHOLD_STATUS = "provisional_not_frozen"

    def calculate_evidence(self, motion, requirement, case):
        rtype = requirement.get("type")
        limb = "leg" if rtype == "leg_direction" else "arm"
        want = DIRECTION_ALIASES.get(str(self.expected_of(requirement)).lower())
        if want is None:
            return {"unsupported": f"{rtype} value {self.expected_of(requirement)!r}"}
        m = P.as_motion(motion)
        a = self.analyse(m)
        ctx = E.requirement_context(requirement, case)

        dev = {s: P.limb_deviation(m, s, limb) for s in ("left", "right")}
        side, source = ctx["body_side"], "body_side"
        if side not in ("left", "right"):
            side = max(dev, key=lambda s: np.percentile(dev[s], 95))
            source = "most_active"

        window, window_source = (0, len(m)), "whole_clip"
        if ctx["action_req"] is not None:
            _, evs = E.events_for_action(a, ctx["action_req"], case)
            evs = [e for e in evs if e.side in (side, None, "both")]
            if evs:
                best = max(evs, key=lambda e: dev[side][e.start:e.end].max())
                window, window_source = (best.start, best.end), f"event:{best.kind}"
        k = window[0] + int(np.argmax(dev[side][window[0]:window[1]]))

        length = a.scale["leg_length" if limb == "leg" else "arm_length"]
        v = P.limb_vectors(m, side, limb)[k] - np.array([0.0, -length, 0.0])
        right, up, forward = (x[k] for x in P.body_frame(m))
        lat = float(v @ right) * (1 if side == "right" else -1) / length
        fwd = float(v @ forward) / length
        vert = float(v @ up) / length
        comps = {"outward": max(lat, 0), "inward": max(-lat, 0), "forward": max(fwd, 0),
                 "backward": max(-fwd, 0), "up": max(vert, 0), "down": max(-vert, 0)}
        rivals = [c for c in HORIZONTAL if c != want]
        competitor = max(comps[c] for c in rivals)
        return {"limb": limb, "side": side, "side_source": source, "direction": want,
                "peak_frame": k, "peak_deviation": float(dev[side][k]), "window": list(window),
                "window_source": window_source, "components": comps,
                "competitor": float(competitor), "score": float(comps[want])}

    def decide(self, ev, expected):
        t = self.thresholds
        c = ev["components"]
        want = ev["direction"]
        ok = c[want] >= t["min_extent"] and c[want] >= t["min_dominance"] * ev["competitor"]
        return verdict(ok), (f"{ev['side']} {ev['limb']} at frame {ev['peak_frame']}: {want}={c[want]:.2f} "
                             f"(need >= {t['min_extent']}), best other horizontal={ev['competitor']:.2f}")


class TorsoGeometryEvaluator(RuleEvaluator):
    EVALUATOR_NAME = "TorsoGeometryEvaluator"
    PROVISIONAL_THRESHOLDS = {"min_tilt_deg": 15.0, "min_dominance": 1.5}
    CURRENT_THRESHOLDS = dict(PROVISIONAL_THRESHOLDS)
    CURRENT_THRESHOLD_STATUS = "provisional_not_frozen"
    ALIASES = {"forward": "forward", "front": "forward", "down": "forward",
               "backward": "backward", "back": "backward",
               "left": "left", "right": "right"}

    def calculate_evidence(self, motion, requirement, case):
        want = self.ALIASES.get(str(self.expected_of(requirement)).lower())
        if want is None:
            return {"unsupported": f"torso_direction value {self.expected_of(requirement)!r}"}
        m = P.as_motion(motion)
        hips_right = m[:, P.R_HIP] - m[:, P.L_HIP]
        hips_right[:, 1] = 0
        hips_right /= np.linalg.norm(hips_right, axis=1, keepdims=True) + 1e-8
        fwd = np.cross(hips_right, P.UP)
        spine = m[:, P.NECK] - m[:, P.PELVIS]
        spine /= np.linalg.norm(spine, axis=1, keepdims=True) + 1e-8
        comp = {
            "forward": np.degrees(np.arcsin(np.clip((spine * fwd).sum(1), -1, 1))),
            "right": np.degrees(np.arcsin(np.clip((spine * hips_right).sum(1), -1, 1))),
        }
        series = {"forward": comp["forward"], "backward": -comp["forward"],
                  "right": comp["right"], "left": -comp["right"]}
        k = int(np.argmax(series[want]))
        others = [abs(series[d][k]) for d in series if d not in (want,) and
                  {d, want} not in ({"forward", "backward"}, {"left", "right"})]
        return {"direction": want, "peak_frame": k, "tilt_deg": float(series[want][k]),
                "other_axis_deg": float(max(others)), "score": float(series[want][k])}

    def decide(self, ev, expected):
        t = self.thresholds
        ok = ev["tilt_deg"] >= t["min_tilt_deg"] and ev["tilt_deg"] >= t["min_dominance"] * ev["other_axis_deg"]
        return verdict(ok), (f"torso {ev['direction']} tilt {ev['tilt_deg']:.0f} deg "
                             f"(need >= {t['min_tilt_deg']}), other axis {ev['other_axis_deg']:.0f} deg")


register_evaluator("LimbGeometryEvaluator", LimbGeometryEvaluator)
register_evaluator("TorsoGeometryEvaluator", TorsoGeometryEvaluator)
