"""
leg_direction / arm_direction -> LimbGeometryEvaluator
torso_direction -> TorsoGeometryEvaluator

Pilot only exercises leg_direction (e.g. "kicks the left leg out to the side").
arm_direction and torso_direction are implemented and unit-tested against
synthetic motions, but there's no Pilot data to calibrate them against yet.

LimbGeometry: pick the limb (leg for leg_direction, arm for arm_direction),
pick the side (body_side applying to the same action, otherwise whichever
limb is more active), then find the peak frame - the frame with the biggest
deviation from the hanging pose inside the action's events (e.g. during the
kick, if one is detected). At that frame, project the effector's displacement
from the hanging pose (0, -L, 0) onto the body frame and divide by limb length
to get outward/inward/forward/backward/up/down components. PASS when the
requested component clears min_extent AND beats the next-largest horizontal
component by min_dominance.

TorsoGeometry: pelvis，neck vector vs vertical, lean direction read off the
hip frame. PASS when the tilt clears min_tilt_deg in the requested direction
and dominates the other axis by min_dominance.
"""

import numpy as np

import events as E
import primitives as P
from common import register_evaluator
from rule_base import current, verdict

RuleEvaluator = current()  # rebind to whatever common.py is loaded right now

# map every direction word a prompt might use to one of the six canonical names
DIRECTION_ALIASES = {
    "side": "outward", "sideways": "outward", "out": "outward", "outward": "outward", "lateral": "outward",
    "inward": "inward", "across": "inward",
    "forward": "forward", "front": "forward", "forwards": "forward",
    "backward": "backward", "back": "backward", "backwards": "backward", "behind": "backward",
    "up": "up", "upward": "up", "upwards": "up", "overhead": "up",
    "down": "down", "downward": "down",
}
HORIZONTAL = ("outward", "inward", "forward", "backward")  # up/down aren't in the competition for dominance


class LimbGeometryEvaluator(RuleEvaluator):
    EVALUATOR_NAME = "LimbGeometryEvaluator"
    PROVISIONAL_THRESHOLDS = {"min_extent": 0.35, "min_dominance": 1.2}
    CURRENT_THRESHOLDS = {"min_extent": 0.2, "min_dominance": 1.2}
    CURRENT_THRESHOLD_STATUS = "frozen_v1.0"

    def calculate_evidence(self, motion, requirement, case):
        rtype = requirement.get("type")
        limb = "leg" if rtype == "leg_direction" else "arm"  # same evaluator handles both, only the limb changes
        want = DIRECTION_ALIASES.get(str(self.expected_of(requirement)).lower())
        if want is None:
            return {"unsupported": f"{rtype} value {self.expected_of(requirement)!r}"}
        m = P.as_motion(motion)
        a = self.analyse(m)  # gives us scale.leg_length / scale.arm_length below
        ctx = E.requirement_context(requirement, case)

        # per-frame distance of each side's limb from its hanging-straight-down rest pose
        dev = {s: P.limb_deviation(m, s, limb) for s in ("left", "right")}
        side, source = ctx["body_side"], "body_side"
        if side not in ("left", "right"):
            
            # prompt didn't say which side - just go with whichever one actually moved
            side = max(dev, key=lambda s: np.percentile(dev[s], 95))
            source = "most_active"

        window, window_source = (0, len(m)), "whole_clip"
        if ctx["action_req"] is not None:
            
            # prompt also names an action (e.g. "kick") - narrow the search to that action's window
            _, evs = E.events_for_action(a, ctx["action_req"], case)
            evs = [e for e in evs if e.side in (side, None, "both")]
            if evs:
                best = max(evs, key=lambda e: dev[side][e.start:e.end].max())
                window, window_source = (best.start, best.end), f"event:{best.kind}"
        k = window[0] + int(np.argmax(dev[side][window[0]:window[1]]))  # frame of peak deviation in that window

        length = a.scale["leg_length" if limb == "leg" else "arm_length"]
        
        # effector's offset from the hanging pose, in metres
        v = P.limb_vectors(m, side, limb)[k] - np.array([0.0, -length, 0.0])
        right, up, forward = (x[k] for x in P.body_frame(m))
        
        # project onto the body frame and normalize by limb length so the result is unit-free;
        # flip the lateral sign so "outward" is always positive, whichever side we picked
        lat = float(v @ right) * (1 if side == "right" else -1) / length
        fwd = float(v @ forward) / length
        vert = float(v @ up) / length
        
        # split each signed axis into its two one-sided components
        comps = {"outward": max(lat, 0), "inward": max(-lat, 0), "forward": max(fwd, 0),
                 "backward": max(-fwd, 0), "up": max(vert, 0), "down": max(-vert, 0)}
        rivals = [c for c in HORIZONTAL if c != want]
        competitor = max(comps[c] for c in rivals)    # strongest other horizontal direction, for the dominance check
        return {"limb": limb, "side": side, "side_source": source, "direction": want,
                "peak_frame": k, "peak_deviation": float(dev[side][k]), "window": list(window),
                "window_source": window_source, "components": comps,
                "competitor": float(competitor), "score": float(comps[want])}

    def decide(self, ev, expected):
        t = self.thresholds
        c = ev["components"]
        want = ev["direction"]
        
        # pass only if the requested direction reaches far enough AND clearly beats the runner-up
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
        # hip-to-hip vector, flattened to the horizontal plane - this is the body's "right" axis per frame
        hips_right = m[:, P.R_HIP] - m[:, P.L_HIP]
        hips_right[:, 1] = 0
        hips_right /= np.linalg.norm(hips_right, axis=1, keepdims=True) + 1e-8
        fwd = np.cross(hips_right, P.UP)  # forward is perpendicular to right and up
        spine = m[:, P.NECK] - m[:, P.PELVIS]  # pelvis-to-neck vector gives the lean direction
        spine /= np.linalg.norm(spine, axis=1, keepdims=True) + 1e-8
        comp = {
            "forward": np.degrees(np.arcsin(np.clip((spine * fwd).sum(1), -1, 1))),
            "right": np.degrees(np.arcsin(np.clip((spine * hips_right).sum(1), -1, 1))),
        }
        # mirror each axis to get all four lean directions from the two angle series
        series = {"forward": comp["forward"], "backward": -comp["forward"],
                  "right": comp["right"], "left": -comp["right"]}
        k = int(np.argmax(series[want]))  # frame of peak lean in the requested direction
        # compare against the other axis only - skip the opposite direction, which is just -want by construction
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
