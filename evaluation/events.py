"""
events.py — shared event layer (whole team)
===========================================

Turns a standardised motion into a list of timed events:

    Event(kind, start, end, side, attrs)      frames, end exclusive, 20 fps

kinds: walk, turn, jump, kick, reach, raise_hand, raise_hands

Every evaluator that needs "when did X happen" (action, count, order,
simultaneous, turn_direction, leg_direction, attribute) reads the SAME events,
so e.g. OrderEvaluator and CountEvaluator can never disagree about where the
jumps are.

Event detection is deliberately lenient (EVENT_PARAMS). The PASS / FAIL
strictness lives in each evaluator's CURRENT_THRESHOLDS, which are calibrated
against Human Gold. EVENT_PARAMS are only changed via a reviewed PR because
they affect every evaluator.

The second half of the file links requirements to events using the v1.2
definition fields: applies_to, event_sequence, event_group.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import numpy as np

import primitives as P

# ============================================================================
# Detection parameters (lenient; distances in metres for a 1.7 m person and
# scaled by body height; rates per second)
# ============================================================================
EVENT_PARAMS = {
    # pre-filter: centred moving average over frames (removes per-frame jitter)
    "smooth_window": 5,
    # walk
    "walk_min_speed": 0.30,        # m/s, smoothed root speed          (plan §5: 0.3 m/s)
    "walk_min_frames": 20,         # 1 s                               (plan §5: >= 1 s)
    "walk_merge_gap": 6,
    "step_min_amplitude": 0.12,    # inter-foot oscillation, x leg length
    "walk_min_steps": 2,           # alternations of the leading foot  (plan §5: feet alternate)
    # turn
    "turn_min_rate": 25.0,         # deg/s
    "turn_min_angle": 30.0,        # deg, net yaw change of one turn event
    "turn_merge_gap": 6,
    # jump
    "jump_clearance": 0.05,        # lowest foot above ground (m) = airborne
    "jump_min_peak_clearance": 0.07,
    "jump_min_flight_frames": 2,   # plan §5: both feet off >= 2 frames
    "jump_merge_gap": 3,
    "running_flight_max_frames": 4,
    # legs
    "kick_min_deviation": 0.6,     # x leg length
    "kick_min_clearance": 0.15,    # kicking foot above ground (m)
    "kick_min_frames": 3,
    # arms
    "reach_min_deviation": 0.7,    # x arm length
    "reach_min_frames": 4,
    "raise_min_above_shoulder": 0.05,  # wrist above shoulder (m)
    "raise_min_frames": 5,         # plan §5: wrist above shoulder >= 0.25 s
    "both_min_overlap_frames": 5,
}

ACTION_TO_EVENT = {
    "walk": "walk", "walks": "walk", "walking": "walk",
    "turn": "turn", "turns": "turn", "turning": "turn",
    "jump": "jump", "jumps": "jump", "hop": "jump", "hops": "jump",
    "kick": "kick", "kicks": "kick",
    "reach": "reach", "reaches": "reach",
    "raise_hand": "raise_hand", "raise": "raise_hand", "raise_arm": "raise_hand",
    "raise_hands": "raise_hands", "hands": "raise_hands", "raise_arms": "raise_hands",
}


@dataclass
class Event:
    kind: str
    start: int
    end: int                      # exclusive
    side: str | None = None       # left / right / both / None
    attrs: dict = field(default_factory=dict)

    @property
    def duration(self):
        return self.end - self.start

    @property
    def center(self):
        return 0.5 * (self.start + self.end - 1)

    def overlap(self, other):
        return max(0, min(self.end, other.end) - max(self.start, other.start))

    def to_dict(self):
        d = asdict(self)
        d["start_s"] = round(self.start / P.FPS, 2)
        d["end_s"] = round(self.end / P.FPS, 2)
        return d


# ============================================================================
# Detectors
# ============================================================================
def _hysteresis_crossings(signal, threshold):
    """Number of sign changes of a signal, only counting excursions beyond ±threshold."""
    state, changes = 0, 0
    for x in signal:
        s = 1 if x > threshold else -1 if x < -threshold else 0
        if s and state and s != state:
            changes += 1
        if s:
            state = s
    return changes


def detect_walk(m, prm, scale, flights):
    speed = P.root_track(m)["speed"]
    moving = speed >= prm["walk_min_speed"]
    for f in flights:                          # hops are not steps
        moving[f.start:f.end] = False
    events = []
    feet = m[:, P.L_ANKLE][:, [0, 2]] - m[:, P.R_ANKLE][:, [0, 2]]
    for a, b in P.segments(moving, prm["walk_min_frames"], prm["walk_merge_gap"]):
        xz = m[a:b, P.PELVIS][:, [0, 2]]
        net = xz[-1] - xz[0]
        path = float(np.linalg.norm(np.diff(xz, axis=0), axis=1).sum())
        direction = net / (np.linalg.norm(net) + 1e-8)
        proj = feet[a:b] @ direction
        proj = proj - P.smooth(proj, 21)             # remove slow drift
        steps = _hysteresis_crossings(proj, prm["step_min_amplitude"] * scale["leg_length"] / 2)
        if steps < prm["walk_min_steps"]:
            continue
        events.append(Event("walk", a, b, None, {
            "net_displacement_m": float(np.linalg.norm(net)),
            "path_length_m": path,
            "mean_speed_mps": float(speed[a:b].mean()),
            "steps": int(steps),
            "cadence_steps_per_s": float(steps / ((b - a) / P.FPS)),
        }))
    return events


def detect_turns(m, prm):
    yaw = P.heading_yaw(m)
    rate = P.smooth(np.gradient(yaw) * P.FPS, 5) if len(m) > 1 else np.zeros(len(m))
    events = []
    for sign, side in ((1, "left"), (-1, "right")):
        mask = sign * rate >= prm["turn_min_rate"]
        for a, b in P.segments(mask, 2, prm["turn_merge_gap"]):
            # extend to where the rotation actually starts / stops
            a0 = a
            while a0 > 0 and sign * rate[a0 - 1] > 0.3 * prm["turn_min_rate"]:
                a0 -= 1
            b0 = b
            while b0 < len(m) and sign * rate[b0] > 0.3 * prm["turn_min_rate"]:
                b0 += 1
            delta = float(yaw[b0 - 1] - yaw[a0])
            if sign * delta < prm["turn_min_angle"]:
                continue
            events.append(Event("turn", a0, b0, side, {
                "yaw_change_deg": delta,                 # + = left
                "peak_rate_deg_s": float(np.max(np.abs(rate[a0:b0]))),
            }))
    return sorted(events, key=lambda e: e.start)


def detect_jumps(m, prm, scale, contacts_h):
    s = scale["scale"]
    ground = min(contacts_h["left"].min(), contacts_h["right"].min())
    lowest = np.minimum(contacts_h["left"], contacts_h["right"]) - ground
    airborne = lowest > prm["jump_clearance"] * s
    pelvis = m[:, P.PELVIS, 1]
    stand = np.median(pelvis[~airborne]) if (~airborne).any() else np.median(pelvis)
    events = []
    for a, b in P.segments(airborne, prm["jump_min_flight_frames"], prm["jump_merge_gap"]):
        peak = float(lowest[a:b].max())
        if peak < prm["jump_min_peak_clearance"] * s:
            continue
        events.append(Event("jump", a, b, None, {
            "flight_frames": int(b - a),
            "peak_clearance_m": peak,
            "pelvis_rise_m": float(pelvis[a:b].max() - stand),
        }))
    return events


def detect_kicks(m, prm, scale, contacts_h, flights):
    s = scale["scale"]
    ground = min(contacts_h["left"].min(), contacts_h["right"].min())
    airborne = np.zeros(len(m), bool)
    for f in flights:
        airborne[f.start:f.end] = True
    right, up, forward = P.body_frame(m)
    events = []
    for side, other in (("left", "right"), ("right", "left")):
        dev = P.limb_deviation(m, side, "leg")
        clear = contacts_h[side] - ground
        support = contacts_h[other] - ground
        mask = (dev >= prm["kick_min_deviation"]) & (clear >= prm["kick_min_clearance"] * s) \
            & (support < 2 * prm["jump_clearance"] * s) & ~airborne
        vec = P.limb_vectors(m, side, "leg")
        for a, b in P.segments(mask, prm["kick_min_frames"], 2):
            k = a + int(np.argmax(dev[a:b]))
            v = vec[k] / scale["leg_length"]
            outward = (v @ right[k]) * (1 if side == "right" else -1)
            events.append(Event("kick", a, b, side, {
                "peak_frame": int(k),
                "peak_deviation": float(dev[k]),
                "peak_clearance_m": float(clear[k]),
                "outward": float(outward),                 # + = away from the body midline
                "forward": float(v @ forward[k]),
                "up": float(v @ up[k] + 1.0),              # height gained vs hanging, x leg length
            }))
    return sorted(events, key=lambda e: e.start)


def detect_arm_events(m, prm, scale):
    s = scale["scale"]
    events = {"reach": [], "raise_hand": []}
    raised = {}
    for side in ("left", "right"):
        j = P.SIDE_JOINTS[side]
        dev = P.limb_deviation(m, side, "arm")
        above = m[:, j["wrist"], 1] - m[:, j["shoulder"], 1]
        for a, b in P.segments(dev >= prm["reach_min_deviation"], prm["reach_min_frames"], 3):
            k = a + int(np.argmax(dev[a:b]))
            events["reach"].append(Event("reach", a, b, side, {"peak_frame": int(k),
                                                             "peak_deviation": float(dev[k])}))
        mask = above >= prm["raise_min_above_shoulder"] * s
        raised[side] = mask
        for a, b in P.segments(mask, prm["raise_min_frames"], 3):
            events["raise_hand"].append(Event("raise_hand", a, b, side, {
                "peak_wrist_above_shoulder_m": float(above[a:b].max())}))
    both = [Event("raise_hands", a, b, "both", {})
            for a, b in P.segments(raised["left"] & raised["right"], prm["both_min_overlap_frames"], 3)]
    return events["reach"], sorted(events["raise_hand"], key=lambda e: e.start), both


# ============================================================================
# Analysis (cached per motion)
# ============================================================================
class MotionAnalysis:
    """All primitives + events of one motion. Build with analyse(motion)."""

    def __init__(self, motion, params=None):
        m = P.as_motion(motion)
        if not np.isfinite(m).all():
            raise ValueError("Motion contains NaN or Inf values.")
        self.params = {**EVENT_PARAMS, **(params or {})}
        w = int(self.params["smooth_window"])
        if w > 1:
            m = P.smooth(m.reshape(len(m), -1), w).reshape(m.shape)
        self.motion = m
        self.scale = P.body_scale(m)
        self.yaw = P.heading_yaw(m)
        self.root = P.root_track(m)
        self.foot_h = P.foot_heights(m)
        prm = self.params

        jumps = detect_jumps(m, prm, self.scale, self.foot_h)
        walks = detect_walk(m, prm, self.scale, jumps)
        # short flights inside a walk are running strides, not jumps
        jumps = [j for j in jumps
                 if not (j.duration <= prm["running_flight_max_frames"]
                         and any(w.start <= j.start and j.end <= w.end for w in walks))]
        reach, raise_hand, raise_hands = detect_arm_events(m, prm, self.scale)
        self.events = {
            "walk": walks,
            "turn": detect_turns(m, prm),
            "jump": jumps,
            "kick": detect_kicks(m, prm, self.scale, self.foot_h, jumps),
            "reach": reach,
            "raise_hand": raise_hand,
            "raise_hands": raise_hands,
        }

    def get(self, kind, side=None):
        evs = self.events.get(kind, [])
        if side in ("left", "right"):
            evs = [e for e in evs if e.side == side]
        return evs

    def summary(self):
        return {k: [e.to_dict() for e in v] for k, v in self.events.items()}


_CACHE = {}


def analyse(motion, params=None):
    """Cached MotionAnalysis: every evaluator on the same motion reuses one analysis."""
    m = np.ascontiguousarray(np.asarray(motion, dtype=np.float64))
    key = (hashlib.sha1(m.tobytes()).hexdigest(), m.shape, repr(sorted((params or {}).items())))
    if key not in _CACHE:
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[key] = MotionAnalysis(m, params)
    return _CACHE[key]


# ============================================================================
# Requirement linking (definition v1.2: applies_to / event_sequence / event_group)
# ============================================================================
def requirements_by_id(case):
    return {r.get("id"): r for r in (case or {}).get("requirements", []) if r.get("id")}


def linked_action(requirement, case):
    """The action requirement a modifier refers to: applies_to first, else the nearest preceding action."""
    by_id = requirements_by_id(case)
    for rid in requirement.get("applies_to") or []:
        r = by_id.get(rid)
        if r and r.get("type") == "action":
            return r
    reqs = (case or {}).get("requirements", [])
    idx = next((i for i, r in enumerate(reqs) if r is requirement or r == requirement), None)
    if idx is not None:
        for r in reversed(reqs[:idx]):
            if r.get("type") == "action":
                return r
    return None


def modifiers_of(action_req, case, req_type):
    """Values of requirements of req_type that apply to this action (e.g. its body_side)."""
    rid = (action_req or {}).get("id")
    return [r.get("value") for r in (case or {}).get("requirements", [])
            if r.get("type") == req_type and rid in (r.get("applies_to") or [])]


def event_kind(action_value):
    return ACTION_TO_EVENT.get(str(action_value).lower().strip())


def events_for_action(analysis, action_req, case, use_side=True):
    """Events matching an action requirement; filtered by its body_side if one applies."""
    if action_req is None:
        return None, []
    kind = event_kind(action_req.get("value"))
    if kind is None:
        return None, []
    side = None
    if use_side:
        sides = modifiers_of(action_req, case, "body_side")
        side = sides[0] if sides else None
    return kind, analysis.get(kind, side if side in ("left", "right") else None)


def events_for_entry(analysis, entry, case, use_side=True):
    """Events for one entry of event_sequence / event_group ({"event", "requirement_ids"})."""
    by_id = requirements_by_id(case)
    for rid in entry.get("requirement_ids") or []:
        r = by_id.get(rid)
        if r and r.get("type") == "action":
            return events_for_action(analysis, r, case, use_side)
    kind = event_kind(entry.get("event"))
    return kind, (analysis.get(kind) if kind else [])


def entries_for(requirement, case, key):
    """event_sequence / event_group entries; falls back to value + applies_to (older definitions)."""
    entries = requirement.get(key)
    if entries:
        return entries
    ids = requirement.get("applies_to") or []
    values = requirement.get("value") or []
    return [{"event": v, "requirement_ids": [ids[i]] if i < len(ids) else []}
            for i, v in enumerate(values)]


def detect_events(motion, params=None):
    """All events of a motion as one list sorted by start frame (plan §3 ②)."""
    a = analyse(motion, params)
    return sorted((e for evs in a.events.values() for e in evs), key=lambda e: (e.start, e.kind))


def requirement_context(requirement, case):
    """
    Which action a requirement refers to, and how that action is constrained
    (plan §3: requirement_context; lives here so common.py stays untouched).

    Returns dict:
      action_req   the action requirement (via applies_to, else the nearest preceding action)
      action       its value, e.g. "kick";  event_kind: event type used for it, or None
      body_side    body_side value applying to that action, or None
      step         position of that action in the prompt's order requirement (0-based), or None
      previous     action requirement of the previous step, or None
    """
    action_req = requirement if requirement.get("type") == "action" else linked_action(requirement, case)
    ctx = {"action_req": action_req, "action": None, "event_kind": None,
           "body_side": None, "step": None, "previous": None}
    if action_req is None:
        return ctx
    ctx["action"] = action_req.get("value")
    ctx["event_kind"] = event_kind(ctx["action"])
    sides = modifiers_of(action_req, case, "body_side")
    ctx["body_side"] = sides[0] if sides else None
    rid = action_req.get("id")
    by_id = requirements_by_id(case)
    for r in (case or {}).get("requirements", []):
        if r.get("type") != "order":
            continue
        for k, entry in enumerate(entries_for(r, case, "event_sequence")):
            if rid in (entry.get("requirement_ids") or []):
                ctx["step"] = k
                if k > 0:
                    prev_ids = entries_for(r, case, "event_sequence")[k - 1].get("requirement_ids") or []
                    ctx["previous"] = next((by_id[i] for i in prev_ids if i in by_id), None)
    return ctx
