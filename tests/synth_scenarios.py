"""
Synthetic, physically plausible test motions for the v1.2 Pilot prompts.

A motion is composed from a timeline of base segments (stand / walk / turn /
jump / kick) plus arm overlays (raise / reach) in the character's local frame
(facing +Z, right = +X), then placed in the world with the current heading.
Feet are pinned to the ground except during jumps.

Used only by tests: every evaluator must PASS the correct scenario and FAIL
the controlled negatives (left/right swapped, order reversed, wrong count ...).
"""
import numpy as np

import synth_motion as sm

FPS = 20
REST = sm.REST
L_LEG, R_LEG = (1, 4, 7, 10), (2, 5, 8, 11)
L_ARM, R_ARM = (16, 18, 20), (17, 19, 21)


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _pose_limb(pose, chain, R):
    root = pose[chain[0]].copy()
    for j in chain[1:]:
        pose[j] = root + R @ (REST[j] - REST[chain[0]])


def leg(pose, side, flex=0.0, abduct=0.0):
    """flex > 0 = foot forward (+Z); abduct > 0 = foot outward."""
    chain = L_LEG if side == "left" else R_LEG
    out = -1 if side == "left" else 1
    _pose_limb(pose, chain, _rz(out * abduct) @ _rx(-flex))


def arm(pose, side, flex=0.0, abduct=0.0):
    chain = L_ARM if side == "left" else R_ARM
    out = -1 if side == "left" else 1
    _pose_limb(pose, chain, _rz(out * abduct) @ _rx(-flex))


def smoothstep(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def bump(u):
    return np.sin(np.pi * np.clip(u, 0, 1)) ** 2


# ----------------------------------------------------------------------------- composer
def compose(base, overlays=(), total=None):
    """
    base     : list of (kind, frames, params) played one after another
               kinds: stand | walk(dir, speed, cadence) | turn(deg, + = left) | jump(height)
                      | kick(side, how) ; how = side | forward
    overlays : list of (kind, start, end, params) arm actions
               kinds: raise(side: left/right/both) | reach_across(side) | reach_same(side)
    """
    frames = []
    for kind, n, prm in base:
        frames += [(kind, i, n, prm) for i in range(n)]
    if total is not None:
        frames = (frames + [("stand", 0, 1, {})] * total)[:total]
    T = len(frames)
    motion = np.zeros((T, 22, 3))
    pos = np.zeros(3)
    yaw = 0.0
    walk_phase = 0.0
    for t, (kind, i, n, prm) in enumerate(frames):
        pose = REST.copy()
        lift = 0.0
        u = i / max(n - 1, 1)
        if kind == "walk":
            cadence = prm.get("cadence", 1.8)         # steps per second
            walk_phase += np.pi * cadence / FPS
            s = np.sin(walk_phase)
            direction = prm.get("dir", "forward")
            amp = np.radians(prm.get("amp", 25))
            if direction in ("forward", "backward"):
                leg(pose, "left", flex=amp * s)
                leg(pose, "right", flex=-amp * s)
                arm(pose, "left", flex=-0.8 * amp * s)
                arm(pose, "right", flex=0.8 * amp * s)
            else:
                lead, trail = ("left", "right") if direction == "left" else ("right", "left")
                leg(pose, lead, abduct=amp * max(s, 0))
                leg(pose, trail, abduct=amp * max(-s, 0))
            local = {"forward": (0, 1), "backward": (0, -1), "left": (-1, 0), "right": (1, 0)}[direction]
            v = prm.get("speed", 1.0) / FPS
            right = np.array([np.cos(yaw), 0, np.sin(yaw)])
            fwd = np.array([-np.sin(yaw), 0, np.cos(yaw)])
            pos = pos + v * (local[0] * right + local[1] * fwd)
        elif kind == "turn":
            yaw += np.radians(prm["deg"]) * (smoothstep((i + 1) / n) - smoothstep(i / n))
            s = np.sin(np.pi * 2 * u)
            leg(pose, "left", flex=np.radians(8) * s)
            leg(pose, "right", flex=-np.radians(8) * s)
        elif kind == "jump":
            flight = prm.get("flight", 8)
            start = (n - flight) // 2
            k = i - start
            if 0 <= k < flight:
                w = (k + 0.5) / flight
                lift = 4 * prm.get("height", 0.3) * w * (1 - w)
        elif kind == "kick":
            e = bump(u)
            if prm.get("how", "side") == "side":
                leg(pose, prm["side"], abduct=np.radians(70) * e)
            else:
                leg(pose, prm["side"], flex=np.radians(80) * e)

        for okind, a, b, oprm in overlays:
            if not (a <= t < b):
                continue
            e = smoothstep(min((t - a) / 6, (b - 1 - t) / 6, 1.0))
            sides = ["left", "right"] if oprm.get("side") == "both" else [oprm["side"]]
            for side in sides:
                if okind == "raise":
                    arm(pose, side, abduct=np.radians(160) * e)
                elif okind in ("reach_across", "reach_same"):
                    chain = L_ARM if side == "left" else R_ARM
                    across = okind == "reach_across"
                    target_sh = (R_ARM if side == "left" else L_ARM)[0] if across else chain[0]
                    goal = REST[target_sh] + np.array([0, -0.06, 0.10])
                    wrist, elbow = chain[2], chain[1]
                    pose[wrist] = (1 - e) * REST[wrist] + e * goal
                    mid = 0.5 * (pose[chain[0]] + pose[wrist]) + np.array([0, -0.08, 0.15]) * e
                    pose[elbow] = (1 - e) * REST[elbow] + e * mid

        # pin the lower foot to the ground, then add flight
        feet_low = min(pose[[7, 8, 10, 11], 1].min(), 10)
        pose[:, 1] += -feet_low + REST[10, 1] + lift
        right = np.array([np.cos(yaw), 0, np.sin(yaw)])
        fwd = np.array([-np.sin(yaw), 0, np.cos(yaw)])
        up = np.array([0, 1.0, 0])
        motion[t] = pos + pose[:, [0]] * right + pose[:, [1]] * up + pose[:, [2]] * fwd
    return sm.standardise(motion)


# ----------------------------------------------------------------------------- scenarios
def S(n):
    return ("stand", n, {})


def W(n, direction="forward", speed=1.0, cadence=1.8):
    return ("walk", n, {"dir": direction, "speed": speed, "cadence": cadence})


def TURN(n, deg):
    return ("turn", n, {"deg": deg})


def J(n=14, height=0.3):
    return ("jump", n, {"height": height})


def KICK(side, how="side", n=30):
    return ("kick", n, {"side": side, "how": how})


def pilot_motion(prompt_id):
    """A motion that satisfies every requirement of the prompt."""
    ps = {
        "C1-01": lambda: compose([S(10), W(80), S(10)]),
        "C1-02": lambda: compose([S(10), W(80, "backward", 0.8), S(10)]),
        "C1-06": lambda: compose([S(20), TURN(40, 90), S(40)]),
        "C1-07": lambda: compose([S(20), TURN(40, -90), S(40)]),
        "C2-01": lambda: compose([S(10), W(80, "left", 0.8), S(10)]),
        "C2-02": lambda: compose([S(10), W(80, "right", 0.8), S(10)]),
        "C2-09": lambda: compose([S(50), KICK("left"), S(68)]),
        "C2-10": lambda: compose([S(50), KICK("right"), S(68)]),
        "C2-19": lambda: compose([S(196)], [("reach_across", 40, 150, {"side": "left"})]),
        "C2-20": lambda: compose([S(196)], [("reach_across", 40, 150, {"side": "right"})]),
        "C3-01": lambda: compose([S(10), W(128, speed=0.55, cadence=1.4), S(10)]),
        "C3-02": lambda: compose([S(10), W(128, speed=1.7, cadence=2.4), S(10)]),
        "C4-01": lambda: compose([S(20), J(), S(20), TURN(40, -90), S(54)]),
        "C4-02": lambda: compose([S(20), TURN(40, -90), S(20), J(), S(54)]),
        "C4-09": lambda: compose([S(20), J(), S(16), J(), S(84)]),
        "C4-10": lambda: compose([S(20), J(), S(16), J(), S(16), J(), S(102)]),
        "C4-19": lambda: compose([S(10), W(176), S(10)], [("raise", 30, 170, {"side": "right"})]),
        "C5-16": lambda: compose([S(10), TURN(40, -90), W(80), S(20), J(20), S(26)],
                                 [("raise", 146, 176, {"side": "both"})]),
    }
    return ps[prompt_id]()


# Controlled negatives: (prompt_id, motion, requirement ids expected to FAIL)
def negatives():
    return [
        ("C1-01", compose([S(10), W(80, "backward", 0.8), S(10)]), {"r2"}),
        ("C1-06", compose([S(20), TURN(40, -90), S(40)]), {"r2"}),
        ("C1-07", compose([S(100)]), {"r1", "r2"}),
        ("C2-01", compose([S(10), W(80, "right", 0.8), S(10)]), {"r2"}),
        ("C2-09", compose([S(50), KICK("right"), S(68)]), {"r2"}),
        ("C2-09", compose([S(50), KICK("left", "forward"), S(68)]), {"r3"}),
        ("C2-19", compose([S(196)], [("reach_across", 40, 150, {"side": "right"})]), {"r2", "r3"}),
        ("C2-19", compose([S(196)], [("reach_same", 40, 150, {"side": "left"})]), {"r3", "r4"}),
        ("C4-01", pilot_motion("C4-02"), {"r5"}),
        ("C4-01", compose([S(20), J(), S(20), TURN(40, 90), S(54)]), {"r4"}),
        ("C4-09", compose([S(20), J(), S(128)]), {"r2"}),
        ("C4-10", compose([S(20), J(), S(16), J(), S(144)]), {"r2"}),
        ("C4-19", compose([S(10), W(90), S(96)], [("raise", 120, 180, {"side": "right"})]), {"r5"}),
        ("C4-19", compose([S(10), W(176), S(10)], [("raise", 30, 170, {"side": "left"})]), {"r4"}),
        ("C5-16", compose([S(10), J(20), S(10), W(80), S(10), TURN(40, -90), S(26)],
                          [("raise", 10, 32, {"side": "both"})]), {"r10"}),
    ]
