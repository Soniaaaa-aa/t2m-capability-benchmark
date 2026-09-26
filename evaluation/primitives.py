"""
primitives.py — shared measurements on a standardised motion (whole team)
=========================================================================

Pure measurements, no PASS/FAIL decisions. Every evaluator builds on these, so
two evaluators never measure the same thing in two different ways.

Input: standardised motion [T, 22, 3], 20 fps, metres,
       +X = Right, +Y = Up, +Z = Forward (HumanML3D joint order, anatomical labels).

Sign conventions (tested in tests/test_primitives_events.py):
  * body_frame: right = left→right across hips + shoulders; forward = right × up.
    The standardised frame is mirrored (left-handed), so forward is right × up,
    not up × right.
  * heading_yaw: degrees, unwrapped, 0 = facing +Z, POSITIVE = turned LEFT.
"""

from __future__ import annotations

import numpy as np

FPS = 20.0

# HumanML3D joint indices (anatomical)
PELVIS, L_HIP, R_HIP = 0, 1, 2
L_KNEE, R_KNEE = 4, 5
L_ANKLE, R_ANKLE = 7, 8
L_FOOT, R_FOOT = 10, 11
NECK, HEAD = 12, 15
L_SHOULDER, R_SHOULDER = 16, 17
L_ELBOW, R_ELBOW = 18, 19
L_WRIST, R_WRIST = 20, 21

SIDE_JOINTS = {
    "left": {"hip": L_HIP, "knee": L_KNEE, "ankle": L_ANKLE, "foot": L_FOOT,
             "shoulder": L_SHOULDER, "elbow": L_ELBOW, "wrist": L_WRIST},
    "right": {"hip": R_HIP, "knee": R_KNEE, "ankle": R_ANKLE, "foot": R_FOOT,
              "shoulder": R_SHOULDER, "elbow": R_ELBOW, "wrist": R_WRIST},
}
UP = np.array([0.0, 1.0, 0.0])


def as_motion(motion):
    motion = np.asarray(motion, dtype=np.float64)
    if motion.ndim != 3 or motion.shape[1:] != (22, 3):
        raise ValueError(f"Expected [T, 22, 3], got {motion.shape}")
    return motion


def smooth(x, window=5):
    """Centred moving average along axis 0 (edges padded)."""
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or len(x) < 2:
        return x
    w = min(window, len(x))
    pad = w // 2
    xp = np.concatenate([np.repeat(x[:1], pad, axis=0), x, np.repeat(x[-1:], w - 1 - pad, axis=0)])
    kernel = np.ones(w) / w
    if x.ndim == 1:
        return np.convolve(xp, kernel, mode="valid")
    return np.stack([np.convolve(xp[:, i], kernel, mode="valid") for i in range(x.shape[1])], axis=1)


def segments(mask, min_len=1, merge_gap=0):
    """Runs of True in a boolean array → list of (start, end) with end exclusive."""
    mask = np.asarray(mask, dtype=bool)
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(mask)])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= min_len]


# ----------------------------------------------------------------------------- body scale
def body_scale(motion):
    """Median segment lengths used to normalise thresholds (metres)."""
    m = as_motion(motion)
    shoulder_width = float(np.median(np.linalg.norm(m[:, R_SHOULDER] - m[:, L_SHOULDER], axis=1)))
    leg = np.median([np.linalg.norm(m[:, SIDE_JOINTS[s]["hip"]] - m[:, SIDE_JOINTS[s]["knee"]], axis=1)
                     + np.linalg.norm(m[:, SIDE_JOINTS[s]["knee"]] - m[:, SIDE_JOINTS[s]["ankle"]], axis=1)
                     for s in ("left", "right")])
    arm = np.median([np.linalg.norm(m[:, SIDE_JOINTS[s]["shoulder"]] - m[:, SIDE_JOINTS[s]["elbow"]], axis=1)
                     + np.linalg.norm(m[:, SIDE_JOINTS[s]["elbow"]] - m[:, SIDE_JOINTS[s]["wrist"]], axis=1)
                     for s in ("left", "right")])
    height = float(np.median(m[:, HEAD, 1] - np.minimum(m[:, L_FOOT, 1], m[:, R_FOOT, 1])))
    return {"shoulder_width": shoulder_width, "leg_length": float(leg), "arm_length": float(arm),
            "height": max(height, 1e-6), "scale": max(height, 1e-6) / 1.7}


# ----------------------------------------------------------------------------- orientation
def body_frame(motion):
    """Per-frame unit vectors (right, up, forward), each [T, 3]."""
    m = as_motion(motion)
    across = (m[:, R_HIP] - m[:, L_HIP]) + (m[:, R_SHOULDER] - m[:, L_SHOULDER])
    across[:, 1] = 0.0
    norm = np.linalg.norm(across, axis=1)
    right = np.zeros_like(across)
    last = np.array([1.0, 0.0, 0.0])
    for t in range(len(m)):
        if norm[t] > 1e-6:
            last = across[t] / norm[t]
        right[t] = last
    up = np.repeat(UP[None], len(m), axis=0)
    forward = np.cross(right, up)   # left-handed standardised frame: right × up = forward
    return right, up, forward


def heading_yaw(motion, smooth_window=5):
    """Facing direction per frame in degrees, unwrapped; 0 = +Z, positive = turned left."""
    _, _, forward = body_frame(motion)
    yaw = np.unwrap(np.arctan2(-forward[:, 0], forward[:, 2]))
    return np.degrees(smooth(yaw, smooth_window))


def to_body(vectors, frame_t):
    """Express world vectors [N, 3] in the body frame of one frame: (right, up, forward) components."""
    right, up, forward = frame_t
    v = np.asarray(vectors, dtype=np.float64)
    return np.stack([v @ right, v @ up, v @ forward], axis=-1)


# ----------------------------------------------------------------------------- translation
def root_track(motion, fps=FPS, smooth_window=5):
    """Root XZ position, horizontal velocity (m/s) and speed (m/s), per frame."""
    m = as_motion(motion)
    xz = m[:, PELVIS][:, [0, 2]]
    vel = np.gradient(xz, axis=0) * fps if len(m) > 1 else np.zeros_like(xz)
    vel = smooth(vel, smooth_window)
    return {"xz": xz, "velocity": vel, "speed": np.linalg.norm(vel, axis=1)}


def displacement_in_frame(motion, start, end, frame_index=None):
    """
    Root displacement between frames start and end-1, expressed in the body frame
    at frame_index (default: start). Returns (lateral_right, forward) in metres.
    """
    m = as_motion(motion)
    end = min(end, len(m)) - 1
    frame_index = start if frame_index is None else frame_index
    right, _, forward = (x[frame_index] for x in body_frame(m))
    d = m[end, PELVIS] - m[start, PELVIS]
    d[1] = 0.0
    return float(d @ right), float(d @ forward)


# ----------------------------------------------------------------------------- contacts
def foot_heights(motion):
    """Lowest point of each foot (min of ankle and toe height), per frame."""
    m = as_motion(motion)
    return {s: np.minimum(m[:, SIDE_JOINTS[s]["ankle"], 1], m[:, SIDE_JOINTS[s]["foot"], 1])
            for s in ("left", "right")}


def foot_contacts(motion, contact_height=0.05, contact_speed=0.6, fps=FPS):
    """
    Per-foot contact (height AND speed thresholds, scaled by body height) plus
    the both-feet-off-ground mask:
        {"left": bool[T], "right": bool[T], "airborne": bool[T]}
    """
    m = as_motion(motion)
    s = body_scale(m)["scale"]
    h = foot_heights(m)
    ground = min(h["left"].min(), h["right"].min())
    out = {}
    for side in ("left", "right"):
        ankle = m[:, SIDE_JOINTS[side]["ankle"]]
        v = np.linalg.norm(np.gradient(ankle, axis=0), axis=1) * fps if len(m) > 1 else np.zeros(len(m))
        out[side] = ((h[side] - ground) < contact_height * s) & (v < contact_speed * s)
    lowest = np.minimum(h["left"], h["right"]) - ground
    out["airborne"] = lowest > contact_height * s
    return out


def effector_tracks(motion):
    """
    Wrists relative to their shoulder and ankles relative to their hip, expressed
    in the per-frame body frame: components (right, up, forward), metres, [T, 3].
    Keys: left_wrist, right_wrist, left_ankle, right_ankle.
    """
    m = as_motion(motion)
    right, up, forward = body_frame(m)
    out = {}
    for side in ("left", "right"):
        j = SIDE_JOINTS[side]
        for name, (end, root) in {"wrist": ("wrist", "shoulder"), "ankle": ("ankle", "hip")}.items():
            v = m[:, j[end]] - m[:, j[root]]
            out[f"{side}_{name}"] = np.stack([(v * right).sum(1), (v * up).sum(1), (v * forward).sum(1)], 1)
    return out


# ----------------------------------------------------------------------------- limbs
def limb_vectors(motion, side, limb):
    """End effector minus limb root per frame: arm → wrist − shoulder, leg → ankle − hip."""
    m = as_motion(motion)
    j = SIDE_JOINTS[side]
    if limb == "arm":
        return m[:, j["wrist"]] - m[:, j["shoulder"]]
    return m[:, j["ankle"]] - m[:, j["hip"]]


def limb_deviation(motion, side, limb):
    """Distance of the effector from the neutral hanging pose, / limb length, per frame."""
    scale = body_scale(motion)
    length = scale["arm_length"] if limb == "arm" else scale["leg_length"]
    v = limb_vectors(motion, side, limb)
    return np.linalg.norm(v - np.array([0.0, -length, 0.0]), axis=1) / length
