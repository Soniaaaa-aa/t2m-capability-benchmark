"""Synthetic HumanML3D-like 22-joint motions in the standardised frame
(+X right, +Y up, +Z forward), for offline testing of BodySideEvaluator."""
import numpy as np

T_DEFAULT = 100

# Rest pose (A-pose-ish), metres. Anatomical labels: 1 = left hip (at -X), etc.
REST = np.array([
    [0.00, 0.95, 0.00],   # 0 pelvis
    [-0.10, 0.90, 0.00],  # 1 L hip
    [0.10, 0.90, 0.00],   # 2 R hip
    [0.00, 1.05, 0.00],   # 3 spine1
    [-0.10, 0.50, 0.02],  # 4 L knee
    [0.10, 0.50, 0.02],   # 5 R knee
    [0.00, 1.18, 0.00],   # 6 spine2
    [-0.10, 0.08, 0.00],  # 7 L ankle
    [0.10, 0.08, 0.00],   # 8 R ankle
    [0.00, 1.30, 0.00],   # 9 spine3
    [-0.10, 0.02, 0.12],  # 10 L foot
    [0.10, 0.02, 0.12],   # 11 R foot
    [0.00, 1.50, 0.00],   # 12 neck
    [-0.07, 1.45, 0.00],  # 13 L collar
    [0.07, 1.45, 0.00],   # 14 R collar
    [0.00, 1.65, 0.02],   # 15 head
    [-0.18, 1.42, 0.00],  # 16 L shoulder
    [0.18, 1.42, 0.00],   # 17 R shoulder
    [-0.21, 1.15, 0.00],  # 18 L elbow
    [0.21, 1.15, 0.00],   # 19 R elbow
    [-0.23, 0.90, 0.02],  # 20 L wrist
    [0.23, 0.90, 0.02],   # 21 R wrist
], dtype=np.float64)

L_ARM = (16, 18, 20)
R_ARM = (17, 19, 21)
L_LEG = (1, 4, 7, 10)
R_LEG = (2, 5, 8, 11)


def rest(T=T_DEFAULT):
    return np.repeat(REST[None], T, axis=0).copy()


def envelope(T, start=0.3, end=0.7):
    """0 -> 1 -> 0 bump over [start, end] of the clip."""
    t = np.linspace(0, 1, T)
    e = np.clip((t - start) / (end - start), 0, 1)
    return np.sin(np.pi * e) ** 2


def rotate_limb(m, chain, axis, angle):
    """Rotate joints chain[1:] about chain[0] by angle[t] around a fixed axis."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    for t in range(len(m)):
        a = angle[t]
        R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
        root = m[t, chain[0]].copy()
        for j in chain[1:]:
            m[t, j] = root + R @ (m[t, j] - root)
    return m


def translate(m, velocity_xyz):
    T = len(m)
    offs = np.outer(np.arange(T), velocity_xyz)
    m += offs[:, None, :]
    return m


def arm_swing(m, amp=np.radians(30), freq=2.0):
    t = np.linspace(0, 1, len(m))
    rotate_limb(m, L_ARM, [1, 0, 0], amp * np.sin(2 * np.pi * freq * t))
    rotate_limb(m, R_ARM, [1, 0, 0], -amp * np.sin(2 * np.pi * freq * t))
    return m


def leg_swing(m, amp=np.radians(25), freq=2.0):
    t = np.linspace(0, 1, len(m))
    rotate_limb(m, L_LEG, [1, 0, 0], -amp * np.sin(2 * np.pi * freq * t))
    rotate_limb(m, R_LEG, [1, 0, 0], amp * np.sin(2 * np.pi * freq * t))
    return m


def walk(T=T_DEFAULT):
    m = rest(T)
    leg_swing(m)
    arm_swing(m)
    return translate(m, [0, 0, 0.025])


def side_kick(side, T=T_DEFAULT):
    m = rest(T)
    chain = L_LEG if side == "left" else R_LEG
    # abduction: left leg swings toward -X, right toward +X (rotation about Z)
    sign = -1 if side == "left" else 1
    rotate_limb(m, chain, [0, 0, 1], sign * np.radians(70) * envelope(T))
    return m


def reach_across(side, T=T_DEFAULT):
    """Hand toward the opposite shoulder: flex the arm forward/up, then adduct."""
    m = rest(T)
    chain = L_ARM if side == "left" else R_ARM
    env = envelope(T, 0.2, 0.8)
    rotate_limb(m, chain, [1, 0, 0], -np.radians(100) * env)     # raise forward
    sign = 1 if side == "left" else -1
    rotate_limb(m, chain, [0, 1, 0], sign * np.radians(60) * env)  # swing across
    return m


def raise_hand(side, m):
    chain = L_ARM if side == "left" else R_ARM
    sign = -1 if side == "left" else 1
    rotate_limb(m, chain, [0, 0, 1], sign * np.radians(160) * envelope(len(m), 0.15, 0.85))
    return m


def walk_raise(side, T=T_DEFAULT):
    m = rest(T)
    leg_swing(m)
    t = np.linspace(0, 1, T)
    other = R_ARM if side == "left" else L_ARM
    rotate_limb(m, other, [1, 0, 0], np.radians(30) * np.sin(2 * np.pi * 2 * t))
    raise_hand(side, m)
    return translate(m, [0, 0, 0.025])


def jump_both_hands(T=T_DEFAULT):
    m = rest(T)
    raise_hand("left", m)
    raise_hand("right", m)
    m[:, :, 1] += 0.3 * envelope(T, 0.4, 0.6)[:, None]
    return m


def rotate_y(m, degrees):
    a = np.radians(degrees)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    root0 = m[0, 0].copy()
    return (m - root0) @ R.T + root0


def swap_left_right(m):
    """Mirror X and swap left/right joint labels -> the opposite-side motion."""
    pairs = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21)]
    out = m.copy()
    out[:, :, 0] *= -1
    for a, b in pairs:
        out[:, [a, b]] = out[:, [b, a]]
    return out


def standardise(m):
    m = m.copy()
    m[:, :, 1] -= m[:, :, 1].min()
    m[:, :, 0] -= m[0, 0, 0]
    m[:, :, 2] -= m[0, 0, 2]
    return m.astype(np.float32)
