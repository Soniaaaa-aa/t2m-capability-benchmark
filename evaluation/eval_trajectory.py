"""
eval_trajectory.py — Direction requirements (TrajectoryEvaluator)
==================================================================

Owner: <team member A>   (edit this file only via your own branch + Pull Request)

Contents, moved unchanged from Benchmark_Df5_English.ipynb:
  - TrajectoryEvaluator            (STEP 10A)  evidence: dx, dz, dominant axis, ...
  - calculate_required_direction_ratio (STEP 10B)
  - DirectionDecisionRule          (STEP 10D)  pilot candidate rule, NOT FROZEN

Plus, at the end of the file (added, original classes untouched):
  - TrajectoryRequirementEvaluator  common-interface wrapper, registered as
    "TrajectoryEvaluator" so run_benchmark.ipynb evaluates Direction
    requirements automatically. Current rule: CURRENT_MIN_DISPLACEMENT.

TODO (owner, optional): TrajectoryEvaluator itself does not follow the common
interface (no BaseEvaluator, signature evaluate(motion, required_direction)).
Once it does, the wrapper can be removed.
Since 2026-09-26 the default direction rule is BodyFrameDirectionEvaluator
(eval_trajectory_ext.py); this world-frame rule is kept for comparison.
"""

import numpy as np


class TrajectoryEvaluator:
    """
    Use the root trajectory of the Standardised Motion to
    evaluate Direction Requirements.

    Standardised Motion coordinate system:
        +X = Right
        -X = Left
        +Z = Forward
        -Z = Backward
        +Y = Up

    Input motion shape:
        [T, J, 3]

        T = number of frames
        J = number of joints
        3 = XYZ coordinates

    Root joint:
        Joint 0
    """

    # Directions supported by this evaluator
    SUPPORTED_DIRECTIONS = {
        "forward",
        "backward",
        "left",
        "right",
    }

    def __init__(self):
        pass

    # --------------------------------------------------------
    # 1. Extract the root trajectory on the XZ plane
    # --------------------------------------------------------
    def extract_trajectory(self, motion):

        motion = np.asarray(motion)

        # Motionが [T, J, 3] の3次元配列になっているか確認
        if motion.ndim != 3:
            raise ValueError(
                f"MotionのShapeは [T, J, 3] である必要があります。"
                f"現在のShape: {motion.shape}"
            )

        # 最後の次元がXYZの3座標になっているか確認
        if motion.shape[2] != 3:
            raise ValueError(
                f"XYZ座標が必要です。現在のShape: {motion.shape}"
            )

        # Joint 0（ルートジョイント）のXYZ座標を取得
        root_xyz = motion[:, 0, :]

        # XYZのうちXとZだけを取得
        # Yは上下方向なのでDirection評価では使用しない
        trajectory_xz = root_xyz[:, [0, 2]]

        return trajectory_xz

    # --------------------------------------------------------
    # 2. 軌跡からDirection判定に必要なEvidenceを計算する
    # --------------------------------------------------------
    def calculate_evidence(self, motion):

        # ルートジョイントのXZ軌跡を取得
        trajectory = self.extract_trajectory(motion)

        # Motionの開始位置と終了位置
        start = trajectory[0]
        end = trajectory[-1]

        # 開始位置から終了位置までの移動量
        displacement = end - start

        # X方向への移動量
        dx = float(displacement[0])

        # Z方向への移動量
        dz = float(displacement[1])

        # XZ平面上での総移動距離（直線距離）
        total_displacement = float(
            np.linalg.norm(displacement)
        )

        # X・Zそれぞれの移動量の絶対値
        abs_dx = abs(dx)
        abs_dz = abs(dz)

        # ----------------------------------------------------
        # どちらの軸への移動が大きいかを確認
        # ----------------------------------------------------
        if abs_dx > abs_dz:
            dominant_axis = "X"

        elif abs_dz > abs_dx:
            dominant_axis = "Z"

        else:
            dominant_axis = "equal"

        # ----------------------------------------------------
        # Directional Dominanceを計算
        #
        # Example:
        # X方向 = 2.0m
        # Z方向 = 0.5m
        #
        # dominance_ratio = 2.0 / 0.5 = 4.0
        #
        # → X方向への移動がZ方向より4倍大きい
        # ----------------------------------------------------
        minor = min(abs_dx, abs_dz)
        major = max(abs_dx, abs_dz)

        if minor > 1e-8:
            dominance_ratio = major / minor

        elif major > 1e-8:
            dominance_ratio = float("inf")

        else:
            # XにもZにもほとんど移動していない
            dominance_ratio = 0.0

        # ----------------------------------------------------
        # displacementだけを使った仮のDirection判定
        #
        # ※ これはまだ最終的なPASS/FAIL判定ではない
        # ----------------------------------------------------
        if dominant_axis == "X":

            # +X = Right
            # -X = Left
            raw_direction = "right" if dx > 0 else "left"

        elif dominant_axis == "Z":

            # +Z = Forward
            # -Z = Backward
            raw_direction = "forward" if dz > 0 else "backward"

        else:
            raw_direction = "undetermined"

        # 計算したEvidenceをまとめて返す
        return {
            "trajectory_xz": trajectory,

            "start_x": float(start[0]),
            "start_z": float(start[1]),

            "end_x": float(end[0]),
            "end_z": float(end[1]),

            "dx": dx,
            "dz": dz,

            "total_displacement": total_displacement,

            "abs_dx": abs_dx,
            "abs_dz": abs_dz,

            "dominant_axis": dominant_axis,
            "dominance_ratio": dominance_ratio,

            "raw_direction": raw_direction,
        }

    # --------------------------------------------------------
    # 3. 1つのDirection Requirementを評価する
    # --------------------------------------------------------
    def evaluate(self, motion, required_direction):

        # "LEFT" や "Left" が入力されても
        # "left" に統一する
        required_direction = required_direction.lower()

        # 対応していないDirectionが入力された場合はエラー
        if required_direction not in self.SUPPORTED_DIRECTIONS:
            raise ValueError(
                f"対応していないDirectionです: {required_direction}"
            )

        # MotionからDirection Evidenceを計算
        evidence = self.calculate_evidence(motion)

        # ----------------------------------------------------
        # 重要：
        #
        # この段階ではThresholdを使った
        # 最終的なPASS / FAIL判定はまだ行わない。
        #
        # Thresholdは今後、
        #
        #   ・Pilot Motion
        #   ・Human Gold Label
        #
        # を比較して決定する。
        # ----------------------------------------------------

        return {
            # 評価しているRequirementの種類
            "requirement_type": "direction",

            # Promptが要求しているDirection
            "required_direction": required_direction,

            # 現在の単純な軌跡計算から推定されたDirection
            "predicted_direction_raw":
                evidence["raw_direction"],

            # Threshold未設定なので、まだPASS/FAILは出さない
            "pass_fail": None,

            # Direction判定に使用するEvidence
            "evidence": evidence,
        }


# ============================================================
# 2. Calculate Required Direction Ratio
# ============================================================

def calculate_required_direction_ratio(
    dx,
    dz,
    required_direction,
    eps=1e-8,
):
    """
    For the Required Direction,

        required displacement
        ---------------------
        orthogonal displacement

    calculate the required-direction displacement and ratio.


    Example:

        Required = left

        dx = -1.2
        dz = +0.5

        required displacement   = 1.2
        orthogonal displacement = 0.5

        ratio = 2.4


    Note:

    If the motion moves opposite to the required direction,
    required displacementは負になる。
    """

    required_direction = required_direction.lower()


    # --------------------------------------------------------
    # Required方向の移動量を取得
    # --------------------------------------------------------

    if required_direction == "left":

        # -X = Left
        required_displacement = -dx

        # Z方向は直交方向
        orthogonal_displacement = abs(dz)


    elif required_direction == "right":

        # +X = Right
        required_displacement = dx

        # Z方向は直交方向
        orthogonal_displacement = abs(dz)


    elif required_direction == "forward":

        # +Z = Forward
        required_displacement = dz

        # X方向は直交方向
        orthogonal_displacement = abs(dx)


    elif required_direction == "backward":

        # -Z = Backward
        required_displacement = -dz

        # X方向は直交方向
        orthogonal_displacement = abs(dx)


    else:

        raise ValueError(
            f"Unsupported direction: {required_direction}"
        )


    # --------------------------------------------------------
    # Calculate ratio
    # --------------------------------------------------------

    if orthogonal_displacement > eps:

        ratio = (
            required_displacement
            / orthogonal_displacement
        )

    elif required_displacement > eps:

        ratio = float("inf")

    elif required_displacement < -eps:

        ratio = float("-inf")

    else:

        ratio = 0.0


    return (
        float(required_displacement),
        float(orthogonal_displacement),
        float(ratio),
    )


# ============================================================
# STEP 10D — Initial Direction Decision Rule (Pilot candidate — NOT FROZEN)
# ============================================================

class DirectionDecisionRule:

    def __init__(self, min_displacement=0.50):
        self.min_displacement = float(min_displacement)

    def evaluate(self, evidence, required_direction):

        required_direction = required_direction.lower()

        dx = float(evidence["dx"])
        dz = float(evidence["dz"])

        if required_direction == "forward":
            required_displacement = dz
            correct_sign = dz > 0

        elif required_direction == "backward":
            required_displacement = -dz
            correct_sign = dz < 0

        elif required_direction == "right":
            required_displacement = dx
            correct_sign = dx > 0

        elif required_direction == "left":
            required_displacement = -dx
            correct_sign = dx < 0

        else:
            raise ValueError(
                f"Unsupported direction: {required_direction}"
            )

        sufficient_displacement = (
            required_displacement >= self.min_displacement
        )

        prediction = (
            "PASS"
            if correct_sign and sufficient_displacement
            else "FAIL"
        )

        return {
            "required_direction": required_direction,
            "prediction": prediction,
            "correct_sign": correct_sign,
            "required_displacement": required_displacement,
            "min_displacement_threshold": self.min_displacement,
            "sufficient_displacement": sufficient_displacement,
            "dominant_axis": evidence.get("dominant_axis"),
            "dominance_ratio": evidence.get("dominance_ratio"),
        }


# ============================================================
# Adapter for the unified runner (common.run_all_evaluators)
# ------------------------------------------------------------
# Added so that Direction requirements are evaluated automatically by
# run_benchmark.ipynb. The classes above are NOT modified; this wrapper
# only calls TrajectoryEvaluator + DirectionDecisionRule and returns the
# common result format. Owner may replace it once TrajectoryEvaluator
# itself follows BaseEvaluator.
# ============================================================

from common import BaseEvaluator, register_evaluator  # noqa: E402


class TrajectoryRequirementEvaluator(BaseEvaluator):
    """Common-interface wrapper around TrajectoryEvaluator + DirectionDecisionRule."""

    EVALUATOR_NAME = "TrajectoryEvaluator"

    # Current rule — STEP 10D pilot candidate. Update here after calibration.
    CURRENT_MIN_DISPLACEMENT = 0.50
    CURRENT_THRESHOLD_STATUS = "pilot_candidate_not_frozen"

    def __init__(self, min_displacement=None, threshold_status=None):
        self.min_displacement = (
            self.CURRENT_MIN_DISPLACEMENT if min_displacement is None else float(min_displacement)
        )
        self.threshold_status = threshold_status or self.CURRENT_THRESHOLD_STATUS
        self.trajectory = TrajectoryEvaluator()
        self.rule = DirectionDecisionRule(min_displacement=self.min_displacement)

    @classmethod
    def for_benchmark(cls):
        return cls()

    def evaluate(self, motion, requirement, evaluation_case=None):
        required = str(requirement.get("value", requirement.get("expected"))).lower()
        raw = self.trajectory.evaluate(motion=motion, required_direction=required)
        decision = self.rule.evaluate(evidence=raw["evidence"], required_direction=required)
        evidence = {k: v for k, v in raw["evidence"].items() if k != "trajectory_xz"}
        evidence.update({
            "predicted_direction_raw": raw["predicted_direction_raw"],
            "correct_sign": decision["correct_sign"],
            "required_displacement": decision["required_displacement"],
            "sufficient_displacement": decision["sufficient_displacement"],
        })
        return {
            "requirement_id": requirement.get("requirement_id", requirement.get("id")),
            "requirement_type": "direction",
            "expected_value": required,
            "evaluator_name": self.EVALUATOR_NAME,
            "pass_fail": decision["prediction"],
            "score": float(decision["required_displacement"]),
            "threshold_status": self.threshold_status,
            "evidence": evidence,
            "reason": (
                f"required={required}, dx={evidence['dx']:+.3f} m, dz={evidence['dz']:+.3f} m, "
                f"required displacement={decision['required_displacement']:+.3f} m "
                f"(min {self.min_displacement} m), correct sign={decision['correct_sign']}"
            ),
        }


register_evaluator("TrajectoryEvaluator", TrajectoryRequirementEvaluator)
