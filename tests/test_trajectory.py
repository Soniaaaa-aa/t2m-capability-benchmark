"""Basic checks for eval_trajectory.py (owner: team member A — extend as needed)."""
import numpy as np

import eval_trajectory as tr
import synth_motion as sm


def test_raw_direction_from_root_trajectory():
    ev = tr.TrajectoryEvaluator()
    walk = sm.standardise(sm.walk())                       # moves toward +Z
    assert ev.evaluate(walk, "forward")["predicted_direction_raw"] == "forward"
    left = sm.standardise(sm.rotate_y(sm.walk(), -90))     # moves toward -X
    assert ev.evaluate(left, "left")["predicted_direction_raw"] == "left"
    assert ev.evaluate(walk, "forward")["pass_fail"] is None


def test_required_direction_ratio():
    disp, orth, ratio = tr.calculate_required_direction_ratio(-1.2, 0.5, "left")
    assert (disp, orth) == (1.2, 0.5) and np.isclose(ratio, 2.4)


def test_initial_direction_rule():
    rule = tr.DirectionDecisionRule(min_displacement=0.5)
    assert rule.evaluate({"dx": 0.1, "dz": 2.0}, "forward")["prediction"] == "PASS"
    assert rule.evaluate({"dx": 0.1, "dz": 0.3}, "forward")["prediction"] == "FAIL"
    assert rule.evaluate({"dx": 1.0, "dz": 0.0}, "left")["prediction"] == "FAIL"
