import json

import numpy as np
import pytest

import common
import eval_body_side as bs
import synth_motion as sm

A = lambda v: {"type": "action", "value": v}
R = lambda t, v: {"type": t, "value": v}

# name: (motion factory, prompt, requirements, body_side index, expected limb)
SCENARIOS = {
    "C2-09": (lambda: sm.side_kick("left"), "A person kicks the left leg out to the side.",
              [A("kick"), R("body_side", "left"), R("leg_direction", "side")], 1, "leg"),
    "C2-10": (lambda: sm.side_kick("right"), "A person kicks the right leg out to the side.",
              [A("kick"), R("body_side", "right"), R("leg_direction", "side")], 1, "leg"),
    "C2-19": (lambda: sm.reach_across("left"),
              "A person moves the left hand across the body toward the right shoulder.",
              [A("reach"), R("body_side", "left"), R("target", "right_shoulder"),
               R("relation", "cross_body")], 1, "arm"),
    "C2-20": (lambda: sm.reach_across("right"),
              "A person moves the right hand across the body toward the left shoulder.",
              [A("reach"), R("body_side", "right"), R("target", "left_shoulder"),
               R("relation", "cross_body")], 1, "arm"),
    "C4-19": (lambda: sm.walk_raise("right"), "A person walks forward while raising the right hand.",
              [A("walk"), R("direction", "forward"), A("raise_hand"), R("body_side", "right"),
               R("simultaneous", ["walk", "raise"])], 3, "arm"),
    "C5-16": (sm.jump_both_hands,
              "A person turns right, walks forward, and then jumps once with both hands raised.",
              [A("turn"), R("turn_direction", "right"), A("walk"), R("direction", "forward"),
               A("jump"), R("count", 1), A("raise_hands"), R("body_side", "both"),
               R("simultaneous", ["jump", "hands"]), R("order", ["turn", "walk", "jump"])], 7, "arm"),
}


def case(prompt, reqs, pid="X"):
    return {"prompt_id": pid, "prompt": prompt, "requirements": reqs}


@pytest.fixture
def ev():
    return bs.BodySideEvaluator(thresholds={}, threshold_status="provisional")


@pytest.mark.parametrize("name", SCENARIOS)
def test_correct_side_passes_and_limb_inferred(ev, name):
    make, prompt, reqs, idx, limb = SCENARIOS[name]
    r = ev.evaluate(sm.standardise(make()), reqs[idx], case(prompt, reqs))
    assert r["pass_fail"] == "PASS", r["reason"]
    assert r["limb"] == limb


@pytest.mark.parametrize("name", [n for n in SCENARIOS if n != "C5-16"])
def test_opposite_side_fails(ev, name):
    make, prompt, reqs, idx, _ = SCENARIOS[name]
    req = dict(reqs[idx], value="right" if reqs[idx]["value"] == "left" else "left")
    reqs2 = list(reqs)
    reqs2[idx] = req
    assert ev.evaluate(sm.standardise(make()), req, case(prompt, reqs2))["pass_fail"] == "FAIL"


def test_negatives(ev):
    walk = sm.standardise(sm.walk())
    _, p516, r516, i516, _ = SCENARIOS["C5-16"]
    _, p419, r419, i419, _ = SCENARIOS["C4-19"]
    assert ev.evaluate(walk, r516[i516], case(p516, r516))["pass_fail"] == "FAIL"   # arm swing != both raised
    assert ev.evaluate(walk, r419[i419], case(p419, r419))["pass_fail"] == "FAIL"
    one_hand = sm.standardise(sm.raise_hand("left", sm.rest()))
    assert ev.evaluate(one_hand, r516[i516], case(p516, r516))["pass_fail"] == "FAIL"
    idle = ev.evaluate(sm.standardise(sm.rest()), R("body_side", "left"),
                       case("A person kicks the left leg.", [A("kick"), R("body_side", "left")]))
    assert idle["pass_fail"] == "FAIL" and idle["predicted_side"] == "none"


@pytest.mark.parametrize("deg", [37, 90, 180])
def test_rotation_invariance(ev, deg):
    _, _, reqs, idx, _ = SCENARIOS["C2-09"]
    base = ev.evaluate(sm.standardise(sm.side_kick("left")), reqs[idx], case("", reqs))["evidence"]
    rot = ev.evaluate(sm.standardise(sm.rotate_y(sm.side_kick("left"), deg)), reqs[idx], case("", reqs))["evidence"]
    assert rot["left_activity"] == pytest.approx(base["left_activity"], abs=1e-6)
    assert rot["laterality_index"] == pytest.approx(base["laterality_index"], abs=1e-6)


def test_scale_mirror_and_relabel(ev):
    _, _, reqs, idx, _ = SCENARIOS["C2-09"]
    base = ev.evaluate(sm.standardise(sm.side_kick("left")), reqs[idx], case("", reqs))
    small = ev.evaluate(sm.standardise(sm.side_kick("left") * 0.8), reqs[idx], case("", reqs))
    assert small["evidence"]["left_activity"] == pytest.approx(base["evidence"]["left_activity"], abs=1e-6)
    flipped = sm.standardise(sm.side_kick("left"))
    flipped[:, :, 0] *= -1                         # X-axis sign does not matter
    assert ev.evaluate(flipped, reqs[idx], case("", reqs))["pass_fail"] == "PASS"
    swapped = sm.standardise(sm.swap_left_right(sm.side_kick("left")))   # = right kick
    assert ev.evaluate(swapped, reqs[idx], case("", reqs))["pass_fail"] == "FAIL"


def test_interface_and_modes():
    _, _, reqs, idx, _ = SCENARIOS["C2-09"]
    m = sm.standardise(sm.side_kick("left"))
    r = bs.BodySideEvaluator().evaluate(m, reqs[idx], case("", reqs))
    assert r["pass_fail"] is None and r["predicted_side_raw"] == "left"
    assert {"requirement_id", "requirement_type", "expected_value", "evaluator_name",
            "pass_fail", "score", "evidence", "reason"} <= set(r)
    json.dumps(r, default=float)
    assert common.get_evaluator("BodySideEvaluator") is bs.BodySideEvaluator
    with pytest.raises(ValueError):
        bs.BodySideEvaluator().evaluate(m, R("body_side", "middle"), None)
    assert bs.BodySideEvaluator().evaluate(m, R("body_side", "Left"), None)["limb_source"] == "auto"


@pytest.mark.parametrize("text,limb", [("raise_leg", "leg"), ("raise_hand", "arm"), ("kick", "leg"),
                                       ("wave", "arm"), ("jump", None),
                                       ("A person stomps the left foot.", "leg")])
def test_limb_words(text, limb):
    assert bs.BodySideEvaluator._limb_from_words(text) == limb


def test_pilot_workflow_helpers(tmp_path):
    """collect -> calibrate -> evaluate, with one wrong-leg motion labelled FAIL."""
    cases, labels = [], {}
    for pid, (make, prompt, reqs, idx, _) in SCENARIOS.items():
        motion = sm.side_kick("left") if pid == "C2-10" else make()   # C2-10: wrong leg
        path = tmp_path / f"{pid}.npy"
        np.save(path, sm.standardise(motion))
        cases.append({"model": "M", "prompt_id": pid, "prompt": prompt,
                      "motion_path": str(path), "requirements": reqs})
        labels[pid] = {"prompt": prompt, "requirements": [
            {"requirement_index": i, "type": r["type"], "value": r["value"],
             "human_label": ("FAIL" if pid == "C2-10" and r["type"] == "body_side" else "PASS")}
            for i, r in enumerate(reqs)]}

    rows = bs.collect_body_side_evidence(cases, labels)
    assert len(rows) == 6
    cal = bs.calibrate_body_side_thresholds(rows)
    assert cal["correct"] == 6 and cal["status"] == "calibrated_small_sample"
    final = bs.evaluate_body_side(cases, cal["thresholds"], cal["status"], labels)
    assert all(r["pass_fail"] == r["human_label"] for r in final)
    assert bs.calibrate_body_side_thresholds([dict(r, human_label=None) for r in rows])["status"] == "provisional"
