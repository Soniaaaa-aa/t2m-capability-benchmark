"""Tests for the hybrid-plan rule layer: primitives, events, the new evaluators, validation."""
from pathlib import Path

import numpy as np
import pytest

import common
import events as E
import primitives as P
import synth_motion as sm
import synth_scenarios as S
import validation as V

REPO = Path(__file__).resolve().parents[1]
DEFINITION = REPO / "benchmark" / "pilot_benchmark_definition.json"


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    common.load_all_evaluators(verbose=False)
    _, prompts = common.load_benchmark_definition(DEFINITION)
    d = tmp_path_factory.mktemp("Synth")
    for p in prompts:
        np.save(d / f"{p['prompt_id']}.npy", S.pilot_motion(p["prompt_id"]))
    reg, _ = common.register_motions(prompts, "Synth", d, verbose=False)
    cases = common.build_evaluation_cases(prompts, reg, "Synth", verbose=False)
    return cases, common.run_all_evaluators(cases)


def rotate_scene(m, deg):
    a = np.radians(deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return (m - m[0, 0] * [1, 0, 1]) @ R.T


# ----------------------------------------------------------------------------- sign conventions
def test_body_frame_and_yaw_signs():
    rest = sm.standardise(sm.rest(20))
    right, _, forward = P.body_frame(rest)
    assert np.allclose(right[0], [1, 0, 0]) and np.allclose(forward[0], [0, 0, 1])
    left_turn = S.pilot_motion("C1-06")
    assert P.heading_yaw(left_turn)[-1] == pytest.approx(90, abs=5)          # + = left
    assert P.heading_yaw(S.pilot_motion("C1-07"))[-1] == pytest.approx(-90, abs=5)


def test_left_walk_is_left_in_body_frame():
    lat, fwd = P.displacement_in_frame(S.pilot_motion("C2-01"), 0, 100)
    assert lat < -1.0 and abs(fwd) < 0.2


def test_event_timeline_of_composite_prompt():
    evs = E.detect_events(S.pilot_motion("C5-16"))
    first = {}
    for e in evs:
        first.setdefault(e.kind, e.start)
    assert first["turn"] < first["walk"] < first["jump"]
    assert [e.side for e in evs if e.kind == "turn"] == ["right"]


def test_requirement_context_uses_applies_to_and_order():
    case = {"requirements": [
        {"id": "r1", "type": "action", "value": "turn"},
        {"id": "r3", "type": "action", "value": "walk"},
        {"id": "r4", "type": "direction", "value": "forward", "applies_to": ["r3"]},
        {"id": "r7", "type": "action", "value": "raise_hands"},
        {"id": "r8", "type": "body_side", "value": "both", "applies_to": ["r7"]},
        {"id": "r10", "type": "order", "value": ["turn", "walk"], "applies_to": ["r1", "r3"],
         "event_sequence": [{"event": "turn", "requirement_ids": ["r1"]},
                            {"event": "walk", "requirement_ids": ["r3"]}]}]}
    ctx = E.requirement_context(case["requirements"][2], case)
    assert ctx["action"] == "walk" and ctx["step"] == 1 and ctx["previous"]["value"] == "turn"
    assert E.requirement_context(case["requirements"][3], case)["body_side"] == "both"


# ----------------------------------------------------------------------------- whole Pilot
def test_every_pilot_requirement_is_evaluated_and_passes(pilot):
    cases, rows = pilot
    assert len(rows) == 61
    assert all(r["status"] == "EVALUATED" for r in rows), [r for r in rows if r["status"] != "EVALUATED"]
    ids = {(c["prompt_id"], i): req["id"] for c in cases for i, req in enumerate(c["requirements"])}
    failing = {(r["prompt_id"], ids[(r["prompt_id"], r["requirement_index"])])
               for r in rows if r["pass_fail"] != "PASS"}
    assert failing == set()
    assert {r["evaluator"] for r in rows if r["requirement_type"] == "direction"} == {"BodyFrameDirectionEvaluator"}


def test_world_frame_rule_still_fails_turn_then_walk(pilot):
    """Why the team switched: A's world-frame rule calls C5-16's walk after a right turn 'right'."""
    cases, _ = pilot
    c = next(c for c in cases if c["prompt_id"] == "C5-16")
    world = common.get_evaluator("TrajectoryEvaluator").for_benchmark()
    out = world.evaluate(common.load_motion(c["motion_path"]), c["requirements"][3], c)
    assert out["pass_fail"] == "FAIL"


def test_body_frame_direction_fixes_turn_then_walk(pilot):
    cases, _ = pilot
    ev = common.get_evaluator("BodyFrameDirectionEvaluator").for_benchmark()
    for c in cases:
        for req in c["requirements"]:
            if req["type"] == "direction":
                out = ev.evaluate(common.load_motion(c["motion_path"]), req, c)
                assert out["pass_fail"] == "PASS", (c["prompt_id"], out["reason"])


@pytest.mark.parametrize("k", range(len(S.negatives())))
def test_controlled_negatives_fail(pilot, tmp_path, k):
    cases, _ = pilot
    pid, motion, must_fail = S.negatives()[k]
    path = tmp_path / f"{pid}.npy"
    np.save(path, motion)
    case = {**next(c for c in cases if c["prompt_id"] == pid), "motion_path": str(path)}
    rows = common.run_all_evaluators([case])
    failed = {req["id"] for req, r in zip(case["requirements"], rows) if r["pass_fail"] == "FAIL"}
    assert must_fail <= failed


@pytest.mark.parametrize("pid", ["C1-01", "C1-06", "C2-01", "C2-09", "C2-19", "C4-01", "C4-19", "C5-16"])
def test_rules_do_not_depend_on_initial_facing(pilot, tmp_path, pid):
    cases, rows = pilot
    case = next(c for c in cases if c["prompt_id"] == pid)
    path = tmp_path / f"{pid}.npy"
    np.save(path, rotate_scene(common.load_motion(case["motion_path"]), 37))
    new = common.run_all_evaluators([{**case, "motion_path": str(path)}])
    for r in new:
        assert r["pass_fail"] == "PASS", (pid, r["requirement_type"], r["reason"])


def test_robust_to_joint_jitter(pilot, tmp_path):
    """2 cm i.i.d. per-joint, per-frame noise must not change any Pilot decision."""
    cases, rows = pilot
    rng = np.random.default_rng(0)
    noisy = []
    for c in cases:
        m = common.load_motion(c["motion_path"])
        path = tmp_path / f"{c['prompt_id']}.npy"
        np.save(path, (m + rng.normal(0, 0.02, m.shape)).astype(np.float32))
        noisy.append({**c, "motion_path": str(path)})
    new = common.run_all_evaluators(noisy)
    assert [r["pass_fail"] for r in new] == [r["pass_fail"] for r in rows]


# ----------------------------------------------------------------------------- evaluator details
def test_evidence_only_without_thresholds(pilot):
    cases, _ = pilot
    case = next(c for c in cases if c["prompt_id"] == "C4-10")
    cls = common.get_evaluator("CountEvaluator")
    out = cls(thresholds=None).evaluate(common.load_motion(case["motion_path"]), case["requirements"][1], case)
    assert out["pass_fail"] is None and out["evidence"]["detected_count"] == 3


def test_unknown_action_goes_to_human_review():
    m = S.pilot_motion("C4-09")
    ev = common.get_evaluator("ActionEvaluator").for_benchmark()
    out = ev.evaluate(m, {"id": "r1", "type": "action", "value": "cartwheel"}, {})
    assert out["pass_fail"] is None and out["evidence"]["judge"] == "human_required"


def test_attribute_paired_and_fallback(pilot, tmp_path):
    cases, _ = pilot
    slow = next(c for c in cases if c["prompt_id"] == "C3-01")
    ev = common.get_evaluator("AttributeEvaluator").for_benchmark()
    out = ev.evaluate(common.load_motion(slow["motion_path"]), slow["requirements"][2], slow)
    assert out["evidence"]["mode"] == "paired" and out["pass_fail"] == "PASS"
    # the fast walk stored under the slow prompt id -> slow must FAIL against its partner
    np.save(tmp_path / "C3-01.npy", S.pilot_motion("C3-02"))
    np.save(tmp_path / "C3-02.npy", S.pilot_motion("C3-01"))
    swapped = {**slow, "motion_path": str(tmp_path / "C3-01.npy")}
    assert ev.evaluate(np.load(tmp_path / "C3-01.npy"), slow["requirements"][2], swapped)["pass_fail"] == "FAIL"
    # no partner file -> absolute fallback
    alone = {**slow, "motion_path": str(tmp_path / "nowhere" / "C3-01.npy")}
    out = ev.evaluate(common.load_motion(slow["motion_path"]), slow["requirements"][2], alone)
    assert out["evidence"]["mode"] == "absolute_fallback" and out["pass_fail"] == "PASS"


def test_arm_direction_and_torso_direction():
    common.load_all_evaluators(verbose=False)
    m = S.compose([S.S(100)], [("raise", 20, 80, {"side": "right"})])
    case = {"prompt": "A person raises the right arm up.", "requirements": [
        {"id": "r1", "type": "action", "value": "raise_hand"},
        {"id": "r2", "type": "body_side", "value": "right", "applies_to": ["r1"]},
        {"id": "r3", "type": "arm_direction", "value": "up", "applies_to": ["r1"]}]}
    limb = common.get_evaluator("LimbGeometryEvaluator").for_benchmark()
    assert limb.evaluate(m, case["requirements"][2], case)["pass_fail"] == "PASS"
    assert limb.evaluate(m, {**case["requirements"][2], "value": "forward"}, case)["pass_fail"] == "FAIL"

    lean = sm.rest(60)
    upper = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    sm.rotate_limb(lean, [0] + upper, [1, 0, 0], np.full(60, np.radians(30)))   # lean forward (+Z)
    torso = common.get_evaluator("TorsoGeometryEvaluator").for_benchmark()
    req = {"id": "r1", "type": "torso_direction", "value": "forward"}
    assert torso.evaluate(sm.standardise(lean), req, {})["pass_fail"] == "PASS"
    assert torso.evaluate(sm.standardise(lean), {**req, "value": "backward"}, {})["pass_fail"] == "FAIL"


# ----------------------------------------------------------------------------- validation tools
def test_cohen_kappa():
    assert V.cohen_kappa(["PASS", "FAIL"] * 5, ["PASS", "FAIL"] * 5) == 1.0
    assert V.cohen_kappa(["PASS", "PASS", "FAIL", "FAIL"], ["PASS", "FAIL", "PASS", "FAIL"]) == 0.0


def test_synthetic_negatives_are_caught(pilot, tmp_path):
    cases, _ = pilot
    gold = {c["prompt_id"]: {"prompt": c["prompt"], "requirements": [
        {"requirement_index": i, "type": r["type"], "value": r["value"], "human_label": "PASS"}
        for i, r in enumerate(c["requirements"])]} for c in cases}
    neg_cases, neg_labels = V.make_synthetic_negatives(cases, gold, tmp_path)
    assert {"C1-06__swap_lr", "C2-19__swap_lr", "C4-01__reverse", "C1-01__reverse"} <= {c["prompt_id"] for c in neg_cases}
    rows = common.run_all_evaluators(neg_cases, neg_labels)
    must = [r for r in rows if r["human_label"] == "FAIL"]
    assert must and all(r["pass_fail"] == "FAIL" for r in must), \
        [(r["prompt_id"], r["requirement_type"], r["reason"]) for r in must if r["pass_fail"] != "FAIL"]
    report = V.agreement_report(rows, verbose=False)
    assert report["body_side"]["agreement"] == 1.0


def test_grid_search_and_leave_one_model_out():
    cls = common.get_evaluator("RotationEvaluator")
    rows = []
    for model, angles in {"A": [20, 40, 60, 80], "B": [25, 45, 65, 85], "C": [30, 50, 70, 90]}.items():
        for a in angles:
            rows.append({"model": model, "evaluator": "RotationEvaluator", "requirement_type": "turn_direction",
                         "expected_value": "left", "pass_fail": "PASS",
                         "human_label": "PASS" if a >= 50 else "FAIL",
                         "evidence": {"score": a, "n_turn_events": 1, "dominant_yaw_change_deg": a,
                                      "net_yaw_change_deg": a}})
    best, table = V.grid_search(cls, rows, {"min_turn_deg": [30, 45, 50, 60]})
    assert best["min_turn_deg"] == 50 and table[0]["kappa"] == 1.0
    lomo = V.leave_one_model_out(cls, rows, {"min_turn_deg": [30, 45, 50, 60]})
    assert set(lomo) == {"A", "B", "C"} and all(v["test_n"] == 4 for v in lomo.values())


def test_setup_cell_can_be_rerun_after_reloading_common(pilot):
    """run_benchmark STEP 0 does importlib.reload(common) and load_all_evaluators() again."""
    import importlib
    cases, _ = pilot
    importlib.reload(common)
    common.load_all_evaluators(verbose=False)
    common.load_all_evaluators(verbose=False)
    rows = common.run_all_evaluators(cases[:3])
    assert all(r["status"] == "EVALUATED" for r in rows)


def test_body_frame_direction_turn_and_walk_counts_as_left(pilot):
    """'Walks to the left' by turning left and walking forward = left of the starting facing."""
    cases, _ = pilot
    m = S.compose([S.S(10), S.TURN(20, 90), S.W(70)])
    ev = common.get_evaluator("BodyFrameDirectionEvaluator").for_benchmark()
    left = next(c for c in cases if c["prompt_id"] == "C2-01")
    right = next(c for c in cases if c["prompt_id"] == "C2-02")
    assert ev.evaluate(m, left["requirements"][1], left)["pass_fail"] == "PASS"
    assert ev.evaluate(m, right["requirements"][1], right)["pass_fail"] == "FAIL"
