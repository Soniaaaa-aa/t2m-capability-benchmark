"""Tests for evaluator discovery, the unified runner and input loading (common.py)."""
import json
import textwrap

import numpy as np
import pytest

import common
import eval_trajectory as tr
import synth_motion as sm

A = lambda v: {"type": "action", "value": v}
R = lambda t, v: {"type": t, "value": v}


def test_discovery_imports_all_eval_files_except_template():
    loaded = common.load_all_evaluators(verbose=False)
    assert "eval_trajectory" in loaded and "eval_body_side" in loaded
    assert "eval_template" not in loaded
    assert {"TrajectoryEvaluator", "BodySideEvaluator"} <= set(common.EVALUATOR_REGISTRY)


def test_new_evaluator_file_is_picked_up_without_notebook_change(tmp_path):
    (tmp_path / "eval_dummy.py").write_text(textwrap.dedent('''
        from common import BaseEvaluator, register_evaluator
        class DummyEvaluator(BaseEvaluator):
            def evaluate(self, motion, requirement, evaluation_case=None):
                return {"pass_fail": "PASS", "score": 1.0, "reason": "dummy"}
        register_evaluator("DummyEvaluator", DummyEvaluator)
    '''))
    try:
        assert common.load_all_evaluators(tmp_path, verbose=False) == ["eval_dummy"]
        assert "DummyEvaluator" in common.EVALUATOR_REGISTRY
    finally:
        common.EVALUATOR_REGISTRY.pop("DummyEvaluator", None)


def _cases(tmp_path):
    specs = [
        ("C1-01", "A person walks forward.", sm.walk(), [A("walk"), R("direction", "forward")]),
        ("C2-01", "A person walks to the left.", sm.rotate_y(sm.walk(), -90), [A("walk"), R("direction", "left")]),
        ("C2-09", "A person kicks the left leg out to the side.", sm.side_kick("left"),
         [A("kick"), R("body_side", "left"), R("leg_direction", "side")]),
        ("C1-06", "A person turns to the left.", sm.rest(), [A("turn"), R("turn_direction", "left")]),
    ]
    cases, labels = [], {}
    for pid, text, motion, reqs in specs:
        path = tmp_path / f"{pid}.npy"
        np.save(path, sm.standardise(motion))
        cases.append({"model": "M", "prompt_id": pid, "prompt": text, "motion_path": str(path),
                      "requirements": reqs})
        labels[pid] = {"prompt": text, "requirements": [
            {"requirement_index": i, "type": r["type"], "value": r["value"],
             "human_label": "FAIL" if pid == "C2-01" and r["type"] == "direction" else "PASS"}
            for i, r in enumerate(reqs)]}
    return cases, labels


def test_run_all_evaluators(tmp_path):
    common.load_all_evaluators(verbose=False)
    cases, labels = _cases(tmp_path)
    rows = common.run_all_evaluators(cases, labels)

    assert len(rows) == sum(len(c["requirements"]) for c in cases)
    by = {(r["prompt_id"], r["requirement_type"]): r for r in rows}
    assert by[("C1-01", "direction")]["pass_fail"] == "PASS"
    # C2-01 motion: the whole scene is rotated, so the person walks FORWARD in their own
    # frame. Body-frame direction (default since 2026-09-26) says "not left".
    assert by[("C2-01", "direction")]["evaluator"] == "BodyFrameDirectionEvaluator"
    assert by[("C2-01", "direction")]["pass_fail"] == "FAIL"
    assert by[("C2-01", "direction")]["match"] is True             # human said FAIL
    assert by[("C2-09", "body_side")]["pass_fail"] == "PASS"
    assert by[("C2-09", "body_side")]["match"] is True
    # C1-06 motion is a rest pose: turn_direction is now evaluated (and fails).
    assert by[("C1-06", "turn_direction")]["status"] == "EVALUATED"
    assert by[("C1-06", "turn_direction")]["pass_fail"] == "FAIL"
    # A mapped evaluator that nobody has written yet is still reported, not raised.
    rows_missing = common.run_all_evaluators(cases[:1], None, config={"action": "NotWrittenYet"})
    assert rows_missing[0]["status"] == "NOT_IMPLEMENTED"

    summary = common.summarize_results(rows, verbose=False)
    assert summary["direction"]["PASS"] == 1 and summary["direction"]["agree"] == 2

    # A's world-frame rule is still available on request
    world = common.run_all_evaluators(cases, labels, config={**common.EVALUATION_CONFIG,
                                                             "direction": "TrajectoryEvaluator"})
    wby = {(r["prompt_id"], r["requirement_type"]): r for r in world}
    assert wby[("C2-01", "direction")]["pass_fail"] == "PASS"


def test_direction_wrapper_equals_original_rule(tmp_path):
    """The wrapper must give exactly what TrajectoryEvaluator + DirectionDecisionRule give."""
    common.load_all_evaluators(verbose=False)
    wrapper = common.get_evaluator("TrajectoryEvaluator").for_benchmark()
    for deg, req in [(0, "forward"), (-90, "left"), (90, "right"), (60, "right"), (180, "backward"), (0, "left")]:
        m = sm.standardise(sm.rotate_y(sm.walk(), deg))
        raw = tr.TrajectoryEvaluator().evaluate(m, req)
        rule = tr.DirectionDecisionRule(min_displacement=0.50).evaluate(raw["evidence"], req)
        out = wrapper.evaluate(m, R("direction", req), None)
        assert out["pass_fail"] == rule["prediction"]
        assert out["score"] == pytest.approx(rule["required_displacement"])


def test_errors_are_reported_not_raised(tmp_path):
    class Broken(common.BaseEvaluator):
        def evaluate(self, motion, requirement, evaluation_case=None):
            raise RuntimeError("boom")
    common.register_evaluator("BrokenEvaluator", Broken)
    try:
        cases, _ = _cases(tmp_path)
        rows = common.run_all_evaluators(cases[:1], None, config={"direction": "BrokenEvaluator"})
        statuses = {r["requirement_type"]: r["status"] for r in rows}
        assert statuses == {"action": "NO_EVALUATOR_MAPPED", "direction": "ERROR"}
    finally:
        common.EVALUATOR_REGISTRY.pop("BrokenEvaluator", None)


def test_save_results(tmp_path):
    common.load_all_evaluators(verbose=False)
    cases, labels = _cases(tmp_path)
    rows = common.run_all_evaluators(cases, labels)
    json_path, csv_path = common.save_results(rows, tmp_path / "out" / "M_results.json", {"model": "M"})
    payload = json.loads(json_path.read_text())
    assert payload["metadata"]["model"] == "M" and len(payload["results"]) == len(rows)
    assert csv_path.read_text().count("\n") == len(rows) + 1


def test_load_inputs_from_repo(tmp_path):
    cases, labels = _cases(tmp_path)
    repo = tmp_path / "repo"
    (repo / "benchmark").mkdir(parents=True)
    (repo / "labels").mkdir()
    prompts = [{"prompt_id": c["prompt_id"], "capability": "C", "difficulty": "Easy", "text": c["prompt"],
                "target_frames_20fps": 100, "requirements": c["requirements"]} for c in cases]
    (repo / "benchmark" / "pilot_benchmark_definition.json").write_text(json.dumps({"prompts": prompts}))
    common.save_gold_labels(labels, repo / "labels" / common.gold_label_filename("M"))
    inputs = tmp_path / "inputs" / "M"
    inputs.mkdir(parents=True)
    for c in cases:
        np.save(inputs / f"{c['prompt_id']}.npy", np.load(c["motion_path"]))

    data = common.load_inputs("M", repo, input_root=tmp_path / "inputs", verbose=False)
    assert len(data["evaluation_cases"]) == 4
    assert data["human_gold_labels"] == labels
