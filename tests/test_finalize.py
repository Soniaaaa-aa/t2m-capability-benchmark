"""Tests for plan B finalisation: lint, auto/human decision, review sheet, merge, score tables."""
import csv
from pathlib import Path

import numpy as np
import pytest

import common
import finalize as F
import synth_scenarios as S
import validation as V

REPO = Path(__file__).resolve().parents[1]
DEFINITION = REPO / "benchmark" / "pilot_benchmark_definition.json"


@pytest.fixture(scope="module")
def pilot_prompts():
    common.load_all_evaluators(verbose=False)
    return common.load_benchmark_definition(DEFINITION)[1]


@pytest.fixture(scope="module")
def three_models(pilot_prompts, tmp_path_factory):
    """Three synthetic 'models': correct motions, with a few negatives swapped in; gold = truth."""
    negs = S.negatives()
    all_rows, all_cases, golds = [], {}, {}
    for k, model in enumerate(("M1", "M2", "M3")):
        d = tmp_path_factory.mktemp(model)
        must = {}
        for p in pilot_prompts:
            np.save(d / f"{p['prompt_id']}.npy", S.pilot_motion(p["prompt_id"]))
        for j, (pid, m, fail) in enumerate(negs):
            if j % 3 == k and pid not in must:
                np.save(d / f"{pid}.npy", m)
                must[pid] = fail
        reg, _ = common.register_motions(pilot_prompts, model, d, verbose=False)
        cases = common.build_evaluation_cases(pilot_prompts, reg, model, verbose=False)
        rows = common.run_all_evaluators(cases)
        fails = {(r["prompt_id"], r["requirement_index"]) for r in rows if r["pass_fail"] == "FAIL"}
        gold = {c["prompt_id"]: {"prompt": c["prompt"], "requirements": [
            {"requirement_index": i, "type": q["type"], "value": q["value"],
             # "human" = truth: the planted errors, plus FAILs that follow from them
             "human_label": "FAIL" if (q["id"] in must.get(c["prompt_id"], ()) or
                                       (c["prompt_id"] in must and (c["prompt_id"], i) in fails)) else "PASS"}
            for i, q in enumerate(c["requirements"])]} for c in cases}
        all_rows += common.run_all_evaluators(cases, gold)
        all_cases[model], golds[model] = cases, gold
    return all_rows, all_cases, golds


def test_pilot_definition_is_clean(pilot_prompts):
    assert [p for p in F.lint_definition(pilot_prompts) if p["level"] == "error"] == []


def test_lint_catches_out_of_vocabulary_and_bad_links(pilot_prompts):
    bad = [{"prompt_id": "X-1", "text": "A person does a cartwheel twice.", "requirements": [
        {"id": "r1", "type": "action", "value": "cartwheel"},
        {"id": "r2", "type": "count", "value": 2, "applies_to": ["r1"]},
        {"id": "r3", "type": "direction", "value": "up", "applies_to": ["r9"]},
        {"id": "r4", "type": "colour", "value": "red"}]}]
    msgs = " | ".join(p["message"] for p in F.lint_definition(bad))
    for part in ("outside the rule vocabulary", "not countable", "unknown ids", "not supported",
                 "unknown requirement type"):
        assert part in msgs


def test_decide_modes_needs_fail_examples(three_models):
    rows, _, _ = three_models
    mode = F.decide_modes(rows)
    types = mode["types"]
    assert set(types) == set(common.EVALUATION_CONFIG)
    assert types["arm_direction"]["mode"] == "human"            # never in the Pilot
    assert types["torso_direction"]["mode"] == "human"
    for t, m in types.items():
        if m["mode"] == "auto":
            c = m["combined"]
            assert c["kappa"] >= 0.6 and c["n"] >= 10 and c["n_human_fail"] >= 3


def test_synthetic_negatives_can_unlock_auto(three_models, tmp_path):
    rows, cases, golds = three_models
    before = F.decide_modes(rows)["types"]["body_side"]["mode"]
    neg_rows = []
    for model in cases:
        nc, nl = V.make_synthetic_negatives(cases[model], golds[model], tmp_path / model)
        neg_rows += common.run_all_evaluators(nc, nl)
    after = F.decide_modes(rows, neg_rows)["types"]["body_side"]
    assert after["combined"]["n"] > F.decide_modes(rows)["types"]["body_side"]["combined"]["n"]
    assert after["mode"] == "auto" or before == "auto"


def test_mode_file_roundtrip(three_models, tmp_path):
    rows, _, _ = three_models
    path = F.save_mode(F.decide_modes(rows), tmp_path / "evaluation_mode.json", {"commit": "abc"})
    loaded = F.load_mode(path)
    assert loaded["commit"] == "abc" and "ActionEvaluator" in loaded["evaluator_thresholds"]
    assert F.load_mode(tmp_path / "missing.json")["types"] == {}


def test_review_sheet_merge_and_tables(three_models, pilot_prompts, tmp_path):
    rows, _, golds = three_models
    raw = [{k: v for k, v in r.items() if k not in ("human_label", "match")} for r in rows]
    mode = {"types": {t: {"mode": "auto"} for t in ("action", "direction", "turn_direction", "count")}}
    final = F.apply_mode(raw, mode)
    human = [r for r in final if r["final_source"] == "human_required"]
    assert human and all(r["final_label"] is None for r in human)

    items = F.make_review_sheet(final, pilot_prompts, audit_fraction=0.1, min_audit_per_type=20, seed=1)
    ids = [i["review_id"] for i in items]
    assert len(ids) == len(set(ids))
    assert sum(i["why"] == "human_required" for i in items) == len(human)
    audit = [r for r in final if r["audit"]]
    for t in mode["types"]:
        n = sum(r["final_source"] == "auto" and r["requirement_type"] == t for r in final)
        assert sum(r["requirement_type"] == t for r in audit) == min(n, 20)
        assert len({r["model"] for r in audit if r["requirement_type"] == t}) == 3   # spread over models
    assert "pass_fail" not in items[0] and "reason" not in items[0]                   # blind

    # a person fills the sheet in (here: with the known truth), leaving one row blank
    truth = {f"{m}|{pid}|{q['requirement_index']}": q["human_label"]
             for m, g in golds.items() for pid, e in g.items() for q in e["requirements"]}
    path = F.save_review_sheet(items, tmp_path / "review.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        sheet = list(csv.DictReader(f))
    for k, row in enumerate(sheet):
        row["human_label"] = "" if k == 0 else truth[row["review_id"]][0]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    reviews, problems = F.load_review_sheet(path)
    assert problems == [] and len(reviews) == len(items) - 1
    status = F.merge_reviews(final, reviews)
    assert status["pending_human"] <= 1

    rep = F.audit_report(final, verbose=False)
    assert set(rep) <= set(mode["types"])

    tables = F.score_tables(final, pilot_prompts)
    overall = {r["model"]: r for r in tables["overall"]}
    assert set(overall) == {"M1", "M2", "M3"}
    assert all(r["requirements"] == 61 for r in overall.values())
    caps = {r["capability"] for r in tables["by_capability"]}
    assert caps == {"C1", "C2", "C3", "C4", "C5"}
    files = F.save_tables(tables, tmp_path / "tables", {"audit": [dict(type=t, **s) for t, s in rep.items()]})
    assert "summary.md" in files and "by_type.csv" in files
    F.save_final_rows(final, tmp_path / "final" / "final_results.json")


def test_invalid_review_labels_are_reported(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("review_id,human_label\nA|x|0,P\nA|x|1,maybe\nA|x|2,\n", encoding="utf-8-sig")
    labels, problems = F.load_review_sheet(p)
    assert labels == {"A|x|0": "PASS"} and len(problems) == 1


def test_redecide_and_frozen_check(three_models, tmp_path):
    rows, _, _ = three_models
    strict = {"RotationEvaluator": {"min_turn_deg": 179.0}}
    new = F.redecide(rows, strict)
    turns = [r for r in new if r["requirement_type"] == "turn_direction"]
    assert turns and all(r["pass_fail"] == "FAIL" for r in turns)                # 90° turns now fail
    body = [r for r in new if r["requirement_type"] == "body_side"]
    assert [r["pass_fail"] for r in body] == [r["pass_fail"] for r in rows if r["requirement_type"] == "body_side"]
    path = F.save_mode(F.decide_modes(new), tmp_path / "m.json", frozen_thresholds=strict)
    problems = F.check_frozen(F.load_mode(path))
    assert len(problems) == 1 and "RotationEvaluator" in problems[0]            # code not updated yet


def test_body_side_is_calibrated_in_the_unified_notebook(three_models):
    """BodySide (not a RuleEvaluator) works with the shared grid search and redecide."""
    rows, _, _ = three_models
    cls = common.get_evaluator("BodySideEvaluator")
    best, table = V.grid_search(cls, rows, {"min_activity": [0.3, 0.7, 5.0], "side_margin": [0.25]})
    assert table[0]["n"] > 0 and best["min_activity"] != 5.0
    new = F.redecide(rows, {"BodySideEvaluator": {**cls.PROVISIONAL_THRESHOLDS, "min_activity": 50.0}})
    assert all(r["pass_fail"] == "FAIL" for r in new if r["requirement_type"] == "body_side")
