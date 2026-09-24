import json
import zipfile

import numpy as np
import pytest

import common
import synth_motion as sm


def make_prompts():
    return [
        {"prompt_id": "C1-01", "set": "Pilot", "capability": "C1", "difficulty": "Easy",
         "text": "A person walks forward.", "target_frames_20fps": 100,
         "requirements": [{"type": "action", "value": "walk"}, {"type": "direction", "value": "forward"}]},
        {"prompt_id": "C2-09", "set": "Pilot", "capability": "C2", "difficulty": "Medium",
         "text": "A person kicks the left leg out to the side.", "target_frames_20fps": 100,
         "requirements": [{"type": "action", "value": "kick"}, {"type": "body_side", "value": "left"}]},
    ]


def test_validate_standardised_motion():
    common.validate_standardised_motion(np.zeros((5, 22, 3)))
    for bad in (np.zeros((5, 21, 3)), np.zeros((0, 22, 3)), np.full((5, 22, 3), np.nan), [[0]]):
        with pytest.raises((ValueError, TypeError)):
            common.validate_standardised_motion(bad)


def test_config_and_supported_types_agree():
    assert common.SUPPORTED_REQUIREMENT_TYPES == set(common.EVALUATION_CONFIG)


def test_package_to_evaluation_cases(tmp_path):
    prompts = make_prompts()
    pkg = tmp_path / "pkg" / "ModelX"
    pkg.mkdir(parents=True)
    np.save(pkg / "C1-01.npy", sm.standardise(sm.walk(100)))  # C2-09 deliberately missing
    zpath = tmp_path / "ModelX.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(pkg / "C1-01.npy", "ModelX/C1-01.npy")

    root = tmp_path / "inputs"
    common.extract_motion_package(zpath, root, verbose=False)
    registered, results = common.register_motions(prompts, "ModelX", root / "ModelX", verbose=False)
    assert [r["prompt_id"] for r in registered] == ["C1-01"]
    assert registered[0]["frame_match"] is True
    assert [r["validation"] for r in results] == ["PASS", "FAIL"]

    cases = common.build_evaluation_cases(prompts, registered, "ModelX", verbose=False)
    assert len(cases) == 1 and cases[0]["model"] == "ModelX"


def test_benchmark_definition_parsing(tmp_path):
    p = tmp_path / "def.json"
    p.write_text(json.dumps({"benchmark_name": "x", "prompts": make_prompts()}))
    definition, prompts = common.load_benchmark_definition(p)
    assert len(prompts) == 2
    with pytest.raises(KeyError):
        common.parse_benchmark_definition(b'{"nope": 1}')


def test_gold_label_roundtrip_and_lookup(tmp_path):
    cases = [{"prompt_id": p["prompt_id"], "prompt": p["text"], "requirements": p["requirements"]}
             for p in make_prompts()]
    template = common.make_gold_template(cases, verbose=False)
    answers = iter(["X", "P", "F", "U", "P"])  # first answer invalid -> re-asked
    labels = common.enter_gold_labels(template, "ModelX", input_fn=lambda _: next(answers))
    path = common.save_gold_labels(labels, tmp_path / common.gold_label_filename("ModelX"))
    loaded, counts = common.load_gold_labels(path)
    assert counts == {"PASS": 2, "FAIL": 1, "UNCERTAIN": 1, "total": 4}
    assert common.get_human_label(loaded, cases[1], 1) == "P".replace("P", "PASS")

    # Labels made for a different definition are rejected, not silently used.
    changed = json.loads(json.dumps(cases[1]))
    changed["requirements"][1]["value"] = "right"
    with pytest.raises(ValueError):
        common.get_human_label(loaded, changed, 1)


def test_registry_rejects_non_evaluator():
    class NotAnEvaluator:
        pass
    with pytest.raises(TypeError):
        common.register_evaluator("X", NotAnEvaluator)
    with pytest.raises(KeyError):
        common.get_evaluator("DoesNotExist")
