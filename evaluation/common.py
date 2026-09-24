"""
common.py — shared benchmark code (owned by the whole team)
============================================================

Everything that every evaluator depends on lives here:

- benchmark settings and the Standardised Motion input contract,
- loading the benchmark definition and the motion package,
- evaluation-case construction,
- the common Evaluator interface (BaseEvaluator) and registry,
- Human Gold Label helpers.

Changing this file affects every evaluator: open a Pull Request and ask all
members to review it. Evaluator-specific code belongs in eval_<name>.py.

Origin: STEP 2–9F of Benchmark_Df5_English.ipynb, moved without changing the
behaviour (cell outputs are printed by the same functions when verbose=True).
"""

from __future__ import annotations

import json
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


# ============================================================
# STEP 2 — Benchmark common settings
# ============================================================

EXPECTED_JOINTS = 22
EXPECTED_DIMS = 3
DEFAULT_FPS = 20.0

# Requirement Type -> Evaluator mapping (STEP 8)
EVALUATION_CONFIG = {
    "action": "ActionEvaluator",
    "direction": "TrajectoryEvaluator",
    "torso_direction": "TorsoGeometryEvaluator",
    "body_side": "BodySideEvaluator",
    "arm_direction": "LimbGeometryEvaluator",
    "leg_direction": "LimbGeometryEvaluator",
    "turn_direction": "RotationEvaluator",
    "count": "CountEvaluator",
    "order": "OrderEvaluator",
    "attribute": "AttributeEvaluator",
    "simultaneous": "SimultaneousEvaluator",
    "relation": "SpatialRelationEvaluator",
    "target": "SpatialRelationEvaluator",
}

# Derived from EVALUATION_CONFIG so the two lists can never disagree.
SUPPORTED_REQUIREMENT_TYPES = set(EVALUATION_CONFIG)


# ============================================================
# STEP 3 — Benchmark Input Contract
# ============================================================

def validate_standardised_motion(motion_xyz):
    """Validate the numerical parts of the benchmark input contract."""
    if not isinstance(motion_xyz, np.ndarray):
        raise TypeError(f"Expected NumPy array, but received {type(motion_xyz)}")
    if motion_xyz.ndim != 3:
        raise ValueError(f"Expected [T, 22, 3], but received {motion_xyz.shape}")
    if motion_xyz.shape[1:] != (EXPECTED_JOINTS, EXPECTED_DIMS):
        raise ValueError(f"Expected [T, 22, 3], but received {motion_xyz.shape}")
    if motion_xyz.shape[0] < 1:
        raise ValueError("Motion contains no frames.")
    if not np.isfinite(motion_xyz).all():
        raise ValueError("Motion contains NaN or Inf values.")
    return motion_xyz


def load_motion(motion_path):
    """Load and validate one standardised motion file."""
    return validate_standardised_motion(np.load(motion_path, allow_pickle=False))


# ============================================================
# STEP 3.5 — Motion package
# ============================================================

def extract_motion_package(zip_path, input_root, verbose=True):
    """Extract a standardised motion ZIP (<Model>/<prompt_id>.npy) into input_root."""
    input_root = Path(input_root)
    input_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(input_root)

    files = sorted(p for p in input_root.rglob("*") if p.is_file())
    if verbose:
        print("\nPackage extracted ✅")
        print("ZIP        :", Path(zip_path).name)
        print("Destination:", input_root)
        print("\nExtracted files:")
        for path in files:
            print(" -", path.relative_to(input_root))
    return files


# ============================================================
# STEP 5 — Benchmark definition
# ============================================================

def parse_benchmark_definition(data, source="benchmark definition"):
    """Parse benchmark-definition JSON (bytes, str or dict). Returns (definition, prompts)."""
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as e:
            raise ValueError(f"Failed to load JSON:\n{source}\n\n{e}")
    if "prompts" not in data:
        raise KeyError("The JSON does not contain 'prompts'.")
    return data, data["prompts"]


def load_benchmark_definition(path):
    """Load a benchmark-definition JSON file. Returns (definition, prompts)."""
    path = Path(path)
    return parse_benchmark_definition(path.read_bytes(), source=str(path))


def print_benchmark_summary(definition, prompts, source):
    print("\n========================================")
    print("Pilot Benchmark Definition")
    print("========================================")
    print("File      :", source)
    print("Benchmark :", definition.get("benchmark_name", "N/A"))
    print("Version   :", definition.get("schema_version", "N/A"))
    print("Prompts   :", len(prompts))
    print("----------------------------------------")
    for prompt in prompts:
        print(
            f'{prompt["prompt_id"]:7} | '
            f'{prompt["capability"]} | '
            f'{prompt["difficulty"]:6} | '
            f'{prompt["text"]}'
        )
    print("========================================")


# ============================================================
# STEP 6 — Load and validate pre-generated motions
# ============================================================

def register_motions(prompts, model_name, model_input_dir, verbose=True):
    """
    Load <model_input_dir>/<prompt_id>.npy for every prompt and validate it.
    Returns (registered_motions, input_results).
    """
    model_input_dir = Path(model_input_dir)
    registered_motions = []
    input_results = []

    def say(*args):
        if verbose:
            print(*args)

    say("========================================")
    say("Pre-generated Motion Input")
    say("========================================")
    say("Model        :", model_name)
    say("Input folder :", model_input_dir)
    say("Total prompts:", len(prompts))
    say("========================================\n")

    for index, prompt_data in enumerate(prompts, start=1):
        prompt_id = prompt_data["prompt_id"]
        prompt_text = prompt_data["text"]
        capability = prompt_data["capability"]
        difficulty = prompt_data["difficulty"]
        target_frames = prompt_data.get("target_frames_20fps")
        motion_path = model_input_dir / f"{prompt_id}.npy"

        say("----------------------------------------")
        say(f"[{index}/{len(prompts)}]")
        say("Prompt ID :", prompt_id)
        say("Prompt    :", prompt_text)
        say("File      :", motion_path)

        try:
            if not motion_path.exists():
                raise FileNotFoundError(f"Input motion not found: {motion_path}")

            validated_motion = load_motion(motion_path)

            # Target frame length is reported, not forced here.
            frame_match = (
                None if target_frames is None
                else int(validated_motion.shape[0]) == int(target_frames)
            )

            record = {
                "model": model_name,
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "capability": capability,
                "difficulty": difficulty,
                "target_frames": target_frames,
                "motion_path": str(motion_path),
                "shape": list(validated_motion.shape),
                "frame_match": frame_match,
                "validation": "PASS",
            }
            registered_motions.append(record)
            input_results.append(record)

            say("Loaded     :", validated_motion.shape)
            say("Frame match:", frame_match)
            say("Validation : PASS ✅")

        except Exception as e:
            input_results.append({
                "model": model_name,
                "prompt_id": prompt_id,
                "motion_path": str(motion_path),
                "validation": "FAIL",
                "error": str(e),
            })
            say("Validation : FAIL ❌")
            say("Error      :", e)

    say("\n========================================")
    say("Input Validation Summary")
    say("========================================")
    say("PASS:", len(registered_motions))
    say("FAIL:", len(prompts) - len(registered_motions))
    say("========================================")

    return registered_motions, input_results


# ============================================================
# STEP 7 — Evaluation case construction
# ============================================================

def build_evaluation_cases(prompts, registered_motions, model_name, verbose=True):
    motion_registry = {m["prompt_id"]: m for m in registered_motions}
    evaluation_cases = []

    for prompt_data in prompts:
        prompt_id = prompt_data["prompt_id"]
        if prompt_id not in motion_registry:
            if verbose:
                print(f"{prompt_id}: Motion not available ⚠️")
            continue

        motion_data = motion_registry[prompt_id]
        evaluation_cases.append({
            "model": motion_data["model"],
            "prompt_id": prompt_id,
            "prompt": prompt_data["text"],
            "capability": prompt_data["capability"],
            "difficulty": prompt_data["difficulty"],
            "motion_path": motion_data["motion_path"],
            "motion_shape": motion_data["shape"],
            "requirements": prompt_data["requirements"],
            "diagnostic_axes": prompt_data.get("diagnostic_axes", []),
        })

    if verbose:
        print("========================================")
        print("Evaluation Case Construction")
        print("========================================")
        print("Model            :", model_name)
        print("Benchmark Prompts:", len(prompts))
        print("Evaluation Cases :", len(evaluation_cases))
        print("========================================")
        if evaluation_cases:
            example = evaluation_cases[0]
            print("Example:", example["prompt_id"], example["motion_path"], example["motion_shape"])

    return evaluation_cases


# ============================================================
# STEP 8 — Evaluation configuration check
# ============================================================

def check_evaluation_config(evaluation_cases, config=None, verbose=True):
    """Returns (used_requirement_types, missing_types, registered_status)."""
    config = EVALUATION_CONFIG if config is None else config

    used_requirement_types = sorted({
        requirement["type"]
        for case in evaluation_cases
        for requirement in case["requirements"]
    })
    missing_types = [t for t in used_requirement_types if t not in config]
    status = {
        t: (config.get(t), config.get(t) in EVALUATOR_REGISTRY)
        for t in used_requirement_types
    }

    if verbose:
        print("========================================")
        print("Evaluation Configuration")
        print("========================================")
        for requirement_type, (evaluator, implemented) in status.items():
            mark = "✅ implemented" if implemented else "— not yet implemented"
            print(f"{requirement_type:<18} → {str(evaluator):<26} {mark}")
        print("========================================")
        if not missing_types:
            print("All Requirement Types have an evaluator assigned. ✅")
        else:
            print("Some Requirement Types do not have an evaluator assigned. ⚠️")
            for requirement_type in missing_types:
                print(" -", requirement_type)

    return used_requirement_types, missing_types, status


# ============================================================
# STEP 9 — Evaluator interface / registry
# ============================================================

class BaseEvaluator(ABC):
    """
    Common interface for all Requirement Evaluators.
    """

    @abstractmethod
    def evaluate(self, motion, requirement, evaluation_case):
        """
        Evaluate one requirement.

        Parameters
        ----------
        motion : np.ndarray
            Standardised motion with shape [T, 22, 3].

        requirement : dict
            Requirement definition from the benchmark JSON.

        evaluation_case : dict
            Prompt-level evaluation information.

        Returns
        -------
        dict
            Requirement-level evaluation result. Recommended fields:
            requirement_id, requirement_type, expected_value, evaluator_name,
            pass_fail, score, evidence, reason.
        """
        pass


EVALUATOR_REGISTRY = {}


def register_evaluator(name, evaluator_class):
    """
    Register an evaluator class.
    """
    if not issubclass(evaluator_class, BaseEvaluator):
        raise TypeError(
            f"{evaluator_class.__name__} must inherit from BaseEvaluator."
        )
    EVALUATOR_REGISTRY[name] = evaluator_class


def get_evaluator(name):
    """
    Retrieve an evaluator class from the registry.
    """
    if name not in EVALUATOR_REGISTRY:
        raise KeyError(
            f"Evaluator '{name}' is not registered."
        )
    return EVALUATOR_REGISTRY[name]


# ============================================================
# STEP 9C–9F — Human Gold Labels
# ============================================================

VALID_LABELS = {
    "P": "PASS",
    "F": "FAIL",
    "U": "UNCERTAIN",
}


def requirement_value(req):
    return req.get("value", req.get("expected", req.get("target")))


def make_gold_template(evaluation_cases, verbose=True):
    """STEP 9C — one entry per Requirement, human_label = None."""
    if verbose:
        print("=" * 90)
        print("HUMAN GOLD LABEL TEMPLATE")
        print("=" * 90)
        print(f"\nTotal Evaluation Cases: {len(evaluation_cases)}")

    human_gold_template = {}

    for case in evaluation_cases:
        prompt_id = (
            case.get("prompt_id") or case.get("id") or case.get("case_id") or "UNKNOWN"
        )
        prompt_text = (
            case.get("prompt") or case.get("text") or case.get("prompt_text") or ""
        )
        requirements = case.get("requirements", [])

        human_gold_template[prompt_id] = {"prompt": prompt_text, "requirements": []}

        if verbose:
            print("\n" + "=" * 90)
            print(f"Prompt ID : {prompt_id}")
            if prompt_text:
                print(f"Prompt    : {prompt_text}")
            print(f"Requirements: {len(requirements)}")
            print("-" * 90)

        for i, req in enumerate(requirements):
            req_type = req.get("type", "unknown")
            req_value = requirement_value(req)
            human_gold_template[prompt_id]["requirements"].append({
                "requirement_index": i,
                "type": req_type,
                "value": req_value,
                "human_label": None,
            })
            if verbose:
                print(f"[{i}] {req_type:<15} = {str(req_value):<25} Human Gold: ?")

    if verbose:
        print("\n" + "=" * 90)
        print("Template created.")
        print("Human Gold Labels have not been entered yet.")
        print("=" * 90)

    return human_gold_template


def enter_gold_labels(human_gold_template, model_name, input_fn=input):
    """STEP 9D — interactive P/F/U entry."""
    human_gold_labels = {}

    print("=" * 90)
    print(f"HUMAN GOLD LABEL ENTRY — {model_name}")
    print("=" * 90)

    for prompt_id, data in human_gold_template.items():
        print("\n" + "=" * 90)
        print(f"Prompt ID : {prompt_id}")
        print(f"Prompt    : {data['prompt']}")
        print("-" * 90)

        human_gold_labels[prompt_id] = {"prompt": data["prompt"], "requirements": []}

        for req in data["requirements"]:
            idx, typ, value = req["requirement_index"], req["type"], req["value"]
            while True:
                x = input_fn(f"[{idx}] {typ} = {value} → Human Gold [P/F/U]: ").strip().upper()
                if x in VALID_LABELS:
                    break
                print("Invalid input. Please enter P, F, or U.")

            human_gold_labels[prompt_id]["requirements"].append({
                "requirement_index": idx,
                "type": typ,
                "value": value,
                "human_label": VALID_LABELS[x],
            })

    print("\nHuman Gold Label entry completed.")
    return human_gold_labels


def gold_label_filename(model_name, split="pilot"):
    return f"{model_name}_{split}_human_gold_labels.json"


def save_gold_labels(human_gold_labels, path):
    """STEP 9E — save labels as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(human_gold_labels, f, indent=2, ensure_ascii=False)
    return path


def load_gold_labels(path):
    """STEP 9F — load and validate labels. Returns (labels, counts)."""
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return labels, count_gold_labels(labels)


def count_gold_labels(labels):
    counts = {"PASS": 0, "FAIL": 0, "UNCERTAIN": 0, "total": 0}
    for prompt_id, data in labels.items():
        for req in data["requirements"]:
            label = req.get("human_label")
            if label not in ("PASS", "FAIL", "UNCERTAIN"):
                raise ValueError(f"Invalid Human Gold Label: {prompt_id} -> {label}")
            counts[label] += 1
            counts["total"] += 1
    return counts


def get_human_label(human_gold_labels, case, requirement_index):
    """
    Human Gold Label for (Prompt ID, Requirement Index).

    Raises when the prompt or index is missing, or when the stored Requirement
    does not match the benchmark definition (e.g. labels made for an older JSON).
    """
    prompt_id = case.get("prompt_id") or case.get("id") or case.get("case_id")
    if prompt_id is None:
        raise ValueError("Could not obtain Prompt ID from Evaluation Case.")
    if prompt_id not in human_gold_labels:
        raise KeyError(f"Human Gold Label not found: {prompt_id}")

    gold_requirements = human_gold_labels[prompt_id]["requirements"]
    if requirement_index >= len(gold_requirements):
        raise IndexError(
            f"{prompt_id}: Requirement Index {requirement_index} is not present in Human Gold."
        )

    gold_req = gold_requirements[requirement_index]
    case_req = case["requirements"][requirement_index]

    case_type = case_req.get("type")
    case_value = case_req.get("value", case_req.get("expected"))
    gold_type = gold_req.get("type")
    gold_value = gold_req.get("value")

    if case_type != gold_type or case_value != gold_value:
        raise ValueError(
            f"\nRequirement mismatch detected.\n"
            f"Prompt ID : {prompt_id}\n"
            f"Index     : {requirement_index}\n"
            f"Case      : {case_type} = {case_value}\n"
            f"Human Gold: {gold_type} = {gold_value}"
        )

    return gold_req["human_label"]


def iter_requirements(evaluation_cases, requirement_type):
    """Yield (case, requirement_index, requirement) for one requirement type."""
    for case in evaluation_cases:
        for req_index, req in enumerate(case["requirements"]):
            if req.get("type") == requirement_type:
                yield case, req_index, req
