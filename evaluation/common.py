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
    "direction": "BodyFrameDirectionEvaluator",   # body frame (eval_trajectory_ext.py); world frame: "TrajectoryEvaluator"
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

    @classmethod
    def for_benchmark(cls):
        """
        The instance used by run_all_evaluators(): configured with the
        evaluator's CURRENT rule / thresholds (defined in its own eval_*.py).
        Override when the constructor needs arguments.
        """
        return cls()


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


# ============================================================
# Evaluator discovery — no notebook edit needed for new evaluators
# ============================================================

def load_all_evaluators(eval_dir=None, verbose=True):
    """
    Import (or re-import) every evaluation/eval_*.py except eval_template.py.
    Each module registers its evaluator(s) when imported, so adding a new
    eval_<name>.py file is enough — run_benchmark.ipynb does not change.

    Returns the list of imported module names.
    """
    import importlib
    import sys

    eval_dir = Path(eval_dir) if eval_dir else Path(__file__).resolve().parent
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    names = sorted(p.stem for p in eval_dir.glob("eval_*.py") if p.stem != "eval_template")
    loaded = []
    for name in names:
        if name in sys.modules:
            importlib.reload(sys.modules[name])
        else:
            importlib.import_module(name)
        loaded.append(name)

    if verbose:
        print("Evaluator modules    :", ", ".join(loaded) or "(none)")
        print("Registered evaluators:", ", ".join(sorted(EVALUATOR_REGISTRY)) or "(none)")
    return loaded


# ============================================================
# Unified evaluation — every Requirement, every registered evaluator
# ============================================================

def _json_safe(x):
    """Drop large arrays and convert NumPy scalars so results can be saved as JSON."""
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return None if x.size > 16 else x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def run_all_evaluators(evaluation_cases, human_gold_labels=None, config=None):
    """
    Evaluate every Requirement of every case with the evaluator mapped in
    EVALUATION_CONFIG, using each evaluator's current rule (for_benchmark()).

    Requirement types whose evaluator is not implemented yet are listed with
    status NOT_IMPLEMENTED; an exception inside one evaluator is recorded as
    status ERROR instead of stopping the whole run.

    Returns one row per Requirement.
    """
    config = EVALUATION_CONFIG if config is None else config
    instances = {}
    rows = []

    for case in evaluation_cases:
        motion = None
        for req_index, req in enumerate(case["requirements"]):
            req_type = req.get("type")
            evaluator_name = config.get(req_type)

            human = None
            if human_gold_labels and case["prompt_id"] in human_gold_labels:
                human = get_human_label(human_gold_labels, case, req_index)

            row = {
                "model": case.get("model"),
                "prompt_id": case["prompt_id"],
                "requirement_index": req_index,
                "requirement_type": req_type,
                "expected_value": requirement_value(req),
                "evaluator": evaluator_name,
                "status": None,
                "pass_fail": None,
                "score": None,
                "threshold_status": None,
                "human_label": human,
                "match": None,
                "reason": None,
                "evidence": None,
            }

            if evaluator_name not in EVALUATOR_REGISTRY:
                row["status"] = "NOT_IMPLEMENTED" if evaluator_name else "NO_EVALUATOR_MAPPED"
                rows.append(row)
                continue

            try:
                if evaluator_name not in instances:
                    instances[evaluator_name] = EVALUATOR_REGISTRY[evaluator_name].for_benchmark()
                if motion is None:
                    motion = load_motion(case["motion_path"])
                result = instances[evaluator_name].evaluate(motion, req, case)
                row.update({
                    "status": "EVALUATED",
                    "pass_fail": result.get("pass_fail"),
                    "score": result.get("score"),
                    "threshold_status": result.get("threshold_status"),
                    "reason": result.get("reason"),
                    "evidence": _json_safe(result.get("evidence")),
                })
            except Exception as e:  # keep going; report per requirement
                row.update({"status": "ERROR", "reason": f"{type(e).__name__}: {e}"})

            if row["pass_fail"] in ("PASS", "FAIL") and human in ("PASS", "FAIL"):
                row["match"] = row["pass_fail"] == human
            rows.append(row)

    return rows


def summarize_results(rows, verbose=True):
    """Per requirement type: counts, PASS rate and agreement with Human Gold."""
    summary = {}
    for r in rows:
        s = summary.setdefault(r["requirement_type"], {
            "evaluator": r["evaluator"], "total": 0, "evaluated": 0, "PASS": 0, "FAIL": 0,
            "not_implemented": 0, "errors": 0, "labelled": 0, "agree": 0,
        })
        s["total"] += 1
        if r["status"] == "EVALUATED":
            s["evaluated"] += 1
            if r["pass_fail"] in ("PASS", "FAIL"):
                s[r["pass_fail"]] += 1
        elif r["status"] == "ERROR":
            s["errors"] += 1
        else:
            s["not_implemented"] += 1
        if r["match"] is not None:
            s["labelled"] += 1
            s["agree"] += int(r["match"])

    if verbose:
        header = (f"{'Requirement type':<18} {'Evaluator':<26} {'Total':>5} {'PASS':>5} {'FAIL':>5} "
                  f"{'Not impl.':>9} {'Errors':>6} {'Agree w/ Human':>15}")
        print(header)
        print("-" * len(header))
        for t, s in sorted(summary.items()):
            agree = f"{s['agree']}/{s['labelled']}" if s["labelled"] else "-"
            print(f"{t:<18} {str(s['evaluator']):<26} {s['total']:>5} {s['PASS']:>5} {s['FAIL']:>5} "
                  f"{s['not_implemented']:>9} {s['errors']:>6} {agree:>15}")
    return summary


def save_results(rows, path, metadata=None):
    """Save unified results as JSON (with metadata) and a flat CSV next to it."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata or {}, "results": _json_safe(rows)}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = path.with_suffix(".csv")
    fields = ["model", "prompt_id", "requirement_index", "requirement_type", "expected_value",
              "evaluator", "status", "pass_fail", "score", "threshold_status", "human_label",
              "match", "reason"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (json.dumps(r[k]) if isinstance(r.get(k), (list, dict)) else r.get(k))
                             for k in fields})
    return path, csv_path


# ============================================================
# Input loading for analysis notebooks (one call instead of STEP 3.5–9F)
# ============================================================

def load_inputs(model_name, repo_dir, benchmark_file="pilot_benchmark_definition.json",
                input_root="/content/benchmark_inputs", verbose=True):
    """
    Prepare everything an evaluator-analysis notebook needs:
    motion package (uploaded if missing), benchmark definition (repo, or
    uploaded), registered motions, evaluation cases and — if
    labels/<model>_pilot_human_gold_labels.json exists — Human Gold Labels.

    Returns a dict with keys: model_name, model_input_dir, benchmark_definition,
    prompts, registered_motions, evaluation_cases, human_gold_labels, gold_file.
    """
    repo_dir = Path(repo_dir)
    input_root = Path(input_root)
    model_input_dir = input_root / model_name

    def _upload(kind):
        from google.colab import files  # only available in Colab
        print(f"Upload {kind}")
        uploaded = files.upload()
        if not uploaded:
            raise RuntimeError(f"No {kind} was uploaded.")
        return uploaded

    if not model_input_dir.exists():
        uploaded = _upload(f"the standardized motion ZIP for {model_name}")
        zips = [n for n in uploaded if n.lower().endswith(".zip")]
        if len(zips) != 1:
            raise RuntimeError(f"Expected exactly 1 ZIP file, found {len(zips)}.")
        extract_motion_package(zips[0], input_root, verbose=False)
    if not model_input_dir.exists():
        raise FileNotFoundError(f"Model input folder not found: {model_input_dir}")

    bench_path = repo_dir / "benchmark" / benchmark_file
    if bench_path.exists():
        definition, prompts = load_benchmark_definition(bench_path)
        source = str(bench_path.relative_to(repo_dir))
    else:
        uploaded = _upload(f"the benchmark definition JSON ({benchmark_file} is not in the repository)")
        source = next(iter(uploaded))
        definition, prompts = parse_benchmark_definition(uploaded[source], source)

    registered, _ = register_motions(prompts, model_name, model_input_dir, verbose=False)
    cases = build_evaluation_cases(prompts, registered, model_name, verbose=False)

    gold_file = repo_dir / "labels" / gold_label_filename(model_name)
    labels = load_gold_labels(gold_file)[0] if gold_file.exists() else None

    if verbose:
        print("Model            :", model_name)
        print("Benchmark        :", source, "|", definition.get("schema_version", "N/A"))
        print("Motions loaded   :", f"{len(registered)}/{len(prompts)}")
        print("Evaluation cases :", len(cases))
        print("Human Gold Labels:", gold_file.relative_to(repo_dir) if labels else "not found (evidence only)")

    return {
        "model_name": model_name, "model_input_dir": model_input_dir,
        "benchmark_definition": definition, "prompts": prompts,
        "registered_motions": registered, "evaluation_cases": cases,
        "human_gold_labels": labels, "gold_file": gold_file,
    }
