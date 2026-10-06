"""
hybrid_runner.py — non-invasive Primary -> VLM fallback wrapper.

Existing common.py and run_all_evaluators() remain unchanged.

Policy:
1. Run the existing primary evaluators exactly as before.
2. Keep every genuine Primary PASS/FAIL unchanged.
3. Route only Primary pass_fail=None rows to VLMEvaluator.
4. Return one final row per original requirement.
"""

from __future__ import annotations

from common import (
    run_all_evaluators,
    load_motion,
    get_human_label,
    _json_safe,
)
from eval_vlm import VLMEvaluator


def run_all_evaluators_hybrid(
    evaluation_cases,
    human_gold_labels=None,
    config=None,
    vlm_evaluator=None,
):
    primary_rows = run_all_evaluators(
        evaluation_cases=evaluation_cases,
        human_gold_labels=human_gold_labels,
        config=config,
    )

    vlm = vlm_evaluator or VLMEvaluator.for_benchmark()

    case_lookup = {
        case["prompt_id"]: case
        for case in evaluation_cases
    }
    motion_cache = {}
    final_rows = []

    for primary in primary_rows:
        row = dict(primary)

        # Preserve Primary provenance explicitly.
        row["primary_evaluator"] = primary.get("evaluator")
        row["primary_pass_fail"] = primary.get("pass_fail")
        row["vlm_pass_fail"] = None
        row["vlm_confidence"] = None
        row["vlm_reason"] = None
        row["final_source"] = None

        # Primary PASS/FAIL is frozen and is never overwritten.
        if primary.get("pass_fail") in {"PASS", "FAIL"}:
            row["final_source"] = "PRIMARY"
            final_rows.append(row)
            continue

        prompt_id = primary["prompt_id"]
        req_index = primary["requirement_index"]
        case = case_lookup[prompt_id]
        requirement = case["requirements"][req_index]

        try:
            if prompt_id not in motion_cache:
                motion_cache[prompt_id] = load_motion(case["motion_path"])

            vlm_result = vlm.evaluate(
                motion_cache[prompt_id],
                requirement,
                case,
            )

            vlm_pf = vlm_result.get("pass_fail")

            row["vlm_pass_fail"] = vlm_pf
            row["vlm_confidence"] = (
                vlm_result.get("evidence") or {}
            ).get("confidence")
            row["vlm_reason"] = vlm_result.get("reason")
            row["vlm_evidence"] = _json_safe(vlm_result.get("evidence"))

            if vlm_pf in {"PASS", "FAIL"}:
                # Keep standard fields compatible with existing summaries/savers.
                row["evaluator"] = "VLMEvaluator"
                row["status"] = "EVALUATED"
                row["pass_fail"] = vlm_pf
                row["score"] = vlm_result.get("score")
                row["threshold_status"] = vlm_result.get("threshold_status")
                row["reason"] = vlm_result.get("reason")
                row["evidence"] = _json_safe(vlm_result.get("evidence"))
                row["final_source"] = "VLM"

                human = row.get("human_label")
                row["match"] = (
                    vlm_pf == human
                    if human in {"PASS", "FAIL"}
                    else None
                )
            else:
                row["final_source"] = "UNRESOLVED"

        except Exception as e:
            # Do not destroy the Primary row if VLM fails.
            row["vlm_error"] = f"{type(e).__name__}: {e}"
            row["final_source"] = "VLM_ERROR"

        final_rows.append(row)

    return final_rows


def strict_prompt_results(final_requirement_rows, evaluation_cases):
    """
    Strict AND aggregation:
    - FAIL if any requirement FAILs
    - PASS only if every requirement PASSes
    - otherwise unresolved (None)
    """
    by_prompt = {}
    for row in final_requirement_rows:
        by_prompt.setdefault(row["prompt_id"], []).append(row)

    results = []
    for case in evaluation_cases:
        rows = by_prompt.get(case["prompt_id"], [])
        statuses = [r.get("pass_fail") for r in rows]

        if "FAIL" in statuses:
            verdict = "FAIL"
        elif rows and all(x == "PASS" for x in statuses):
            verdict = "PASS"
        else:
            verdict = None

        results.append({
            "model": case.get("model"),
            "prompt_id": case["prompt_id"],
            "prompt": case.get("prompt"),
            "capability": case.get("capability"),
            "difficulty": case.get("difficulty"),
            "requirement_count": len(rows),
            "requirement_pass": statuses.count("PASS"),
            "requirement_fail": statuses.count("FAIL"),
            "requirement_none": statuses.count(None),
            "pass_fail": verdict,
        })

    return results
