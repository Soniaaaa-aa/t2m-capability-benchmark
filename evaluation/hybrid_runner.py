"""
hybrid_runner.py — Primary -> VLM batch fallback wrapper.

Policy:
1. Run the existing primary evaluators exactly as before.
2. Keep every genuine Primary PASS/FAIL unchanged.
3. Collect Primary pass_fail=None requirements by motion.
4. Send all unresolved requirements from the SAME motion
   to VLMEvaluator.evaluate_batch().
5. Use ONE VLM API call per motion.
6. Return one final row per original requirement.

Primary evaluator behaviour is NOT modified.
"""

from __future__ import annotations

from collections import defaultdict

from common import (
    run_all_evaluators,
    load_motion,
    _json_safe,
)

from eval_vlm import VLMEvaluator


def run_all_evaluators_hybrid(
    evaluation_cases,
    human_gold_labels=None,
    config=None,
    vlm_evaluator=None,
):
    """
    Run Primary evaluators first, then use batched VLM fallback
    only for Primary pass_fail=None requirements.

    Important:
    - Primary PASS/FAIL is frozen.
    - VLM never overwrites a Primary PASS/FAIL.
    - All unresolved requirements from the SAME motion are sent
      together in ONE VLM API call.
    """

    # ========================================================
    # STEP 1 — RUN PRIMARY EVALUATORS
    # ========================================================

    primary_rows = run_all_evaluators(
        evaluation_cases=evaluation_cases,
        human_gold_labels=human_gold_labels,
        config=config,
    )

    vlm = (
        vlm_evaluator
        or VLMEvaluator.for_benchmark()
    )

    # ========================================================
    # STEP 2 — CASE LOOKUP
    # ========================================================
    #
    # Use (model, prompt_id), not only prompt_id.
    #
    # This prevents collisions if several models contain
    # the same prompt IDs.
    # ========================================================

    case_lookup = {
        (
            case.get("model"),
            case["prompt_id"],
        ): case
        for case in evaluation_cases
    }

    # ========================================================
    # STEP 3 — PREPARE FINAL ROWS
    # ========================================================

    final_rows = []

    # unresolved_groups:
    #
    # (model, prompt_id)
    #     -> list of information about unresolved requirements

    unresolved_groups = defaultdict(list)

    for row_index, primary in enumerate(primary_rows):

        row = dict(primary)

        # ----------------------------------------------------
        # Preserve Primary provenance explicitly.
        # ----------------------------------------------------

        row["primary_evaluator"] = (
            primary.get("evaluator")
        )

        row["primary_pass_fail"] = (
            primary.get("pass_fail")
        )

        row["vlm_pass_fail"] = None
        row["vlm_confidence"] = None
        row["vlm_reason"] = None
        row["final_source"] = None

        # ----------------------------------------------------
        # Primary PASS / FAIL is frozen.
        # ----------------------------------------------------

        if primary.get("pass_fail") in {
            "PASS",
            "FAIL",
        }:

            row["final_source"] = "PRIMARY"

            final_rows.append(row)

            continue

        # ----------------------------------------------------
        # Primary None -> prepare for VLM batch fallback.
        # ----------------------------------------------------

        model = primary.get("model")
        prompt_id = primary["prompt_id"]
        req_index = primary["requirement_index"]

        key = (
            model,
            prompt_id,
        )

        case = case_lookup.get(key)

        # Backward-compatible fallback:
        # if model is missing from the primary row,
        # try finding the unique prompt_id.

        if case is None:

            matches = [
                c
                for c in evaluation_cases
                if c["prompt_id"] == prompt_id
            ]

            if len(matches) == 1:
                case = matches[0]

                key = (
                    case.get("model"),
                    prompt_id,
                )

            else:
                row["vlm_error"] = (
                    "Could not uniquely resolve "
                    f"evaluation case for {key}"
                )

                row["final_source"] = "VLM_ERROR"

                final_rows.append(row)

                continue

        requirement = (
            case["requirements"][req_index]
        )

        # Save row first.
        final_index = len(final_rows)

        final_rows.append(row)

        # Add this unresolved requirement to its motion group.

        unresolved_groups[key].append({
            "final_row_index": final_index,
            "requirement_index": req_index,
            "requirement": requirement,
            "case": case,
        })

    # ========================================================
    # STEP 4 — BATCH VLM FALLBACK
    # ========================================================

    motion_cache = {}

    # Diagnostic counters.
    vlm_api_calls = 0
    vlm_requirement_count = 0

    for key, items in unresolved_groups.items():

        model, prompt_id = key

        case = items[0]["case"]

        # Requirements passed to evaluate_batch().

        batch_requirements = [
            {
                "requirement_index":
                    item["requirement_index"],

                "requirement":
                    item["requirement"],
            }
            for item in items
        ]

        print(
            f"[VLM BATCH] "
            f"{model} | "
            f"{prompt_id} | "
            f"{len(batch_requirements)} "
            f"unresolved requirement(s)"
        )

        try:

            # ------------------------------------------------
            # Load motion once.
            # ------------------------------------------------

            if key not in motion_cache:

                motion_cache[key] = load_motion(
                    case["motion_path"]
                )

            motion = motion_cache[key]

            # ------------------------------------------------
            # ONE API CALL FOR THIS MOTION
            # ------------------------------------------------

            batch_results = vlm.evaluate_batch(
                motion,
                batch_requirements,
                case,
            )

            vlm_api_calls += 1

            vlm_requirement_count += len(
                batch_requirements
            )

            # ------------------------------------------------
            # Map batch results back to original rows.
            # ------------------------------------------------

            for item in items:

                row_index = item[
                    "final_row_index"
                ]

                req_index = item[
                    "requirement_index"
                ]

                row = final_rows[
                    row_index
                ]

                vlm_result = (
                    batch_results.get(
                        req_index
                    )
                )

                # --------------------------------------------
                # Missing result from batch.
                # --------------------------------------------

                if vlm_result is None:

                    row["vlm_error"] = (
                        "Batch VLM returned no "
                        f"result for requirement "
                        f"index {req_index}"
                    )

                    row["final_source"] = (
                        "VLM_ERROR"
                    )

                    continue

                vlm_pf = vlm_result.get(
                    "pass_fail"
                )

                row["vlm_pass_fail"] = (
                    vlm_pf
                )

                row["vlm_confidence"] = (
                    vlm_result.get(
                        "evidence"
                    )
                    or {}
                ).get(
                    "confidence"
                )

                row["vlm_reason"] = (
                    vlm_result.get(
                        "reason"
                    )
                )

                row["vlm_evidence"] = (
                    _json_safe(
                        vlm_result.get(
                            "evidence"
                        )
                    )
                )

                # --------------------------------------------
                # Valid VLM PASS / FAIL
                # --------------------------------------------

                if vlm_pf in {
                    "PASS",
                    "FAIL",
                }:

                    # Keep standard fields compatible
                    # with existing summaries / savers.

                    row["evaluator"] = (
                        "VLMEvaluator"
                    )

                    row["status"] = (
                        "EVALUATED"
                    )

                    row["pass_fail"] = (
                        vlm_pf
                    )

                    row["score"] = (
                        vlm_result.get(
                            "score"
                        )
                    )

                    row[
                        "threshold_status"
                    ] = vlm_result.get(
                        "threshold_status"
                    )

                    row["reason"] = (
                        vlm_result.get(
                            "reason"
                        )
                    )

                    row["evidence"] = (
                        _json_safe(
                            vlm_result.get(
                                "evidence"
                            )
                        )
                    )

                    row["final_source"] = (
                        "VLM"
                    )

                    # ----------------------------------------
                    # Human Gold comparison,
                    # only when available.
                    # ----------------------------------------

                    human = row.get(
                        "human_label"
                    )

                    row["match"] = (
                        vlm_pf == human
                        if human
                        in {
                            "PASS",
                            "FAIL",
                        }
                        else None
                    )

                else:

                    row["final_source"] = (
                        "UNRESOLVED"
                    )

        except Exception as e:

            # ------------------------------------------------
            # Batch failed.
            #
            # Do NOT destroy Primary rows.
            # Only unresolved rows belonging to this motion
            # receive the VLM_ERROR marker.
            # ------------------------------------------------

            error_message = (
                f"{type(e).__name__}: {e}"
            )

            print(
                f"[VLM BATCH ERROR] "
                f"{model} | "
                f"{prompt_id} | "
                f"{error_message}"
            )

            for item in items:

                row_index = item[
                    "final_row_index"
                ]

                row = final_rows[
                    row_index
                ]

                row["vlm_error"] = (
                    error_message
                )

                row["final_source"] = (
                    "VLM_ERROR"
                )

    # ========================================================
    # STEP 5 — DIAGNOSTIC SUMMARY
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "HYBRID VLM BATCH SUMMARY"
    )

    print("=" * 70)

    print(
        "Unresolved requirements sent to VLM :",
        vlm_requirement_count,
    )

    print(
        "Actual VLM batch API calls           :",
        vlm_api_calls,
    )

    if vlm_api_calls > 0:

        print(
            "Average requirements per API call   :",
            round(
                vlm_requirement_count
                / vlm_api_calls,
                2,
            ),
        )

    print("=" * 70)

    return final_rows


# ============================================================
# STRICT PROMPT-LEVEL RESULTS
# ============================================================

def strict_prompt_results(
    final_requirement_rows,
    evaluation_cases=None,
):
    """
    Strict AND aggregation:

    - FAIL if any requirement FAILs
    - PASS only if every requirement PASSes
    - otherwise unresolved (None)

    evaluation_cases is optional so that this function
    remains compatible with existing notebook calls.
    """

    by_prompt = {}

    for row in final_requirement_rows:

        key = (
            row.get("model"),
            row["prompt_id"],
        )

        by_prompt.setdefault(
            key,
            []
        ).append(row)

    # --------------------------------------------------------
    # If evaluation_cases is supplied, preserve its order.
    # --------------------------------------------------------

    if evaluation_cases is not None:

        results = []

        for case in evaluation_cases:

            key = (
                case.get("model"),
                case["prompt_id"],
            )

            rows = by_prompt.get(
                key,
                []
            )

            statuses = [
                r.get("pass_fail")
                for r in rows
            ]

            if "FAIL" in statuses:

                verdict = "FAIL"

            elif (
                rows
                and all(
                    x == "PASS"
                    for x in statuses
                )
            ):

                verdict = "PASS"

            else:

                verdict = None

            results.append({
                "model":
                    case.get("model"),

                "prompt_id":
                    case["prompt_id"],

                "prompt":
                    case.get("prompt"),

                "capability":
                    case.get("capability"),

                "difficulty":
                    case.get("difficulty"),

                "requirement_count":
                    len(rows),

                "requirement_pass":
                    statuses.count(
                        "PASS"
                    ),

                "requirement_fail":
                    statuses.count(
                        "FAIL"
                    ),

                "requirement_none":
                    statuses.count(
                        None
                    ),

                "pass_fail":
                    verdict,
            })

        return results

    # --------------------------------------------------------
    # Compatibility mode:
    # build prompt results directly from final rows.
    # --------------------------------------------------------

    results = []

    for key, rows in by_prompt.items():

        model, prompt_id = key

        statuses = [
            r.get("pass_fail")
            for r in rows
        ]

        if "FAIL" in statuses:

            verdict = "FAIL"

        elif (
            rows
            and all(
                x == "PASS"
                for x in statuses
            )
        ):

            verdict = "PASS"

        else:

            verdict = None

        first = rows[0]

        results.append({
            "model":
                model,

            "prompt_id":
                prompt_id,

            "prompt":
                first.get("prompt"),

            "capability":
                first.get("capability"),

            "difficulty":
                first.get("difficulty"),

            "requirement_count":
                len(rows),

            "requirement_pass":
                statuses.count(
                    "PASS"
                ),

            "requirement_fail":
                statuses.count(
                    "FAIL"
                ),

            "requirement_none":
                statuses.count(
                    None
                ),

            "pass_fail":
                verdict,
        })

    return results
