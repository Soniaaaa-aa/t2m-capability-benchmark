# Automated evaluation — rules + plan B workflow

Implements the plan document "T2M Benchmark 混合评估方案", simplified to **plan B**:

* every requirement type is judged by **geometric rules** (no embedding model, no VLM);
* the rules are calibrated **once** on the Pilot against Human Gold and frozen;
* a type that does not reach the agreement gate is **not** improved — it is labelled by people in Main;
* in Main, people also label a blind 10 % audit sample of the automatic types.

**Additive**: `eval_trajectory.py` (member A), `eval_body_side.py`, `run_benchmark.ipynb` and the generation
notebooks are unchanged. The only edit to `common.py` is one line (2026-09-26, team decision):
`EVALUATION_CONFIG["direction"] = "BodyFrameDirectionEvaluator"` — direction is judged in the body frame. `run_benchmark.ipynb` picks the new `eval_*.py` files up
automatically, so its STEP 10 evaluates every requirement type.

## Workflow

```
Pilot (each member, own model)          Calibration (once)                   Main (once, all models)
run_benchmark.ipynb                     analysis/analysis_rules.ipynb        run_main.ipynb
  STEP 9  label the Pilot (P/F/U)   ──►   agreement, synthetic negatives,  ──►  lint definition
  STEP 11 <Model>_pilot_results.json      grid search, leave-one-model-out     run all models
  + labels/<Model>_pilot_human_...        auto / human per type                review_sheet.csv ─► people
                                          → evaluation_mode.json               merge, audit, tables
                                          → CURRENT_THRESHOLDS, PR,            → main_outputs.zip
                                            tag v1.0-pilot-frozen
```

## Files

| Layer | File | Contents | Owner |
|---|---|---|---|
| measurements | `primitives.py` | `body_frame`, `heading_yaw` (+ = left), `root_track`, `foot_contacts`, `effector_tracks`, `limb_deviation`, `displacement_in_frame` | A / B |
| events | `events.py` | `detect_events` → `Event(kind, start, end, side, attrs)` for walk, turn, jump, kick, reach, raise_hand, raise_hands; `requirement_context` (v1.2 `applies_to`, `event_sequence`, `event_group`) | C |
| plumbing | `rule_base.py` | `RuleEvaluator`: `PROVISIONAL_THRESHOLDS` / `CURRENT_THRESHOLDS` / evidence-only mode | all |
| turn_direction | `eval_rotation.py` | `RotationEvaluator` | A |
| direction, attribute | `eval_trajectory_ext.py` | `BodyFrameDirectionEvaluator` (**default for direction**); `AttributeEvaluator` (paired slow/fast inside `pair_id`). A's world-frame `TrajectoryEvaluator` in `eval_trajectory.py` stays registered for comparison | A |
| leg / arm / torso direction | `eval_limb.py` | `LimbGeometryEvaluator`, `TorsoGeometryEvaluator` | B |
| target, relation | `eval_spatial.py` | `SpatialRelationEvaluator` | B |
| count, order, simultaneous | `eval_temporal.py` | `CountEvaluator`, `OrderEvaluator`, `SimultaneousEvaluator` | C |
| action | `eval_action.py` | closed-vocabulary rules; other actions → human review | C |
| calibration | `validation.py`, `analysis/analysis_rules.ipynb` | **the one calibration notebook for all evaluators** (incl. Body Side and body-frame Direction): synthetic negatives (L/R swap, time reversal, frozen pose), κ, grid search from saved results, leave-one-model-out, per-type disagreement view | D |
| plan B | `finalize.py`, `run_main.ipynb` | definition lint, auto/human decision (`benchmark/evaluation_mode.json`), review sheet, merge, audit, final tables | whole team |

Evaluator names match `EVALUATION_CONFIG`.

## Closed vocabulary (the Main definition must stay inside it)

`finalize.lint_definition` / `run_main.ipynb` STEP 2 report anything outside it as an error.

| Type | Allowed values |
|---|---|
| action | walk, turn, jump, kick, reach, raise_hand, raise_hands (synonyms: walks, hop, raise, …) |
| direction | forward, backward, left, right |
| turn_direction | left, right |
| body_side | left, right, both |
| leg_direction / arm_direction | side (outward), inward, forward, backward, up, down |
| torso_direction | forward, backward, left, right |
| target | left/right shoulder, elbow, hand, hip, knee, foot; head, neck, chest, pelvis |
| relation | cross_body, above_head, in_front, behind_back, hands_together |
| attribute | slow, fast (as a pair_id pair) |
| count | positive integer, of jump / kick / turn / reach / raise_hand(s) |
| order / simultaneous | need `event_sequence` / `event_group` with ≥ 2 events |

Every non-action requirement should have `applies_to`.

## auto / human gate (`finalize.DEFAULT_POLICY`)

A type is **auto** only if, on the Pilot (all models, real labels + synthetic negatives):
κ ≥ 0.6, n ≥ 10, ≥ 3 human FAIL and ≥ 3 human PASS, raw agreement on real labels ≥ 0.8, and at least one
real label. Otherwise **human**. arm_direction and torso_direction are not in the Pilot and therefore human.

## Provisional thresholds (replaced by calibration)

| Evaluator | Rule | Thresholds |
|---|---|---|
| BodyFrameDirection | displacement during the action in the body frame (facing at the clip start; for a later ordered step, facing when that step starts) ≥ 0.5 m in the requested direction | `min_displacement` 0.5 |
| Action | walk: longest walk ≥ 1 s and ≥ 0.5 m; turn: largest turn ≥ 45°; others: ≥ 1 event | `walk_min_duration_s` 1.0, `walk_min_distance_m` 0.5, `turn_min_deg` 45 |
| Rotation | dominant turn has the requested sign and ≥ 45° | `min_turn_deg` 45 |
| Count | events of the linked action (filtered by its body_side) = N | `tolerance` 0 |
| Order | first event of each listed action; starts increase in order | `min_start_gap_frames` 1 |
| Simultaneous | best overlap ≥ 50 % of the shorter event | `min_overlap_ratio` 0.5 |
| Attribute | fast ≥ 1.2 × partner speed; no partner file → absolute 0.9 / 1.4 m/s | `speed_ratio` 1.2, `min_walk_speed` 0.3 |
| SpatialRelation | target ≤ 0.75 × shoulder width; cross_body ≥ 0.25 × shoulder width past the midline | `target_max_dist_sw`, `cross_min_sw` |
| LimbGeometry | at the action peak: component ≥ 0.35 × limb length and ≥ 1.2 × other horizontal ones | `min_extent`, `min_dominance` |
| TorsoGeometry | tilt ≥ 15°, 1.5 × the other axis | `min_tilt_deg`, `min_dominance` |

Event detection (`events.EVENT_PARAMS`, shared): walk ≥ 0.3 m/s for ≥ 1 s with alternating feet; jump =
both feet off ≥ 2 frames; raise = wrist above shoulder ≥ 0.25 s; 5-frame smoothing. Change only via PR.

## Outputs

| Stage | File | Where it goes |
|---|---|---|
| Pilot | `<Model>_pilot_results.json/.csv` (run_benchmark STEP 11) | shared Drive folder |
| Pilot | `labels/<Model>_pilot_human_gold_labels.json` | repository (PR) |
| Calibration | `benchmark/evaluation_mode.json` + frozen `CURRENT_THRESHOLDS` | repository (one PR), tag `v1.0-pilot-frozen` |
| Main | `raw/<Model>_main_results.json/.csv` | inside `main_outputs.zip` |
| Main | `review_sheet.csv` → filled by people → `review_sheet_filled.csv` | shared sheet |
| Main | `main_state.json` (to resume after a Colab reset) | inside `main_outputs.zip` |
| Main | `final/final_main_results.json/.csv` (final_source, final_label per requirement) | inside `main_outputs.zip` |
| Main | `tables/overall, by_capability, by_difficulty, by_type, audit, evaluation_mode .csv` + `summary.md` | paper |

## Verified (synthetic motions only)

* `pytest`: all 61 Pilot requirements evaluated; correct motions PASS (all 61); 15 controlled
  negatives FAIL on the intended requirement; same decisions after rotating the scene by 37° and with 2 cm
  jitter; all synthetic negatives caught; finalize steps (lint, gate, review sheet, merge, tables).
* Dry run of the whole chain with three synthetic models: analysis_rules.ipynb → evaluation_mode.json →
  run_main.ipynb → review sheet → merge → tables.
* A's Direction analysis notebook still reproduces Df5 exactly (it uses `TrajectoryEvaluator` directly).

Not verified yet: real model outputs and real Human Gold agreement — that is the calibration step.

## Direction: body frame (decided 2026-09-26)

Direction is judged relative to the person, not the world:

* "walks to the left" = left of the facing at the start of the clip (side-stepping, or turning left and
  walking, both count);
* "turns right, walks forward" (C5-16) = forward after the turn (facing when the walk starts).

A's world-frame `TrajectoryEvaluator` called C5-16's correct walk "right" (world +X) and failed it; it is
kept unchanged and can still be used for comparison:

```python
rows = run_all_evaluators(evaluation_cases, HUMAN_GOLD_LABELS,
                          config={**EVALUATION_CONFIG, "direction": "TrajectoryEvaluator"})
```

The body-frame rule is calibrated in `analysis/analysis_rules.ipynb` (`BodyFrameDirectionEvaluator`,
`min_displacement`). Reviewers use the same convention (runbook §7).
