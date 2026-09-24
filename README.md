# T2M Diagnostic Benchmark

Capability-oriented benchmark for Text-to-Motion models. The project has two independent halves,
connected only by a fixed file format:

```
generation/<model>/   ──►  <Model>_standardized_<split>.zip  ──►  evaluation/
(one folder per model)      (Standardised Motion contract)        (shared evaluators, same rules for every model)
```

```
t2m-benchmark/
├── generation/                    # GENERATION — one folder per T2M model
│   ├── motionhiflow/              #   runner + Colab export notebook
│   └── <model>/ ...
├── evaluation/                    # EVALUATION — shared by all models
│   ├── common.py                  #   settings, input contract, loading, BaseEvaluator/registry,
│   │                              #   Human Gold helpers, automatic import + unified runner
│   ├── eval_trajectory.py         #   direction  → TrajectoryEvaluator   (member A)
│   ├── eval_body_side.py          #   body_side  → BodySideEvaluator     (member B)
│   ├── eval_template.py           #   copy this to start a new evaluator (members C, D)
│   ├── run_benchmark.ipynb        #   shared runner: load data, run ALL evaluators, save results
│   └── analysis/                  #   one calibration notebook per evaluator, owned by its author
│       ├── analysis_direction.ipynb
│       └── analysis_body_side.ipynb
├── benchmark/                     # benchmark definition JSON — single source of truth for both halves
├── labels/                        # Human Gold Labels, one JSON per model
├── tests/                         # pytest (synthetic motions in tests/synth_motion.py)
└── .github/workflows/tests.yml    # runs the tests on every Pull Request
```

Evaluators are **one file per evaluator, not per model**: every model is scored with exactly the
same code and thresholds. What differs per model is only its generated motions (Google Drive) and its
Human Gold Labels (`labels/`).

## Standardised Motion contract (the interface between the two halves)

Every model package is a ZIP containing `<ModelName>/<prompt_id>.npy`:

| | |
|---|---|
| shape | `[T, 22, 3]` (HumanML3D joint order) |
| frames | `T == target_frames_20fps` of the prompt, 20 fps |
| units | metres |
| axes | `+X` = Right, `+Y` = Up, `+Z` = Forward |
| origin | ground `Y = 0`, first-frame root `XZ = (0, 0)` |

Generation code never imports evaluation code and vice versa.

## Using it in Colab

Open `evaluation/run_benchmark.ipynb` in Colab (File → Open notebook → GitHub; a CPU runtime is
enough). STEP 0 clones or pulls this repository and imports `common.py` plus **every
`evaluation/eval_*.py` automatically**. Set `MODEL_NAME` in STEP 4 and run top to bottom: STEP 10
runs all implemented evaluators with their current rules, STEP 11 saves JSON + CSV results.

Each evaluator's calibration (evidence vs Human Gold, threshold search, validation) lives in its own
notebook under `evaluation/analysis/`.

Private repository: create a GitHub fine-grained token (read access to this repo) and add it as a
Colab secret named `GITHUB_TOKEN`.

## Working rules

1. **Only edit the files you own**: your `evaluation/eval_<name>.py`, your
   `evaluation/analysis/analysis_<name>.ipynb`, your `generation/<model>/` folder, your tests.
2. **Shared files** — `evaluation/common.py`, `evaluation/run_benchmark.ipynb`, `benchmark/` — change
   only through a Pull Request reviewed by the whole team; every evaluator depends on them.
   Adding an evaluator does **not** require changing them.
3. **Never push to `main` directly.** Branch → Pull Request → one review → merge.
4. **Logic in `.py`, not in the notebook.** Notebook cells only call functions and print.
5. **No large or generated files in Git**: `.npy`, ZIPs, checkpoints, GIFs/videos (see `.gitignore`).
   Share motion packages via Google Drive.
6. **Freeze before the Main split.** When the Pilot rules are final, tag the commit
   (e.g. `v1.0-pilot-frozen`) and run the Main split with exactly that tag.

### Day-to-day workflow

```bash
git checkout main && git pull                     # start from the latest main
git checkout -b feat/rotation-evaluator           # your own branch
# ... edit evaluation/eval_rotation.py and tests/test_rotation.py ...
python -m pytest -q                               # optional locally; CI runs it on every PR
git add evaluation/eval_rotation.py tests/test_rotation.py
git commit -m "Add RotationEvaluator evidence"
git push -u origin feat/rotation-evaluator        # then open a Pull Request on GitHub
```

Without git on your computer: on GitHub, open the file (or folder), edit in the browser or use
*Add file → Upload files*, and choose *Create a new branch and start a pull request*.

To test your branch in Colab before it is merged, set `BRANCH = "feat/..."` in STEP 0.

### Adding an evaluator (no notebook edit needed)

1. Copy `evaluation/eval_template.py` to `evaluation/eval_<name>.py`, rename the class,
   implement `calculate_evidence` / `decide`.
2. Uncomment `register_evaluator("<Name>Evaluator", <Name>Evaluator)` at the bottom. The name must
   match `EVALUATION_CONFIG` in `evaluation/common.py`.
3. Put the rule you want the benchmark to use in `CURRENT_THRESHOLDS` / `CURRENT_THRESHOLD_STATUS`
   (`for_benchmark()` returns the evaluator configured with them).
4. Optional: `evaluation/analysis/analysis_<name>.ipynb` for calibration (copy
   `analysis_body_side.ipynb`) and `tests/test_<name>.py`.
5. Open a Pull Request. Once merged, STEP 0 imports the file and STEP 10 runs it automatically.

### Changing a threshold

Calibrate in your analysis notebook, then update `CURRENT_*` in your `eval_<name>.py` through a
Pull Request. The results file saved in STEP 11 records the rules and the framework commit used.

Developing in a Colab notebook is fine — when it works, copy only the class (and helper functions)
into your `eval_<name>.py`. Do not use Colab's "Download .py" output as-is: it contains every cell,
including uploads and prints, which run on import.

### Adding a model

Create `generation/<model>/` with the code that produces the ZIP above, reading prompts from
`benchmark/`. See `generation/README.md`.

### Human Gold Labels

Created in STEP 9D/9E of the notebook and stored as `labels/<Model>_pilot_human_gold_labels.json`.
Add the file via a Pull Request so that every member calibrates against the same labels. Labels are
checked against the benchmark definition when loaded; labels made for an older definition raise an
error instead of being used silently.

## Migration notes (from `Benchmark_Df5_English.ipynb`)

- STEP 2–9F moved to `evaluation/common.py`; `TrajectoryEvaluator`,
  `calculate_required_direction_ratio` and `DirectionDecisionRule` moved **unchanged** to
  `evaluation/eval_trajectory.py`. The Direction workflow cells (10A–10F) are unchanged apart from
  `get_human_label(HUMAN_GOLD_LABELS, ...)` taking the labels explicitly.
- Verified: on the same synthetic Pilot package and labels, the new notebook reproduces Df5's
  registered motions, evaluation cases, Human Gold template/labels, direction evidence, threshold
  search and pilot validation exactly.
- Fixed: Df5 STEP 6/7 used `SELECTED_MODEL` / `MOTION_INPUT_DIR`, which STEP 4 never defined
  (NameError on a fresh runtime). Now `MODEL_NAME` / `MODEL_INPUT_DIR` throughout.
- `SUPPORTED_REQUIREMENT_TYPES` is now derived from `EVALUATION_CONFIG` (the old list disagreed with it).
- Df5 STEP 10A–10F now live in `evaluation/analysis/analysis_direction.ipynb` (unchanged).
- `TrajectoryEvaluator` does not follow the common interface itself, so a small wrapper
  (`TrajectoryRequirementEvaluator`, appended to `eval_trajectory.py`; original classes untouched)
  registers it for the unified runner with the STEP 10D rule (0.50 m, not frozen). Verified: the
  runner's Direction PASS/FAIL equals Df5 STEP 10E on the synthetic Pilot. Owner may replace the
  wrapper once `TrajectoryEvaluator` inherits `BaseEvaluator`.
