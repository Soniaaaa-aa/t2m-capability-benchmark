"""
validation.py — calibration and validation tools (plan §6)
==========================================================

Owner: D (plan §8)

  make_synthetic_negatives  left/right swap, time reversal and a frozen first pose
                            of motions whose requirement was labelled PASS ->
                            guaranteed FAIL cases (calibration / tests only,
                            never in model scores)
  agreement_report          per requirement type: n, raw agreement, Cohen's κ,
                            "insufficient evidence" when n < min_n (plan: 10),
                            meets_gate when κ >= 0.6
  grid_search               re-fit an evaluator's thresholds from SAVED
                            results rows (evidence is stored), no motion re-run
  leave_one_model_out       fit on 3 models, test on the 4th, rotate

Rows are the output of common.run_all_evaluators (or the JSON saved by
common.save_results: payload["results"]).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

import common

LR_PAIRS = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21)]


# ============================================================================
# Synthetic negatives
# ============================================================================
def swap_left_right(motion):
    """Mirror X and swap left/right joint labels: a left kick becomes a right kick."""
    m = np.array(motion, dtype=np.float64, copy=True)
    m[..., 0] *= -1
    for a, b in LR_PAIRS:
        m[:, [a, b]] = m[:, [b, a]]
    return m


def time_reverse(motion):
    m = np.array(motion, dtype=np.float64, copy=True)[::-1]
    m[..., 0] -= m[0, 0, 0]
    m[..., 2] -= m[0, 0, 2]
    return np.ascontiguousarray(m)


def freeze(motion):
    """Hold the first pose for the whole clip: no action happens, so every requirement fails."""
    m = np.array(motion, dtype=np.float64, copy=True)
    return np.repeat(m[:1], len(m), axis=0)


def _flips_under_swap(req):
    t, v = req.get("type"), req.get("value")
    if t in ("body_side", "turn_direction", "direction", "torso_direction"):
        return v in ("left", "right")
    if t == "target":
        return isinstance(v, str) and (v.startswith("left_") or v.startswith("right_"))
    return False


def _flips_under_reverse(req):
    t, v = req.get("type"), req.get("value")
    if t == "order":
        return len(req.get("event_sequence") or v or []) == 2
    if t == "direction":
        return v in ("forward", "backward")
    if t == "turn_direction":
        return v in ("left", "right")
    return False


NEGATIVE_TRANSFORMS = {
    "swap_lr": (swap_left_right, _flips_under_swap),
    "reverse": (time_reverse, _flips_under_reverse),
    # "easy" negative: gives every type FAIL examples (action, count, ... otherwise have
    # almost none in the Pilot). Report agreement on real labels separately.
    "freeze": (freeze, lambda req: True),
}


def make_synthetic_negatives(evaluation_cases, human_gold_labels, out_dir, transforms=None):
    """
    For every case and transform, requirements that were human-labelled PASS
    and flip under the transform become FAIL; all others are UNCERTAIN
    (ignored in agreement). Returns (negative_cases, negative_labels) usable
    directly with common.run_all_evaluators.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    neg_cases, neg_labels = [], {}
    for case in evaluation_cases:
        gold = human_gold_labels.get(case["prompt_id"])
        if gold is None:
            continue
        motion = common.load_motion(case["motion_path"])
        for name, (transform, flips) in NEGATIVE_TRANSFORMS.items():
            if transforms is not None and name not in transforms:
                continue
            labels = []
            for i, req in enumerate(case["requirements"]):
                human = gold["requirements"][i].get("human_label")
                lab = "FAIL" if (human == "PASS" and flips(req)) else "UNCERTAIN"
                labels.append({"requirement_index": i, "type": req.get("type"),
                               "value": req.get("value"), "human_label": lab})
            if not any(x["human_label"] == "FAIL" for x in labels):
                continue
            pid = f"{case['prompt_id']}__{name}"
            path = out_dir / f"{case.get('model', 'model')}_{pid}.npy"
            np.save(path, transform(motion).astype(np.float32))
            neg_cases.append({**case, "prompt_id": pid, "motion_path": str(path), "synthetic": name})
            neg_labels[pid] = {"prompt": case.get("prompt"), "requirements": labels}
    return neg_cases, neg_labels


# ============================================================================
# Agreement
# ============================================================================
def cohen_kappa(pred, gold):
    pred, gold = list(pred), list(gold)
    n = len(pred)
    if n == 0:
        return None
    po = sum(p == g for p, g in zip(pred, gold)) / n
    labels = set(pred) | set(gold)
    pe = sum((pred.count(c) / n) * (gold.count(c) / n) for c in labels)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def _labelled(rows):
    return [r for r in rows if r.get("pass_fail") in ("PASS", "FAIL")
            and r.get("human_label") in ("PASS", "FAIL")]


def agreement_report(rows, min_n=10, kappa_gate=0.6, verbose=True):
    """Per requirement type: n, agreement, κ, gate — the table for plan §6 step 4."""
    report = {}
    for t in sorted({r["requirement_type"] for r in rows}):
        lab = _labelled([r for r in rows if r["requirement_type"] == t])
        pred = [r["pass_fail"] for r in lab]
        gold = [r["human_label"] for r in lab]
        n = len(lab)
        kappa = cohen_kappa(pred, gold) if n else None
        report[t] = {
            "n": n,
            "agreement": (sum(p == g for p, g in zip(pred, gold)) / n) if n else None,
            "kappa": kappa,
            "n_human_fail": gold.count("FAIL"),
            "insufficient_evidence": n < min_n,
            "meets_gate": (kappa is not None and kappa >= kappa_gate and n >= min_n),
        }
    if verbose:
        print(f"{'type':16s} {'n':>4s} {'agree':>7s} {'kappa':>7s} {'#FAIL':>6s}  status")
        for t, s in report.items():
            ag = "-" if s["agreement"] is None else f"{s['agreement']:.2f}"
            ka = "-" if s["kappa"] is None else f"{s['kappa']:.2f}"
            status = "insufficient" if s["insufficient_evidence"] else ("OK" if s["meets_gate"] else "below gate")
            print(f"{t:16s} {s['n']:4d} {ag:>7s} {ka:>7s} {s['n_human_fail']:6d}  {status}")
    return report


# ============================================================================
# Threshold fitting from saved rows
# ============================================================================
def _score(evaluator_cls, thresholds, rows):
    inst = evaluator_cls(thresholds=thresholds)
    pred, gold = [], []
    for r in rows:
        ev = r.get("evidence") or {}
        if ev.get("unsupported"):
            continue
        p, _ = inst.decide_from_evidence(ev, r["expected_value"])
        if p in ("PASS", "FAIL"):
            pred.append(p)
            gold.append(r["human_label"])
    n = len(pred)
    acc = sum(p == g for p, g in zip(pred, gold)) / n if n else 0.0
    return cohen_kappa(pred, gold) if n else None, acc, n


def grid_search(evaluator_cls, rows, grid, requirement_types=None):
    """
    grid: {"threshold_name": [values, ...]}; unspecified thresholds keep their
    PROVISIONAL value. Ranks by κ, then agreement. Returns (best_thresholds, table).
    """
    name = evaluator_cls.EVALUATOR_NAME
    sel = [r for r in _labelled(rows) if r.get("evaluator") == name
           and (requirement_types is None or r["requirement_type"] in requirement_types)]
    keys = list(grid)
    table = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        th = {**evaluator_cls.PROVISIONAL_THRESHOLDS, **dict(zip(keys, combo))}
        kappa, acc, n = _score(evaluator_cls, th, sel)
        table.append({"thresholds": th, "kappa": kappa, "agreement": acc, "n": n})
    # Rank by κ, then agreement; among exact ties prefer the values closest to the
    # plan's provisional thresholds (avoids drifting to the most lenient grid corner
    # when few labels cannot tell the candidates apart).
    prov = evaluator_cls.PROVISIONAL_THRESHOLDS

    def distance(th):
        return sum(abs(th[k] - prov[k]) / (abs(prov[k]) or 1.0) for k in keys if k in prov)

    table.sort(key=lambda x: (-(x["kappa"] if x["kappa"] is not None else -2), -x["agreement"],
                              distance(x["thresholds"])))
    return (table[0]["thresholds"] if table else None), table


def leave_one_model_out(evaluator_cls, rows, grid, requirement_types=None):
    """Fit on all models but one, test on the held-out one (plan §6 step 3)."""
    models = sorted({r.get("model") for r in rows if r.get("model")})
    out = {}
    for held in models:
        train = [r for r in rows if r.get("model") != held]
        test = [r for r in _labelled(rows) if r.get("model") == held
                and r.get("evaluator") == evaluator_cls.EVALUATOR_NAME
                and (requirement_types is None or r["requirement_type"] in requirement_types)]
        best, _ = grid_search(evaluator_cls, train, grid, requirement_types)
        kappa, acc, n = _score(evaluator_cls, best, test) if best else (None, 0.0, 0)
        out[held] = {"fitted_on": [m for m in models if m != held], "thresholds": best,
                     "test_kappa": kappa, "test_agreement": acc, "test_n": n}
    return out
