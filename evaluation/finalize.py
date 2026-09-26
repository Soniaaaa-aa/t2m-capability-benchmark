"""
finalize.py — plan B: from rule results to the final benchmark scores
======================================================================

Owner: whole team (change via reviewed PR)

Plan B = keep the rule evaluators, calibrate once on the Pilot, and send every
requirement type the rules cannot judge reliably to humans instead of
improving the rules. This module holds the four steps around that decision:

  1. lint_definition      check a benchmark definition before generating / evaluating:
                          closed action vocabulary, known values, applies_to /
                          event_sequence / event_group present, slow/fast pairs complete
  2. decide_modes         per requirement type: "auto" (rules) or "human",
                          from Pilot agreement with Human Gold (+ synthetic negatives);
                          saved to benchmark/evaluation_mode.json and frozen with the thresholds
  3. apply_mode           Main results -> final_source auto / human_required
     make_review_sheet    one CSV for people: every human_required requirement +
                          a blind stratified audit sample of auto requirements (10 %, >= 20 per type)
     merge_reviews        fill the returned CSV back in
  4. audit_report         agreement of the rules with humans on the audit sample (report in the paper)
     score_tables         final tables: per model × requirement type / capability / difficulty

Nothing here changes an evaluator's decision; it only decides whose label is final.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import common
import events as E
import validation as V

PASS_FAIL = ("PASS", "FAIL")
LABEL_CODES = {"P": "PASS", "PASS": "PASS", "F": "FAIL", "FAIL": "FAIL", "U": "UNCERTAIN", "UNCERTAIN": "UNCERTAIN"}
MODE_FILENAME = "evaluation_mode.json"
COUNTABLE_EVENTS = {"jump", "kick", "turn", "reach", "raise_hand", "raise_hands"}

DEFAULT_POLICY = {
    "kappa_gate": 0.6,            # plan §6: κ >= 0.6 per type
    "min_n": 10,                  # labelled requirements (real + synthetic)
    "min_human_fail": 3,          # κ is meaningless without FAIL examples
    "min_human_pass": 3,
    "min_real_agreement": 0.8,    # raw agreement on real Pilot labels only
    "min_real_n": 1,              # types never seen in the Pilot cannot be auto
}


# ============================================================================
# 1. Definition lint
# ============================================================================
def _vocab():
    import eval_limb
    import eval_spatial
    import eval_trajectory_ext
    return {
        "direction": {"forward", "backward", "left", "right"},
        "turn_direction": {"left", "right", "clockwise", "counterclockwise"},
        "body_side": {"left", "right", "both"},
        "leg_direction": set(eval_limb.DIRECTION_ALIASES),
        "arm_direction": set(eval_limb.DIRECTION_ALIASES),
        "torso_direction": set(eval_limb.TorsoGeometryEvaluator.ALIASES),
        "target": set(eval_spatial.TARGET_JOINTS),
        "relation": {"cross_body", "above_head", "in_front", "in_front_of_body", "front", "behind_back",
                     "behind", "behind_body", "hands_together", "together"},
        "attribute": set(eval_trajectory_ext.SPEED_WORDS),
    }


def lint_definition(prompts):
    """
    Returns a list of problems {"prompt_id", "requirement_id", "level", "message"}.
    level "error"   -> the rules cannot judge it (fix the definition or accept human review)
    level "warning" -> judged, but with a fallback (e.g. no applies_to, no slow/fast partner)
    """
    vocab = _vocab()
    out = []

    def add(p, r, level, msg):
        out.append({"prompt_id": p.get("prompt_id"), "requirement_id": (r or {}).get("id"),
                    "level": level, "message": msg})

    seen = set()
    pairs = defaultdict(list)
    for p in prompts:
        pid = p.get("prompt_id")
        if pid in seen:
            add(p, None, "error", "duplicate prompt_id")
        seen.add(pid)
        reqs = p.get("requirements") or []
        if not reqs:
            add(p, None, "error", "no requirements")
        ids = [r.get("id") for r in reqs]
        if None in ids:
            add(p, None, "warning", "requirement without id (labels then rely on the order)")
        if len([i for i in ids if i]) != len({i for i in ids if i}):
            add(p, None, "error", "duplicate requirement id")
        by_id = {r.get("id"): r for r in reqs if r.get("id")}
        case = {"prompt_id": pid, "prompt": p.get("text", ""), "requirements": reqs}
        if p.get("pair_id"):
            pairs[p["pair_id"]].append(p)

        for r in reqs:
            t, v = r.get("type"), r.get("value")
            if t not in common.EVALUATION_CONFIG:
                add(p, r, "error", f"unknown requirement type {t!r}")
                continue
            if t == "action":
                if E.event_kind(v) is None:
                    add(p, r, "error", f"action {v!r} outside the rule vocabulary "
                                       f"{sorted(set(E.ACTION_TO_EVENT.values()))}")
                continue
            if t in ("order", "simultaneous"):
                key = "event_sequence" if t == "order" else "event_group"
                entries = r.get(key)
                if not entries:
                    add(p, r, "warning", f"{t} without {key}; falls back to value + applies_to")
                    entries = E.entries_for(r, case, key)
                if len(entries) < 2:
                    add(p, r, "error", f"{t} needs at least 2 events")
                for entry in entries:
                    acts = [by_id.get(i) for i in entry.get("requirement_ids") or []]
                    acts = [a for a in acts if a and a.get("type") == "action"]
                    kind = E.event_kind(acts[0]["value"]) if acts else E.event_kind(entry.get("event"))
                    if kind is None:
                        add(p, r, "error", f"{t} event {entry.get('event')!r} has no detector")
                continue
            if not r.get("applies_to"):
                add(p, r, "warning", "no applies_to; the nearest preceding action is used")
            else:
                missing = [i for i in r["applies_to"] if i not in by_id]
                if missing:
                    add(p, r, "error", f"applies_to refers to unknown ids {missing}")
            if t == "count":
                if not isinstance(v, int) or v < 1:
                    add(p, r, "error", f"count value must be a positive integer, got {v!r}")
                act = E.linked_action(r, case)
                kind = E.event_kind(act.get("value")) if act else None
                if kind not in COUNTABLE_EVENTS:
                    add(p, r, "error", f"count of {kind!r} is not countable by rules")
                continue
            allowed = vocab.get(t)
            if allowed is not None and str(v).lower() not in allowed:
                add(p, r, "error", f"{t} value {v!r} not supported; allowed: {sorted(allowed)}")

    for pair_id, members in pairs.items():
        speeds = [str(r.get("value")).lower() for m in members for r in m.get("requirements", [])
                  if r.get("type") == "attribute"]
        if speeds and not ({"slow", "slowly"} & set(speeds) and {"fast", "quickly", "quick"} & set(speeds)):
            add(members[0], None, "warning", f"pair {pair_id} has no slow/fast counterpart; "
                                             f"attribute uses the absolute fallback")
    return out


def print_lint(problems):
    errors = [p for p in problems if p["level"] == "error"]
    warnings = [p for p in problems if p["level"] == "warning"]
    for p in errors + warnings:
        print(f"{p['level'].upper():8s} {p['prompt_id']:8s} {str(p['requirement_id'] or ''):5s} {p['message']}")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    return not errors


# ============================================================================
# 2. auto / human per requirement type
# ============================================================================
def _stats(rows):
    lab = [r for r in rows if r.get("pass_fail") in PASS_FAIL and r.get("human_label") in PASS_FAIL]
    pred = [r["pass_fail"] for r in lab]
    gold = [r["human_label"] for r in lab]
    n = len(lab)
    return {"n": n,
            "agreement": (sum(a == b for a, b in zip(pred, gold)) / n) if n else None,
            "kappa": V.cohen_kappa(pred, gold) if n else None,
            "n_human_fail": gold.count("FAIL"), "n_human_pass": gold.count("PASS")}


def decide_modes(real_rows, synthetic_rows=(), policy=None, types=None):
    """
    real_rows      : Pilot results with Human Gold (all models)
    synthetic_rows : results on validation.make_synthetic_negatives (optional, recommended)
    Returns {"policy", "types": {type: {"mode": "auto"|"human", "reason", "real", "synthetic", "combined"}}}
    """
    pol = {**DEFAULT_POLICY, **(policy or {})}
    types = types or sorted(common.EVALUATION_CONFIG)
    out = {}
    for t in types:
        real = _stats([r for r in real_rows if r.get("requirement_type") == t])
        syn = _stats([r for r in synthetic_rows if r.get("requirement_type") == t])
        comb = _stats([r for r in list(real_rows) + list(synthetic_rows) if r.get("requirement_type") == t])
        checks = [
            (real["n"] >= pol["min_real_n"], f"only {real['n']} real Pilot labels"),
            (comb["n"] >= pol["min_n"], f"n={comb['n']} < {pol['min_n']}"),
            (comb["n_human_fail"] >= pol["min_human_fail"], f"only {comb['n_human_fail']} FAIL labels"),
            (comb["n_human_pass"] >= pol["min_human_pass"], f"only {comb['n_human_pass']} PASS labels"),
            (comb["kappa"] is not None and comb["kappa"] >= pol["kappa_gate"],
             f"kappa={comb['kappa'] if comb['kappa'] is None else round(comb['kappa'], 2)} < {pol['kappa_gate']}"),
            (real["agreement"] is not None and real["agreement"] >= pol["min_real_agreement"],
             f"real agreement={real['agreement'] if real['agreement'] is None else round(real['agreement'], 2)}"
             f" < {pol['min_real_agreement']}"),
        ]
        failed = [msg for ok, msg in checks if not ok]
        out[t] = {"mode": "human" if failed else "auto",
                  "reason": "; ".join(failed) if failed else "meets all gates",
                  "real": real, "synthetic": syn, "combined": comb}
    return {"policy": pol, "types": out}


def print_modes(mode):
    print(f"{'type':16s} {'mode':6s} {'n':>4s} {'kappa':>6s} {'real agr':>8s}  reason")
    for t, m in mode["types"].items():
        c, r = m["combined"], m["real"]
        ka = "-" if c["kappa"] is None else f"{c['kappa']:.2f}"
        ag = "-" if r["agreement"] is None else f"{r['agreement']:.2f}"
        print(f"{t:16s} {m['mode']:6s} {c['n']:4d} {ka:>6s} {ag:>8s}  {m['reason']}")


def current_thresholds():
    """Thresholds each registered evaluator runs with now (stored next to the mode decision)."""
    out = {}
    for name, cls in sorted(common.EVALUATOR_REGISTRY.items()):
        th = getattr(cls, "CURRENT_THRESHOLDS", None)
        if th is None and hasattr(cls, "CURRENT_MIN_DISPLACEMENT"):
            th = {"min_displacement": cls.CURRENT_MIN_DISPLACEMENT}
        out[name] = {"thresholds": dict(th) if th else th, "status": getattr(cls, "CURRENT_THRESHOLD_STATUS", None)}
    return out


def redecide(rows, thresholds_by_evaluator):
    """
    Re-apply decide() with new thresholds to saved rows (evidence is stored, no motion
    re-run). Evaluators with decide_from_evidence (all rule evaluators + BodySide) are
    re-decided; other rows are copied unchanged.
    """
    inst = {}
    out = []
    for r in rows:
        r = dict(r)
        name = r.get("evaluator")
        th = thresholds_by_evaluator.get(name)
        ev = r.get("evidence") or {}
        cls = common.EVALUATOR_REGISTRY.get(name)
        if th is not None and cls is not None and hasattr(cls, "decide_from_evidence") \
                and r.get("status") == "EVALUATED" and not ev.get("unsupported"):
            if name not in inst:
                inst[name] = cls(thresholds=th)
            r["pass_fail"], r["reason"] = inst[name].decide_from_evidence(ev, r["expected_value"])
            if r.get("human_label") in PASS_FAIL:
                r["match"] = r["pass_fail"] == r["human_label"]
        out.append(r)
    return out


def check_frozen(mode):
    """Differences between the thresholds stored with the mode decision and those in the code."""
    stored = mode.get("evaluator_thresholds") or {}
    now = current_thresholds()
    problems = []
    for name, entry in stored.items():
        if name in now and entry.get("thresholds") != now[name]["thresholds"]:
            problems.append(f"{name}: evaluation_mode.json has {entry.get('thresholds')}, "
                            f"code has {now[name]['thresholds']}")
    return problems


def save_mode(mode, path, metadata=None, frozen_thresholds=None):
    """
    frozen_thresholds: {evaluator: thresholds} chosen in calibration; they must be copied
    into the evaluators' CURRENT_THRESHOLDS (check_frozen() warns in run_main.ipynb otherwise).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ths = current_thresholds()
    for name, th in (frozen_thresholds or {}).items():
        ths.setdefault(name, {})["thresholds"] = dict(th)
        ths[name]["status"] = "frozen_v1.0"
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               **(metadata or {}), "policy": mode["policy"], "types": mode["types"],
               "evaluator_thresholds": ths}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_mode(path):
    """Returns {"types": {type: {"mode": ...}}, ...}; a missing file means every type is human."""
    path = Path(path)
    if not path.exists():
        return {"types": {}, "missing_file": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================================
# 3. Main: apply the mode, review sheet, merge
# ============================================================================
def review_id(row):
    return f"{row['model']}|{row['prompt_id']}|{row['requirement_index']}"


def apply_mode(rows, mode):
    """
    Copy of rows with:
      final_source  "auto"            rule result is final
                    "human_required"  type is human, or the rule could not decide
      final_label   PASS / FAIL for auto rows, None until reviewed for human rows
    """
    types = mode.get("types", {})
    out = []
    for r in rows:
        r = dict(r)
        auto = types.get(r["requirement_type"], {}).get("mode") == "auto"
        if auto and r.get("status") == "EVALUATED" and r.get("pass_fail") in PASS_FAIL:
            r["final_source"], r["final_label"] = "auto", r["pass_fail"]
        else:
            r["final_source"], r["final_label"] = "human_required", None
        r["review_label"] = None
        r["audit"] = False
        out.append(r)
    return out


def make_review_sheet(rows, prompts, audit_fraction=0.1, min_audit_per_type=20, seed=0,
                      gif_pattern="{model}_gifs/{prompt_id}.gif"):
    """
    Items for people: every human_required row + a blind audit sample of auto rows
    (per type: max(10 %, 20) spread evenly over models). The sheet never shows the
    automatic result. Marks the sampled rows (row["audit"] = True) in place.
    """
    rng = random.Random(seed)
    meta = {p["prompt_id"]: p for p in prompts}
    items = []

    def item(r, why):
        p = meta.get(r["prompt_id"], {})
        req = (p.get("requirements") or [{}] * (r["requirement_index"] + 1))[r["requirement_index"]]
        return {"review_id": review_id(r), "model": r["model"], "prompt_id": r["prompt_id"],
                "requirement_index": r["requirement_index"], "requirement_id": req.get("id"),
                "type": r["requirement_type"], "value": json.dumps(r["expected_value"], ensure_ascii=False)
                if not isinstance(r["expected_value"], str) else r["expected_value"],
                "prompt": p.get("text", ""), "gif": gif_pattern.format(model=r["model"], prompt_id=r["prompt_id"]),
                "why": why, "human_label": "", "note": ""}

    for r in rows:
        if r["final_source"] == "human_required":
            items.append(item(r, "human_required"))

    by_type = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["final_source"] == "auto":
            by_type[r["requirement_type"]][r["model"]].append(r)
    for t, per_model in sorted(by_type.items()):
        n = sum(len(v) for v in per_model.values())
        k = min(n, max(math.ceil(audit_fraction * n), min_audit_per_type))
        pools = {m: rng.sample(v, len(v)) for m, v in sorted(per_model.items())}
        picked = []
        while len(picked) < k:
            for m in list(pools):
                if pools[m] and len(picked) < k:
                    picked.append(pools[m].pop())
        for r in picked:
            r["audit"] = True
            items.append(item(r, "audit"))

    rng.shuffle(items)
    return items


def save_review_sheet(items, path):
    """CSV (UTF-8 with BOM so Excel shows Chinese correctly). Fill human_label with P / F / U."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["review_id", "model", "prompt_id", "requirement_index", "requirement_id", "type", "value",
              "prompt", "gif", "why", "human_label", "note"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(items)
    return path


def load_review_sheet(path):
    """Returns ({review_id: PASS/FAIL/UNCERTAIN}, problems). Blank labels are left out."""
    labels, problems = {}, []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            raw = (row.get("human_label") or "").strip().upper()
            if not raw:
                continue
            if raw not in LABEL_CODES:
                problems.append(f"line {i}: invalid label {raw!r} (use P / F / U)")
                continue
            labels[row["review_id"]] = LABEL_CODES[raw]
    return labels, problems


def merge_reviews(rows, reviews):
    """Fill human labels in. Auto rows keep the rule label; audited ones record agreement."""
    pending = 0
    for r in rows:
        h = reviews.get(review_id(r))
        r["review_label"] = h
        if r["final_source"] == "human_required":
            r["final_label"] = h if h in PASS_FAIL else None
            pending += h is None
        elif r.get("audit") and h in PASS_FAIL:
            r["audit_match"] = r["pass_fail"] == h
    return {"pending_human": pending}


# ============================================================================
# 4. Reports
# ============================================================================
def audit_report(rows, verbose=True):
    """Rule vs human on the audit sample of auto types (plan §6.6)."""
    rep = {}
    for t in sorted({r["requirement_type"] for r in rows if r.get("audit")}):
        sel = [dict(r, human_label=r["review_label"]) for r in rows
               if r.get("audit") and r["requirement_type"] == t]
        rep[t] = _stats(sel)
        rep[t]["sampled"] = len(sel)
    if verbose:
        print(f"{'type':16s} {'sampled':>7s} {'n':>4s} {'agree':>6s} {'kappa':>6s}")
        for t, s in rep.items():
            ag = "-" if s["agreement"] is None else f"{s['agreement']:.2f}"
            ka = "-" if s["kappa"] is None else f"{s['kappa']:.2f}"
            print(f"{t:16s} {s['sampled']:7d} {s['n']:4d} {ag:>6s} {ka:>6s}")
    return rep


def _rate(labels):
    decided = [x for x in labels if x in PASS_FAIL]
    return (sum(x == "PASS" for x in decided) / len(decided)) if decided else None


def score_tables(rows, prompts):
    """
    by_type        model × requirement type   requirement pass rate
    by_capability  model × capability         requirement pass rate + prompt success rate
    by_difficulty  model × difficulty         same
    overall        model                      same + how many labels are still pending
    Prompt success = every requirement of the prompt is PASS (unknown if any is pending).
    """
    meta = {p["prompt_id"]: p for p in prompts}
    per_prompt = defaultdict(list)
    for r in rows:
        per_prompt[(r["model"], r["prompt_id"])].append(r)

    def prompt_ok(rs):
        labs = [r["final_label"] for r in rs]
        if any(x not in PASS_FAIL for x in labs):
            return None
        return all(x == "PASS" for x in labs)

    def table(key_fn, key_name):
        groups = defaultdict(lambda: {"reqs": [], "prompts": []})
        for (model, pid), rs in per_prompt.items():
            for key in key_fn(pid, rs):
                g = groups[(model, key)]
                g["prompts"].append(prompt_ok(rs))
                g["reqs"] += [r["final_label"] for r in rs if key_name != "requirement_type"
                              or r["requirement_type"] == key]
        out = []
        for (model, key), g in sorted(groups.items()):
            decided_p = [x for x in g["prompts"] if x is not None]
            out.append({"model": model, key_name: key,
                        "requirements": len(g["reqs"]),
                        "requirement_pass_rate": _rate(g["reqs"]),
                        "pending": sum(x not in PASS_FAIL for x in g["reqs"]),
                        "prompts": len(g["prompts"]),
                        "prompt_success_rate": (sum(decided_p) / len(decided_p)) if decided_p else None})
        return out

    by_type = table(lambda pid, rs: sorted({r["requirement_type"] for r in rs}), "requirement_type")
    for row in by_type:                       # prompt success is not meaningful per type
        row.pop("prompts")
        row.pop("prompt_success_rate")
    return {
        "overall": table(lambda pid, rs: ["all"], "scope"),
        "by_capability": table(lambda pid, rs: [meta.get(pid, {}).get("capability", pid.split("-")[0])],
                               "capability"),
        "by_difficulty": table(lambda pid, rs: [meta.get(pid, {}).get("difficulty", "?")], "difficulty"),
        "by_type": by_type,
    }


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return "" if v is None else str(v)


def save_tables(tables, out_dir, extra=None):
    """One CSV per table + summary.md with all tables as Markdown."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = ["# Benchmark results", ""]
    for name, rows in {**tables, **(extra or {})}.items():
        if not rows:
            continue
        cols = list(rows[0])
        with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows([{k: _fmt(v) for k, v in r.items()} for r in rows])
        md += [f"## {name}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        md += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
        md.append("")
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    return sorted(p.name for p in out_dir.iterdir())


def save_final_rows(rows, path, metadata=None):
    """Final requirement-level results (JSON + CSV) incl. final_source / final_label / review_label."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metadata": metadata or {}, "results": common._json_safe(rows)},
                               indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["model", "prompt_id", "requirement_index", "requirement_type", "expected_value", "evaluator",
              "status", "pass_fail", "score", "final_source", "final_label", "audit", "review_label", "reason"]
    with open(path.with_suffix(".csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r.get(k), ensure_ascii=False) if isinstance(r.get(k), (list, dict))
                            else r.get(k)) for k in fields})
    return path, path.with_suffix(".csv")
