"""
eval_temporal.py — count / order / simultaneous (Count-, Order-, SimultaneousEvaluator)
=======================================================================================

Owner: C (plan §8)

All three read the SAME event list from events.py, so "how many jumps" and
"was the jump before the turn" always use one detection.

Linking (definition v1.2)
  count         applies_to -> the action; its body_side (if any) filters the events
  order         event_sequence: [{"event", "requirement_ids"}, ...] in the required order
  simultaneous  event_group:    [{"event", "requirement_ids"}, ...] that must overlap

Pilot: count C4-01/02 (1), C4-09 (2), C4-10 (3), C5-16 (1);
       order C4-01, C4-02, C5-16;  simultaneous C4-19, C5-16.

Decisions (plan §4)
  count         number of events == N
  order         the FIRST event of each listed action; starts strictly increase
                (by >= min_start_gap_frames) in the required order
  simultaneous  best overlap between the listed actions >= min_overlap_ratio
                of the shorter event
"""

import itertools

import events as E
from common import register_evaluator
from rule_base import current, events_json, verdict

RuleEvaluator = current()   # base class bound to the loaded common.py (safe after reload)


class CountEvaluator(RuleEvaluator):
    EVALUATOR_NAME = "CountEvaluator"
    PROVISIONAL_THRESHOLDS = {"tolerance": 0}
    CURRENT_THRESHOLDS = dict(PROVISIONAL_THRESHOLDS)
    CURRENT_THRESHOLD_STATUS = "provisional_not_frozen"

    def calculate_evidence(self, motion, requirement, case):
        ctx = E.requirement_context(requirement, case)
        if ctx["action_req"] is None:
            return {"unsupported": "count does not refer to an action"}
        a = self.analyse(motion)
        kind, evs = E.events_for_action(a, ctx["action_req"], case)
        if kind is None:
            return {"unsupported": f"no event detector for action {ctx['action']!r}",
                    "action": ctx["action"]}
        return {"action": ctx["action"], "event_kind": kind, "body_side": ctx["body_side"],
                "detected_count": len(evs), "events": events_json(evs), "score": len(evs)}

    def decide(self, ev, expected):
        n = int(expected)
        ok = abs(ev["detected_count"] - n) <= self.thresholds["tolerance"]
        return verdict(ok), f"{ev['event_kind']}: detected {ev['detected_count']}, required {n}"


class OrderEvaluator(RuleEvaluator):
    EVALUATOR_NAME = "OrderEvaluator"
    PROVISIONAL_THRESHOLDS = {"min_start_gap_frames": 1}
    CURRENT_THRESHOLDS = dict(PROVISIONAL_THRESHOLDS)
    CURRENT_THRESHOLD_STATUS = "provisional_not_frozen"

    def calculate_evidence(self, motion, requirement, case):
        a = self.analyse(motion)
        steps = []
        for entry in E.entries_for(requirement, case, "event_sequence"):
            kind, evs = E.events_for_entry(a, entry, case)
            if kind is None:
                return {"unsupported": f"no event detector for {entry.get('event')!r}"}
            first = min(evs, key=lambda e: e.start, default=None)
            steps.append({"event": entry.get("event"), "event_kind": kind,
                          "n_events": len(evs),
                          "first_start": first.start if first else None,
                          "first_end": first.end if first else None})
        starts = [s["first_start"] for s in steps]
        gaps = [b - a_ for a_, b in zip(starts, starts[1:])] if None not in starts else []
        return {"steps": steps, "missing": [s["event"] for s in steps if s["first_start"] is None],
                "start_gaps_frames": gaps, "score": min(gaps) if gaps else None}

    def decide(self, ev, expected):
        if ev["missing"]:
            return "FAIL", f"not detected: {ev['missing']}"
        need = self.thresholds["min_start_gap_frames"]
        ok = all(g >= need for g in ev["start_gaps_frames"])
        timeline = " -> ".join(f"{s['event']}@{s['first_start'] / 20:.2f}s" for s in ev["steps"])
        return verdict(ok), f"first occurrences: {timeline}"


class SimultaneousEvaluator(RuleEvaluator):
    EVALUATOR_NAME = "SimultaneousEvaluator"
    PROVISIONAL_THRESHOLDS = {"min_overlap_ratio": 0.5}
    CURRENT_THRESHOLDS = dict(PROVISIONAL_THRESHOLDS)
    CURRENT_THRESHOLD_STATUS = "provisional_not_frozen"

    def calculate_evidence(self, motion, requirement, case):
        a = self.analyse(motion)
        groups = []
        for entry in E.entries_for(requirement, case, "event_group"):
            kind, evs = E.events_for_entry(a, entry, case)
            if kind is None:
                return {"unsupported": f"no event detector for {entry.get('event')!r}"}
            groups.append((entry.get("event"), kind, evs))
        best, best_combo = 0.0, None
        for combo in itertools.product(*[g[2] for g in groups]):
            start = max(e.start for e in combo)
            end = min(e.end for e in combo)
            shortest = min(e.duration for e in combo)
            ratio = max(0, end - start) / max(shortest, 1)
            if ratio > best:
                best, best_combo = ratio, combo
        return {
            "groups": [{"event": g[0], "event_kind": g[1], "n_events": len(g[2]),
                        "events": events_json(g[2], 5)} for g in groups],
            "missing": [g[0] for g in groups if not g[2]],
            "best_overlap_ratio": float(best),
            "best_combination": [e.to_dict() for e in best_combo] if best_combo else None,
            "score": float(best),
        }

    def decide(self, ev, expected):
        if ev["missing"]:
            return "FAIL", f"not detected: {ev['missing']}"
        need = self.thresholds["min_overlap_ratio"]
        return verdict(ev["best_overlap_ratio"] >= need), (
            f"best overlap {ev['best_overlap_ratio']:.0%} of the shorter event (need {need:.0%})")


register_evaluator("CountEvaluator", CountEvaluator)
register_evaluator("OrderEvaluator", OrderEvaluator)
register_evaluator("SimultaneousEvaluator", SimultaneousEvaluator)
