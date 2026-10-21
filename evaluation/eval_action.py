"""
action requirements -> ActionEvaluator.

Vocabulary is fixed to events.ACTION_TO_EVENT (walk, turn, jump, kick, reach,
raise_hand, raise_hands + a few synonyms like "walks"/"hop"/"raise"); anything
outside that is not judged by rule. finalize.lint_definition checks the
benchmark definition stays inside this list.

What counts as a pass, from the events detected in events.py:
  walk         longest walk event long enough (walk_min_duration_s) and far enough
               (walk_min_distance_m) - a "walk" event itself already needs root
               speed >= 0.3 m/s for >= 1s with alternating feet
  turn         largest turn event has |yaw change| >= turn_min_deg
  jump         at least one flight phase (both feet off the ground >= 2 frames)
  kick         at least one kick event (one ankle lifted off neutral, other foot planted)
  reach        at least one reach event (wrist >= 0.7 arm length from neutral)
  raise_hand   wrist above its own shoulder for >= 0.25s
  raise_hands  both wrists above their shoulders at once, for >= 0.25s

Side (left/right/both) is BodySideEvaluator's job and counting repeats is
CountEvaluator's - this file only answers "did it happen at all".

An action we don't recognise returns pass_fail=None, which sends it to human
review in finalize.py instead of guessing.
"""

import events as E
from common import register_evaluator
from rule_base import current, events_json, verdict

RuleEvaluator = current()  # rebind to whatever common.py is loaded right now


class ActionEvaluator(RuleEvaluator):
    EVALUATOR_NAME = "ActionEvaluator"
    PROVISIONAL_THRESHOLDS = {
        "walk_min_duration_s": 1.0,
        "walk_min_distance_m": 0.5,
        "turn_min_deg": 45.0,
        "min_events": 1,
    }
 
    CURRENT_THRESHOLDS = {"walk_min_duration_s": 1.0, "walk_min_distance_m": 0.5, "turn_min_deg": 45.0, "min_events": 1}
    CURRENT_THRESHOLD_STATUS = "frozen_v1.0"

    def calculate_evidence(self, motion, requirement, case):
        action = str(self.expected_of(requirement)).lower()
        kind = E.event_kind(action)
        if kind is None:
            return {"unsupported": f"action {action!r} is outside the rule vocabulary -> human review",
                    "action": action, "judge": "human_required"}

        a = self.analyse(motion)
        evs = a.get(kind)                       # left/right is BodySideEvaluator's job, not ours
        ev = {"action": action, "event_kind": kind, "n_events": len(evs), "events": events_json(evs)}
        if kind == "walk":
            best = max(evs, key=lambda e: e.duration, default=None)
            ev["longest_walk_s"] = best.duration / 20 if best else 0.0
            ev["longest_walk_distance_m"] = best.attrs["net_displacement_m"] if best else 0.0
            ev["score"] = ev["longest_walk_distance_m"]
        elif kind == "turn":
            ev["max_turn_deg"] = max((abs(e.attrs["yaw_change_deg"]) for e in evs), default=0.0)
            ev["score"] = ev["max_turn_deg"]
        else:
            ev["score"] = len(evs)
        return ev

    def decide(self, ev, expected):
        t = self.thresholds
        kind = ev["event_kind"]
        if kind == "walk":
            ok = (ev["longest_walk_s"] >= t["walk_min_duration_s"]
                  and ev["longest_walk_distance_m"] >= t["walk_min_distance_m"])
            return verdict(ok), (f"longest walk {ev['longest_walk_s']:.1f} s, "
                                 f"{ev['longest_walk_distance_m']:.2f} m")
        if kind == "turn":
            ok = ev["max_turn_deg"] >= t["turn_min_deg"]
            return verdict(ok), f"largest turn {ev['max_turn_deg']:.0f} deg (need >= {t['turn_min_deg']})"
        ok = ev["n_events"] >= t["min_events"]
        return verdict(ok), f"{ev['n_events']} {kind} event(s) detected"


register_evaluator("ActionEvaluator", ActionEvaluator)
