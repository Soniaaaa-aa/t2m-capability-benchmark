"""
eval_action.py — action requirements (ActionEvaluator)
======================================================

Owner: C — plan §5, simplified by plan B: rules only, no model judge.

Closed vocabulary (= events.ACTION_TO_EVENT; the Main definition must stay inside it,
finalize.lint_definition checks this): walk, turn, jump, kick, reach, raise_hand, raise_hands
(plus synonyms such as walks / hop / raise).

Rules on the events of events.py
  walk         longest walk event >= walk_min_duration_s and its net displacement
               >= walk_min_distance_m (walk event = root speed >= 0.3 m/s for >= 1 s
               with alternating feet)
  turn         largest turn event |Δyaw| >= turn_min_deg
  jump         >= 1 flight phase (both feet off >= 2 frames)
  kick         >= 1 kick event (one ankle away from neutral and lifted, other foot planted)
  reach        >= 1 reach event (wrist away from neutral >= 0.7 arm length)
  raise_hand   >= 1 wrist above its shoulder for >= 0.25 s
  raise_hands  both wrists above the shoulders together for >= 0.25 s

Which side did it (left / right / both) is BodySideEvaluator's job, how many
times is CountEvaluator's — ActionEvaluator only asks "did this action happen".

An action outside the vocabulary returns pass_fail = None; finalize.py then routes
that requirement to human review. No embedding / VLM judge is used (plan B).
"""

import events as E
from common import register_evaluator
from rule_base import current, events_json, verdict

RuleEvaluator = current()   # base class bound to the loaded common.py (safe after reload)


class ActionEvaluator(RuleEvaluator):
    EVALUATOR_NAME = "ActionEvaluator"
    PROVISIONAL_THRESHOLDS = {
        "walk_min_duration_s": 1.0,
        "walk_min_distance_m": 0.5,
        "turn_min_deg": 45.0,
        "min_events": 1,
    }
    CURRENT_THRESHOLDS = dict(PROVISIONAL_THRESHOLDS)
    CURRENT_THRESHOLD_STATUS = "provisional_not_frozen"

    def calculate_evidence(self, motion, requirement, case):
        action = str(self.expected_of(requirement)).lower()
        kind = E.event_kind(action)
        if kind is None:
            return {"unsupported": f"action {action!r} is outside the rule vocabulary -> human review",
                    "action": action, "judge": "human_required"}

        a = self.analyse(motion)
        evs = a.get(kind)                       # side is not checked here (BodySide does)
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
