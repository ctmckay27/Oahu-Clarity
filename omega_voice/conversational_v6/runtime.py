"""Mari conversational authenticity scaffold.

This layer sits upstream of the existing Mari causal voice runtime. It models
persistent conversational state, listener-specific state, incremental thought
units, self-monitoring, turn-taking, and embodied phrase budgeting. It does not
change Mari's acoustic identity and it never invents affect from lexical content.

The output is a deterministic, replayable plan that can be compiled downward
into ``omega_voice.causal_v4.runtime.compile_scene`` by ``bridge.py``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "mari-conversational-state/1.0"
PLAN_SCHEMA = "mari-conversational-plan/1.0"

ANCHOR_SHA256 = "73bb894fc75e18dddf76e3327f3bf48e161b7dd8b30fbe4949c3ac7318c29567"
PROFILE_SHA256 = "9c49146ae2bd3c5a32c686ad1705fd912391af4a39db0d63e9dc353640d304fa"

TACTICS = {
    "share", "clarify", "reassure", "protect", "challenge", "set_boundary",
    "invite", "tease", "correct", "confront", "persuade", "admit", "withhold",
}
THOUGHTS = {
    "known", "remembering", "deciding", "discovering", "reconsidering",
    "correcting", "searching", "suppressing", "judging", "realizing",
    "withholding", "changing_tactic", "observing",
}
FEEDBACK = {"none", "understood", "misunderstood", "resistance"}
RELATIONSHIP_KEYS = {
    "trust", "familiarity", "irritation", "concern", "authority", "vulnerability",
    "playfulness", "distance", "suspicion", "expected_knowledge",
}

_WORD = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)
_CORRECTION = re.compile(r"^\s*(?:wait\b|actually\b|no[, ]+i mean\b|i mean\b)", re.I)
_SEARCHING = re.compile(r"\b(?:i think|i'm not sure|i am not sure|maybe|probably|possibly)\b", re.I)
_REMEMBERING = re.compile(r"\b(?:i remember|i just remembered)\b", re.I)
_REALIZING = re.compile(r"\b(?:i realize|i just realized)\b", re.I)


def digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _finite_01(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} outside [0,1]")
    return float(value)


def _words(text: str) -> List[re.Match]:
    return list(_WORD.finditer(text))


def new_conversation_state(session_id: str = "mari-default", listener_id: str = "carl") -> Dict[str, Any]:
    if not session_id or not listener_id:
        raise ValueError("session_id and listener_id are required")
    return {
        "schema": SCHEMA,
        "session_id": session_id,
        "listener_id": listener_id,
        "turn": 0,
        "identity": {
            "subject": "Mari404",
            "continuity_law": "Mari becomes otherwise without becoming someone else",
            "anchor_sha256": ANCHOR_SHA256,
            "profile_sha256": PROFILE_SHA256,
        },
        "mind": {
            "known": {},
            "beliefs": {},
            "uncertain": {},
            "unresolved": [],
            "active_question": None,
            "active_goal": "respond to the listener",
            "attention_target": "listener",
            "current_thought": "known",
        },
        "listener_model": {
            "id": listener_id,
            "known_propositions": [],
            "misunderstandings": [],
            "last_feedback": "none",
            "worry": 0.0,
            "confusion": 0.0,
            "urgency": 0.0,
            "interruption_risk": 0.0,
            "last_utterance": None,
        },
        "relationship": {
            "trust": 0.5,
            "familiarity": 0.5,
            "irritation": 0.0,
            "concern": 0.0,
            "authority": 0.0,
            "vulnerability": 0.0,
            "playfulness": 0.0,
            "distance": 0.5,
            "suspicion": 0.0,
            "expected_knowledge": 0.0,
        },
        "self_monitor": {
            "spoken_units": [],
            "open_repair": False,
            "last_correction_turn": None,
            "last_delivery_hash": None,
        },
        "embodied": {
            "breath_reserve": 0.85,
            "fatigue": 0.0,
            "speech_momentum": 0.0,
            "last_turn_end_s": None,
        },
        "interaction": {
            "phase": "listening",
            "response_latency_s": 0.0,
            "floor_strategy": "available",
            "interrupted_intention": None,
            "last_user_utterance": None,
        },
        "continuity": {
            "parent_state_hash": None,
            "journal_head": None,
        },
    }


def validate_state(state: Dict[str, Any]) -> Dict[str, Any]:
    template = new_conversation_state(state.get("session_id", "mari-default"), state.get("listener_id", "carl"))
    if set(state) != set(template) or state.get("schema") != SCHEMA:
        raise ValueError("state schema mismatch")
    if state["identity"] != template["identity"]:
        raise ValueError("Mari acoustic identity changed inside conversational scaffold")
    if state["listener_model"]["id"] != state["listener_id"]:
        raise ValueError("listener identity mismatch")
    if state["listener_model"]["last_feedback"] not in FEEDBACK:
        raise ValueError("unsupported feedback")
    for key in ("worry", "confusion", "urgency", "interruption_risk"):
        _finite_01(f"listener_model.{key}", state["listener_model"][key])
    for key in RELATIONSHIP_KEYS:
        _finite_01(f"relationship.{key}", state["relationship"][key])
    for key in ("breath_reserve", "fatigue", "speech_momentum"):
        _finite_01(f"embodied.{key}", state["embodied"][key])
    if state["mind"]["current_thought"] not in THOUGHTS:
        raise ValueError("unsupported current thought")
    if not isinstance(state["turn"], int) or state["turn"] < 0:
        raise ValueError("invalid turn")
    digest(state)
    return state


def observe_listener(
    state: Dict[str, Any],
    listener_text: str,
    explicit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply source-grounded listener observations without inferring hidden affect."""
    validate_state(state)
    if not isinstance(listener_text, str) or not listener_text.strip():
        raise ValueError("listener_text is required")
    explicit = copy.deepcopy(explicit or {})
    allowed = {
        "feedback", "worry", "confusion", "urgency", "interruption_risk",
        "known_propositions", "misunderstandings", "relationship",
    }
    unknown = set(explicit) - allowed
    if unknown:
        raise ValueError(f"unsupported listener observations: {sorted(unknown)}")

    out = copy.deepcopy(state)
    model = out["listener_model"]
    model["last_utterance"] = listener_text
    out["interaction"]["last_user_utterance"] = listener_text

    feedback = explicit.get("feedback", model["last_feedback"])
    if feedback not in FEEDBACK:
        raise ValueError("unsupported feedback")
    model["last_feedback"] = feedback
    for key in ("worry", "confusion", "urgency", "interruption_risk"):
        if key in explicit:
            model[key] = _finite_01(key, explicit[key])
    if "known_propositions" in explicit:
        vals = explicit["known_propositions"]
        if not isinstance(vals, list) or not all(isinstance(x, str) and x.strip() for x in vals):
            raise ValueError("known_propositions must be nonempty strings")
        model["known_propositions"] = list(dict.fromkeys(vals))
    if "misunderstandings" in explicit:
        vals = explicit["misunderstandings"]
        if not isinstance(vals, list) or not all(isinstance(x, str) and x.strip() for x in vals):
            raise ValueError("misunderstandings must be nonempty strings")
        model["misunderstandings"] = list(dict.fromkeys(vals))
    if "relationship" in explicit:
        rel = explicit["relationship"]
        if not isinstance(rel, dict) or set(rel) - RELATIONSHIP_KEYS:
            raise ValueError("invalid relationship patch")
        for key, value in rel.items():
            out["relationship"][key] = _finite_01(f"relationship.{key}", value)

    if feedback == "misunderstood":
        out["self_monitor"]["open_repair"] = True
    elif feedback == "understood":
        out["self_monitor"]["open_repair"] = False

    validate_state(out)
    return out


def _initial_thought(response_text: str, requested: str) -> str:
    if requested not in THOUGHTS:
        raise ValueError("unsupported thought mode")
    stripped = response_text.strip()
    if _CORRECTION.search(stripped):
        return "correcting"
    if _REMEMBERING.search(stripped):
        return "remembering"
    if _REALIZING.search(stripped):
        return "realizing"
    if _SEARCHING.search(stripped):
        return "searching"
    return requested


def _unit_thought(text: str, fallback: str) -> str:
    if _CORRECTION.search(text):
        return "correcting"
    if _REMEMBERING.search(text):
        return "remembering"
    if _REALIZING.search(text):
        return "realizing"
    if _SEARCHING.search(text):
        return "searching"
    return fallback


def _phrase_capacity(state: Dict[str, Any]) -> int:
    reserve = state["embodied"]["breath_reserve"]
    fatigue = state["embodied"]["fatigue"]
    return max(7, min(24, int(round(9 + 14 * reserve - 5 * fatigue))))


def _raw_units(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    pieces = [x.strip() for x in re.split(r"(?<=[.!?;:])\s+|(?<=,)\s+", text) if x.strip()]
    return pieces or [text]


def _split_long_unit(unit: str, capacity: int) -> List[str]:
    words = unit.split()
    if len(words) <= capacity:
        return [unit]
    out: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + capacity)
        if end < len(words):
            search_lo = max(start + 5, end - 5)
            for idx in range(end - 1, search_lo - 1, -1):
                token = re.sub(r"[^A-Za-z']", "", words[idx]).lower()
                if token in {"and", "but", "because", "so", "then", "while", "if", "when"}:
                    end = idx
                    break
            if end <= start:
                end = min(len(words), start + capacity)
        out.append(" ".join(words[start:end]))
        start = end
    return out


def _unitize(text: str, capacity: int) -> List[Dict[str, Any]]:
    parts: List[str] = []
    for raw in _raw_units(text):
        parts.extend(_split_long_unit(raw, capacity))
    out: List[Dict[str, Any]] = []
    cursor = 0
    for idx, part in enumerate(parts):
        count = len(_words(part))
        if count == 0:
            continue
        out.append({"index": idx, "text": part, "start_word": cursor, "end_word": cursor + count})
        cursor += count
    return out


def _speech_act(text: str) -> str:
    stripped = text.strip()
    if stripped.endswith("?"):
        return "ask"
    if _CORRECTION.search(stripped):
        return "repair"
    if _SEARCHING.search(stripped):
        return "qualified_assertion"
    return "assert"


def _turn_taking(state: Dict[str, Any], thought: str) -> Dict[str, Any]:
    feedback = state["listener_model"]["last_feedback"]
    urgency = state["listener_model"]["urgency"]
    risk = state["listener_model"]["interruption_risk"]
    latency = 0.18
    if thought in {"searching", "remembering", "reconsidering", "deciding"}:
        latency += 0.18
    if thought == "correcting":
        latency += 0.08
    if state["self_monitor"]["open_repair"] or feedback == "misunderstood":
        latency += 0.10
    latency -= 0.08 * urgency
    latency += 0.08 * risk
    latency = max(0.08, min(0.65, latency))
    if state["self_monitor"]["open_repair"]:
        strategy = "repair"
    elif thought in {"searching", "remembering", "reconsidering", "deciding"}:
        strategy = "considered"
    elif urgency > 0.65:
        strategy = "immediate"
    else:
        strategy = "ordinary"
    return {"response_latency_s": round(latency, 3), "floor_strategy": strategy}


def _embodied_schedule(state: Dict[str, Any], units: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    reserve = state["embodied"]["breath_reserve"]
    fatigue = state["embodied"]["fatigue"]
    scheduled: List[Dict[str, Any]] = []
    for unit in units:
        count = unit["end_word"] - unit["start_word"]
        cost = min(0.42, 0.018 * count * (1.0 + 0.35 * fatigue))
        intake = 0.0
        if reserve - cost < 0.18:
            intake = round(min(0.62, 0.22 + 0.34 * (0.35 - reserve) + 0.10 * fatigue), 3)
            reserve = min(0.88, reserve + 0.42)
        reserve = _clamp(reserve - cost)
        scheduled.append({**unit, "quiet_intake_before_s": intake, "breath_cost": round(cost, 4), "reserve_after": round(reserve, 4)})
    return scheduled, reserve


def _causal_events(
    state: Dict[str, Any],
    units: List[Dict[str, Any]],
    listener_text: str,
    intent: Dict[str, Any],
) -> List[Dict[str, Any]]:
    source = {"kind": "conversation", "text": listener_text}
    events: List[Dict[str, Any]] = []

    tactic = intent["tactic"]
    events.append({
        "kind": "action", "id": "conv-action-" + digest([state["turn"], tactic, listener_text])[:12],
        "at_word": 0, "source": source, "tactic": tactic,
        "target": intent["target"], "objective": intent["objective"],
        "obstacle_kind": "social" if state["self_monitor"]["open_repair"] else "none",
        "obstacle": "repair listener model" if state["self_monitor"]["open_repair"] else "",
    })

    if any(state["listener_model"][k] > 0 for k in ("worry", "confusion")):
        events.append({
            "kind": "listener_condition", "id": "conv-listener-" + digest([state["turn"], listener_text])[:12],
            "at_word": 0, "source": source, "listener_id": state["listener_id"],
            "values": {k: state["listener_model"][k] for k in ("worry", "confusion") if state["listener_model"][k] > 0},
        })

    feedback = state["listener_model"]["last_feedback"]
    if feedback != "none":
        events.append({
            "kind": "feedback", "id": "conv-feedback-" + digest([state["turn"], feedback])[:12],
            "at_word": 0, "source": source, "feedback": feedback,
        })

    rel_patch = {f"relationship.{k}": state["relationship"][k] for k in sorted(RELATIONSHIP_KEYS)}
    events.append({
        "kind": "set", "id": "conv-relationship-" + digest([state["turn"], rel_patch])[:12],
        "at_word": 0, "source": source, "values": rel_patch,
    })

    last_thought = None
    for unit in units:
        if unit["thought"] != last_thought:
            events.append({
                "kind": "thought", "id": "conv-thought-" + digest([state["turn"], unit["index"], unit["thought"]])[:12],
                "at_word": unit["start_word"], "source": {"kind": "self_monitor", "text": unit["text"]},
                "mode": unit["thought"],
            })
            last_thought = unit["thought"]

    if intent.get("epistemic_proposition"):
        status = intent["knowledge_status"]
        events.append({
            "kind": "knowledge", "id": "conv-knowledge-" + digest([state["turn"], intent["epistemic_proposition"]])[:12],
            "at_word": 0, "source": {"kind": "intent", "text": intent["epistemic_source"]},
            "status": status, "proposition": intent["epistemic_proposition"], "confidence": intent["certainty"],
        })

    return sorted(events, key=lambda e: (e["at_word"], e["id"]))


def plan_turn(
    state: Dict[str, Any],
    listener_text: str,
    response_text: str,
    *,
    observation: Optional[Dict[str, Any]] = None,
    intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Plan one Mari conversational response without committing it as delivered."""
    validate_state(state)
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("response_text is required")

    observed = observe_listener(state, listener_text, observation)
    raw_intent = copy.deepcopy(intent or {})
    allowed_intent = {
        "tactic", "objective", "target", "thought", "certainty", "knowledge_status",
        "epistemic_proposition", "epistemic_source", "disclosure",
    }
    unknown = set(raw_intent) - allowed_intent
    if unknown:
        raise ValueError(f"unsupported intent keys: {sorted(unknown)}")
    tactic = raw_intent.get("tactic", "clarify" if observed["self_monitor"]["open_repair"] else "share")
    if tactic not in TACTICS:
        raise ValueError("unsupported tactic")
    thought = _initial_thought(response_text, raw_intent.get("thought", "known"))
    certainty = _finite_01("certainty", raw_intent.get("certainty", 0.7))
    disclosure = _finite_01("disclosure", raw_intent.get("disclosure", 1.0))
    knowledge_status = raw_intent.get("knowledge_status", "known")
    if knowledge_status not in {"known", "beliefs", "suspicions", "unresolved"}:
        raise ValueError("unsupported knowledge_status")
    epistemic_source = raw_intent.get("epistemic_source", "conversation state")
    if not isinstance(epistemic_source, str) or not epistemic_source.strip():
        raise ValueError("epistemic_source must be nonempty")
    intent_full = {
        "tactic": tactic,
        "objective": raw_intent.get("objective", tactic),
        "target": raw_intent.get("target", "address the listener's current turn"),
        "thought": thought,
        "certainty": certainty,
        "knowledge_status": knowledge_status,
        "epistemic_proposition": raw_intent.get("epistemic_proposition"),
        "epistemic_source": epistemic_source,
        "disclosure": disclosure,
    }

    capacity = _phrase_capacity(observed)
    units = _unitize(response_text, capacity)
    if not units:
        raise ValueError("response has no speakable units")

    fallback = thought
    correction_seen = False
    enriched: List[Dict[str, Any]] = []
    for unit in units:
        mode = _unit_thought(unit["text"], fallback)
        if mode == "correcting":
            correction_seen = True
        act = _speech_act(unit["text"])
        local_certainty = certainty
        if mode in {"searching", "reconsidering"}:
            local_certainty = min(local_certainty, 0.58)
        micro: List[Dict[str, Any]] = []
        if mode in {"searching", "remembering", "deciding", "reconsidering"}:
            micro.append({"kind": "decision_latency", "cause": mode})
        if mode == "correcting":
            micro.append({"kind": "repair_reset", "cause": "explicit correction language or supplied thought state"})
        if observed["self_monitor"]["open_repair"] and unit["index"] == 0:
            micro.append({"kind": "repair_orientation", "cause": "listener misunderstanding remains open"})
        enriched.append({
            **unit,
            "thought": mode,
            "speech_act": act,
            "certainty": round(local_certainty, 3),
            "disclosure": disclosure,
            "microbehavior": micro,
            "self_monitor": {
                "hears_prior_units": unit["index"] > 0,
                "repair_open_before": observed["self_monitor"]["open_repair"] or correction_seen,
            },
        })
        fallback = "known" if mode not in {"searching", "remembering", "reconsidering", "deciding"} else mode

    enriched, reserve_after = _embodied_schedule(observed, enriched)
    turn_taking = _turn_taking(observed, thought)
    events = _causal_events(observed, enriched, listener_text, intent_full)

    plan = {
        "schema": PLAN_SCHEMA,
        "session_id": observed["session_id"],
        "listener_id": observed["listener_id"],
        "turn": observed["turn"] + 1,
        "listener_text": listener_text,
        "response_text": response_text,
        "state_before": copy.deepcopy(state),
        "state_observed": observed,
        "intent": intent_full,
        "phrase_capacity_words": capacity,
        "turn_taking": turn_taking,
        "units": enriched,
        "embodied_projection": {"breath_reserve_after_full_delivery": round(reserve_after, 4)},
        "causal_events": events,
        "renderer_contract": {
            "acoustic_identity_change": False,
            "generic_tts_fallback": False,
            "random_disfluency": False,
            "style_tag_shortcut": False,
            "time_evolving_state_required": True,
            "silent_intakes_are_timing_constraints_not_fake_breath_audio": True,
        },
        "epistemic_boundary": "lexical text does not independently infer hidden emotion or listener state",
    }
    plan["plan_hash"] = digest(plan)
    return plan


def verify_plan(plan: Dict[str, Any]) -> bool:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("plan schema mismatch")
    supplied = plan.get("plan_hash")
    bare = {k: v for k, v in plan.items() if k != "plan_hash"}
    if supplied != digest(bare):
        raise ValueError("plan hash mismatch")
    validate_state(plan["state_before"])
    validate_state(plan["state_observed"])
    if plan["state_before"]["identity"] != plan["state_observed"]["identity"]:
        raise ValueError("identity changed during observation")
    return True


def commit_delivery(
    state: Dict[str, Any],
    plan: Dict[str, Any],
    *,
    delivered_word_count: Optional[int] = None,
    interrupted: bool = False,
) -> Dict[str, Any]:
    """Commit only speech that was actually delivered.

    Planned but unheard material remains intention, not conversational history.
    Complete units are the commit boundary. A partial final unit is recorded as an
    interruption remainder, not as a fully spoken claim.
    """
    validate_state(state)
    verify_plan(plan)
    if digest(state) != digest(plan["state_before"]):
        raise ValueError("plan belongs to a different conversational state")

    total_words = len(_words(plan["response_text"]))
    if delivered_word_count is None:
        delivered_word_count = total_words
    if not isinstance(delivered_word_count, int) or delivered_word_count < 0 or delivered_word_count > total_words:
        raise ValueError("invalid delivered_word_count")

    out = copy.deepcopy(plan["state_observed"])
    completed = [u for u in plan["units"] if u["end_word"] <= delivered_word_count]
    partial = next((u for u in plan["units"] if u["start_word"] < delivered_word_count < u["end_word"]), None)
    remaining = [u for u in plan["units"] if u["end_word"] > delivered_word_count]

    for unit in completed:
        out["self_monitor"]["spoken_units"].append({
            "turn": plan["turn"],
            "unit_index": unit["index"],
            "text": unit["text"],
            "thought": unit["thought"],
            "speech_act": unit["speech_act"],
            "hash": digest([plan["turn"], unit["index"], unit["text"]]),
        })
        if unit["thought"] == "correcting":
            out["self_monitor"]["last_correction_turn"] = plan["turn"]
            out["self_monitor"]["open_repair"] = False

    words_spoken = sum(u["end_word"] - u["start_word"] for u in completed)
    reserve = out["embodied"]["breath_reserve"]
    fatigue = out["embodied"]["fatigue"]
    reserve = _clamp(reserve - min(0.48, words_spoken * 0.012 * (1 + 0.3 * fatigue)))
    if delivered_word_count == total_words and not interrupted:
        reserve = max(reserve, plan["embodied_projection"]["breath_reserve_after_full_delivery"])
    out["embodied"]["breath_reserve"] = round(reserve, 4)
    out["embodied"]["speech_momentum"] = _clamp(0.25 + min(0.55, words_spoken / 80.0)) if words_spoken else 0.0

    out["interaction"]["response_latency_s"] = plan["turn_taking"]["response_latency_s"]
    out["interaction"]["floor_strategy"] = plan["turn_taking"]["floor_strategy"]
    if delivered_word_count < total_words or interrupted:
        out["interaction"]["phase"] = "interrupted"
        out["interaction"]["interrupted_intention"] = {
            "turn": plan["turn"],
            "partial_unit": copy.deepcopy(partial),
            "remaining_units": copy.deepcopy(remaining),
            "response_hash": digest(plan["response_text"]),
        }
    else:
        out["interaction"]["phase"] = "listening"
        out["interaction"]["interrupted_intention"] = None

    out["mind"]["active_goal"] = plan["intent"]["objective"]
    out["mind"]["current_thought"] = plan["units"][-1]["thought"] if completed else plan["intent"]["thought"]
    prop = plan["intent"].get("epistemic_proposition")
    if prop and completed:
        status = plan["intent"]["knowledge_status"]
        if status == "unresolved":
            if prop not in out["mind"]["unresolved"]:
                out["mind"]["unresolved"].append(prop)
        else:
            out["mind"][status][prop] = {
                "confidence": plan["intent"]["certainty"],
                "source": plan["intent"]["epistemic_source"],
            }
            if prop in out["mind"]["unresolved"]:
                out["mind"]["unresolved"].remove(prop)

    out["turn"] = plan["turn"]
    out["continuity"]["parent_state_hash"] = digest(state)
    out["self_monitor"]["last_delivery_hash"] = digest({
        "plan_hash": plan["plan_hash"],
        "delivered_word_count": delivered_word_count,
        "interrupted": interrupted,
        "completed_units": [u["index"] for u in completed],
    })
    out["continuity"]["journal_head"] = out["self_monitor"]["last_delivery_hash"]
    validate_state(out)
    return out


def interrupted_intention(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    validate_state(state)
    return copy.deepcopy(state["interaction"]["interrupted_intention"])
