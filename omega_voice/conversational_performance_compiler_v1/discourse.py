"""Discourse and information-structure analysis for Mari's conversational performance compiler.

This layer does not infer hidden emotion from text. It resolves concrete
conversational actions, relationships between utterance units, and information
status that can legitimately constrain performance planning.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Set

_WORD = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_CONTRAST = re.compile(r"\b(?:but|instead|rather|not|actually|except|though|however)\b", re.I)
_CORRECTION = re.compile(r"^\s*(?:wait\b|actually\b|no[, ]+i mean\b|i mean\b)", re.I)
_CONTINUATION = re.compile(r"(?:[,;:]|\b(?:and|but|because|so|then|while|if|when))\s*$", re.I)


def _words(text: str) -> List[str]:
    return [m.group(0).lower().replace("’", "'") for m in _WORD.finditer(text)]


def _content_words(text: str) -> Set[str]:
    stop = {
        "a","an","the","i","you","we","they","he","she","it","this","that","these","those",
        "is","am","are","was","were","be","been","being","do","does","did","have","has","had",
        "to","of","for","in","on","at","by","with","from","as","and","or","but","so","if","then",
        "my","your","our","their","me","him","her","us","them","yes","no","yeah","okay","ok",
    }
    return {w for w in _words(text) if w not in stop and len(w) > 1}


def resolve_communicative_state(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve what the utterance is doing without converting affect labels to acoustics."""
    intent = plan["intent"]
    observed = plan["state_observed"]
    units = plan["units"]
    listener = observed["listener_model"]

    has_repair = any(u.get("speech_act") == "repair" or _CORRECTION.search(u["text"]) for u in units)
    has_question = bool(units and units[-1].get("speech_act") == "ask")
    feedback = listener.get("last_feedback", "none")
    tactic = intent.get("tactic", "share")

    if has_repair or feedback == "misunderstood" or tactic == "correct":
        act = "repair"
    elif has_question or tactic == "invite":
        act = "invite_response"
    elif tactic == "set_boundary":
        act = "set_boundary"
    elif tactic == "challenge":
        act = "challenge"
    elif tactic == "admit":
        act = "admit"
    elif tactic == "tease":
        act = "tease"
    elif tactic == "clarify":
        act = "clarify"
    elif tactic == "reassure":
        act = "reassure"
    else:
        act = "share"

    final_text = units[-1]["text"].strip()
    syntactically_open = bool(_CONTINUATION.search(final_text)) or not final_text.endswith((".", "!", "?"))
    response_expected = has_question or act == "invite_response"
    completion = "OPEN" if syntactically_open else "COMPLETE"

    return {
        "act": act,
        "tactic": tactic,
        "objective": intent.get("objective"),
        "target": intent.get("target"),
        "certainty": intent.get("certainty"),
        "listener_feedback": feedback,
        "listener_confusion": listener.get("confusion", 0.0),
        "listener_urgency": listener.get("urgency", 0.0),
        "response_expected": response_expected,
        "completion": completion,
        "basis": {
            "repair_present": has_repair,
            "terminal_question": has_question,
            "explicit_feedback": feedback,
            "syntactically_open": syntactically_open,
        },
        "law": "communicative state may shape performance; relationship/affect labels do not directly become acoustic recipes",
    }


def _focus_for_unit(plan: Dict[str, Any], unit: Dict[str, Any]) -> Dict[str, Any]:
    listener_words = _content_words(plan["listener_text"])
    unit_words = _words(unit["text"])
    content = [w for w in unit_words if w in _content_words(unit["text"])]
    given = [w for w in content if w in listener_words]
    new = [w for w in content if w not in listener_words]

    contrastive: List[str] = []
    words = unit_words
    for i, w in enumerate(words):
        if w in {"not", "instead", "rather", "actually", "but", "except"}:
            contrastive.extend(words[i + 1 : i + 4])
    if _CORRECTION.search(unit["text"]):
        contrastive.extend(new[:3])

    denom = max(1, len(content))
    novelty = len(new) / denom
    contrast_strength = min(1.0, len(set(contrastive)) / 3.0)
    focus_strength = max(0.0, min(1.0, 0.35 * novelty + 0.65 * contrast_strength))

    return {
        "given_terms": sorted(set(given)),
        "new_terms": sorted(set(new)),
        "contrastive_terms": sorted(set(contrastive)),
        "novelty": round(novelty, 4),
        "contrast_strength": round(contrast_strength, 4),
        "focus_strength": round(focus_strength, 4),
        "source": "lexical overlap + explicit contrast/correction structure",
    }


def compile_thought_groups(plan: Dict[str, Any], communicative: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Turn conversational units into discourse-linked thought groups."""
    groups: List[Dict[str, Any]] = []
    listener_is_question = plan["listener_text"].strip().endswith("?")

    for i, unit in enumerate(plan["units"]):
        text = unit["text"].strip()
        speech_act = unit.get("speech_act", "assert")
        if speech_act == "repair" or _CORRECTION.search(text):
            role = "REVISION"
        elif speech_act == "ask":
            role = "QUESTION"
        elif _CONTRAST.search(text):
            role = "CONTRAST"
        elif i == 0 and listener_is_question:
            role = "ANSWER"
        elif i == 0:
            role = "PROPOSITION"
        else:
            role = "ELABORATION"

        if i == 0:
            relation = "RESPONDS_TO_LISTENER"
        elif role == "REVISION":
            relation = "REVISES_PREVIOUS"
        elif role == "CONTRAST":
            relation = "CONTRASTS_WITH_PREVIOUS"
        elif role == "QUESTION":
            relation = "INVITES_RESPONSE"
        else:
            relation = "CONTINUES_PREVIOUS"

        terminal = i == len(plan["units"]) - 1
        closure = "OPEN" if (not terminal or communicative["completion"] == "OPEN") else "TERMINAL"

        groups.append({
            "index": i,
            "unit_index": unit["index"],
            "text": unit["text"],
            "word_span": [unit["start_word"], unit["end_word"]],
            "thought_source": unit.get("thought"),
            "speech_act": speech_act,
            "role": role,
            "relation_to_previous": relation,
            "closure": closure,
            "certainty": unit.get("certainty"),
            "information_focus": _focus_for_unit(plan, unit),
            "explicit_microbehavior": [x.get("kind") for x in unit.get("microbehavior", []) if isinstance(x, dict)],
        })
    return groups
