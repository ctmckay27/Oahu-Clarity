"""Mari presence v7: seven-frontier causal performance layer.

This module extends conversational_v6 without changing G1-1 acoustic identity.
It compiles thought timing, semantic prominence, embodied continuity,
relationship state, causal microprosody, nonlexical intent, and blind-test
metadata into an explicit deterministic plan. No random humanization.
"""
from __future__ import annotations
import copy, hashlib, json, re
from typing import Any, Dict, List, Optional

SCHEMA = "mari-presence-plan/1.0"
NONLEXICAL = {"mm", "mm_hm", "hm", "huh", "small_laugh", "breath_release", "aborted_start", "silence"}
_WORD = re.compile(r"[\w]+(?:['’][\w]+)*", re.UNICODE)

def _digest(v: Any) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",",":"), ensure_ascii=False).encode()).hexdigest()

def _words(text: str) -> List[str]:
    return [m.group(0) for m in _WORD.finditer(text)]

def _prominence(text: str, focus: Optional[List[str]]) -> List[Dict[str, Any]]:
    words=_words(text); wanted={x.casefold() for x in (focus or [])}
    out=[]
    for i,w in enumerate(words):
        if w.casefold() in wanted:
            out.append({"word_index":i,"token":w,"strength":0.82,"cause":"explicit semantic focus"})
    return out

def _relationship_projection(rel: Dict[str, float]) -> Dict[str, float]:
    trust=float(rel.get("trust",.5)); familiarity=float(rel.get("familiarity",.5))
    irritation=float(rel.get("irritation",0)); concern=float(rel.get("concern",0))
    play=float(rel.get("playfulness",0)); distance=float(rel.get("distance",.5))
    return {
      "directness": round(max(0,min(1,.45+.25*trust+.18*familiarity+.20*irritation-.12*distance)),3),
      "warmth": round(max(0,min(1,.30+.28*trust+.22*concern+.14*play-.18*irritation)),3),
      "guardedness": round(max(0,min(1,.25+.35*distance+.25*(1-trust))),3),
      "playfulness": round(max(0,min(1,play)),3),
    }

def compile_presence(conversational_plan: Dict[str, Any], *,
                     semantic_focus: Optional[List[str]]=None,
                     nonlexical: Optional[List[Dict[str, Any]]]=None,
                     blind_id: Optional[str]=None) -> Dict[str, Any]:
    if conversational_plan.get("schema")!="mari-conversational-plan/1.0":
        raise ValueError("requires conversational_v6 plan")
    units=copy.deepcopy(conversational_plan["units"])
    rel=conversational_plan["state_observed"]["relationship"]
    reserve=conversational_plan["state_observed"]["embodied"]["breath_reserve"]
    fatigue=conversational_plan["state_observed"]["embodied"]["fatigue"]
    thought=[]
    micro=[]
    for u in units:
        thought.append({
          "unit":u["index"],"thought":u["thought"],"speech_act":u["speech_act"],
          "latency_before_s": round(.04 + (.12 if u["thought"] in {"searching","remembering","deciding","reconsidering"} else 0)
                                    + (.06 if u["thought"]=="correcting" else 0),3),
          "quiet_intake_before_s":u["quiet_intake_before_s"],
        })
        micro.append({
          "unit":u["index"],
          "onset_variation": round(.05 + .10*fatigue,3),
          "timing_flex": round(.08 + .10*(1-reserve),3),
          "consonant_energy": round(.52 + .12*rel.get("irritation",0),3),
          "phrase_finality": round(.55 + .22*u["certainty"],3),
          "cause":["embodied_state","thought_state","speech_act"],
        })
    nx=[]
    for item in nonlexical or []:
        kind=item.get("kind")
        if kind not in NONLEXICAL: raise ValueError(f"unsupported nonlexical: {kind}")
        cause=item.get("cause")
        if not isinstance(cause,str) or not cause.strip(): raise ValueError("nonlexical event requires represented cause")
        nx.append({"kind":kind,"at_word":int(item.get("at_word",0)),"cause":cause})
    out={
      "schema":SCHEMA,
      "source_plan_hash":conversational_plan["plan_hash"],
      "identity":{"anchor_sha256":conversational_plan["state_observed"]["identity"]["anchor_sha256"],
                  "profile_sha256":conversational_plan["state_observed"]["identity"]["profile_sha256"],
                  "change":False},
      "thought_to_speech":thought,
      "semantic_prominence":_prominence(conversational_plan["response_text"],semantic_focus),
      "microprosody":micro,
      "embodied_continuity":{"reserve_before":reserve,"fatigue":fatigue,
          "reserve_after_full_delivery":conversational_plan["embodied_projection"]["breath_reserve_after_full_delivery"]},
      "relationship_realization":_relationship_projection(rel),
      "nonlexical_events":nx,
      "blind_character_test":{"blind_id":blind_id or ("B-"+_digest(conversational_plan["plan_hash"])[:10]),
          "hide_condition_labels":True,"dimensions":["speaker_continuity","Mari_identity","naturalness","situational_fit","generatedness"]},
      "forbidden":["random_disfluency","fake_breath_audio","generic_humanization","speaker_substitution","prose_acting_instruction"],
    }
    out["presence_hash"]=_digest(out)
    return out

def verify_presence(plan: Dict[str, Any]) -> bool:
    if plan.get("schema")!=SCHEMA: raise ValueError("schema mismatch")
    supplied=plan.get("presence_hash"); bare={k:v for k,v in plan.items() if k!="presence_hash"}
    if supplied!=_digest(bare): raise ValueError("presence hash mismatch")
    if plan["identity"]["change"] is not False: raise ValueError("identity mutation forbidden")
    return True
