"""Mari acoustic actuator v2.

Maps Presence-v7 distinctions into bounded, identity-preserving renderer controls.
This is an actuator/coupling layer, not a new model of Mari.
"""
from __future__ import annotations
import copy, hashlib, json
from typing import Any, Dict, List, Optional

SCHEMA="mari-acoustic-actuator/2.0"
ANCHOR="73bb894fc75e18dddf76e3327f3bf48e161b7dd8b30fbe4949c3ac7318c29567"
PROFILE="9c49146ae2bd3c5a32c686ad1705fd912391af4a39db0d63e9dc353640d304fa"

def _digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def _clamp(x,a,b): return max(a,min(b,float(x)))

def compile_actuation(realization:Dict[str,Any], presence:Dict[str,Any]) -> Dict[str,Any]:
    if realization.get("schema")!="mari-conversational-realization-plan/1.0": raise ValueError("bad realization plan")
    if presence.get("schema")!="mari-presence-plan/1.0": raise ValueError("bad presence plan")
    if presence["identity"]["anchor_sha256"]!=ANCHOR or presence["identity"]["profile_sha256"]!=PROFILE: raise ValueError("identity mismatch")
    if realization["identity"]["anchor_sha256"]!=ANCHOR or realization["identity"]["profile_sha256"]!=PROFILE: raise ValueError("identity mismatch")
    if presence["source_plan_hash"]!=realization["conversation_plan_hash"]: raise ValueError("plans do not share conversational source")
    prominence={x["word_index"]:x for x in presence["semantic_prominence"]}
    micro={x["unit"]:x for x in presence["microprosody"]}
    thought={x["unit"]:x for x in presence["thought_to_speech"]}
    units=[]
    for u in realization["units"]:
        c=copy.deepcopy(u["controls"]); idx=u["index"]
        t=thought.get(idx,{})
        m=micro.get(idx,{})
        focused=[p for wi,p in prominence.items() if u["start_word"]<=wi<u["end_word"]]
        focus_strength=max([p["strength"] for p in focused],default=0.0)
        # Renderer-native channels: rate, volume, pre-pause, join.
        c["pre_pause_s"]=round(_clamp(max(c["pre_pause_s"],t.get("latency_before_s",0)),0,.7),3)
        c["rate"]=round(_clamp(c["rate"]*(1-.018*focus_strength),.90,1.07),4)
        c["volume"]=round(_clamp(c["volume"]*(1+.018*focus_strength),.92,1.06),4)
        # Extra channels are explicit requests to an expanded renderer. They are
        # bounded and independently disable-able for identity-envelope sweeps.
        extra={
          "prominence_strength":round(focus_strength,3),
          "onset_energy":round(_clamp(.50 + .16*(m.get("onset_variation",.05)-.05),.46,.58),3),
          "consonant_energy":round(_clamp(m.get("consonant_energy",.52),.46,.64),3),
          "finality":round(_clamp(m.get("phrase_finality",.65),.45,.84),3),
          "timing_flex":round(_clamp(m.get("timing_flex",.08),.04,.20),3),
        }
        units.append({**u,"controls":c,"expanded_controls":extra,
          "actuation_causes":{"thought":t.get("thought"),"semantic_focus":[p["token"] for p in focused],
                              "micro_causes":m.get("cause",[]),"relationship":presence["relationship_realization"]}})
    out={"schema":SCHEMA,"realization_plan_hash":realization["plan_hash"],"presence_hash":presence["presence_hash"],
         "identity":{"anchor_sha256":ANCHOR,"profile_sha256":PROFILE,"change":False},
         "units":units,"nonlexical_events":copy.deepcopy(presence["nonlexical_events"]),
         "envelope":{"rate":[.90,1.07],"volume":[.92,1.06],"pre_pause_s":[0,.7],
                     "onset_energy":[.46,.58],"consonant_energy":[.46,.64],"finality":[.45,.84],"timing_flex":[.04,.20]},
         "laws":{"speaker_profile_constant":True,"random_humanization":False,"cause_required":True,
                 "silent_fallback":False,"expanded_controls_require_renderer_support":True}}
    out["actuator_hash"]=_digest(out); return out

def envelope_sweep(plan:Dict[str,Any]) -> List[Dict[str,Any]]:
    """Produce one-factor-at-a-time diagnostic conditions around neutral."""
    conditions=[{"id":"control","patch":{}}]
    axes={"rate":[.92,.96,1.04,1.06],"volume":[.94,.97,1.03,1.05],
          "onset_energy":[.47,.50,.55,.57],"consonant_energy":[.47,.51,.58,.63],
          "finality":[.48,.58,.74,.82],"timing_flex":[.05,.08,.14,.19]}
    for axis,vals in axes.items():
        for v in vals: conditions.append({"id":f"{axis}-{v}","patch":{axis:v}})
    return conditions

def blind_manifest(actuator:Dict[str,Any]) -> Dict[str,Any]:
    ids=[{"blind_id":"X"+_digest(c)[:10],"condition":c} for c in envelope_sweep(actuator)]
    return {"schema":"mari-acoustic-blind-manifest/1.0","dimensions":["human_naturalness","same_speaker","Mari_identity","situational_fit","generatedness"],
            "conditions":ids,"reveal_only_after_judgment":True}
