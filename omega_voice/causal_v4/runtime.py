"""Event-sourced Mari performance, independent of any audio renderer.

Coefficients are bounded engineering hypotheses, not measurements of physiology.
No affect is inferred from the words being spoken. Scene events supply causes.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import re
from typing import Any

SCHEMA = "mari-voice-acting-state/1.0"
ANCHOR = "73bb894fc75e18dddf76e3327f3bf48e161b7dd8b30fbe4949c3ac7318c29567"
PROFILE = "9c49146ae2bd3c5a32c686ad1705fd912391af4a39db0d63e9dc353640d304fa"
THOUGHTS = {"known", "remembering", "deciding", "discovering", "reconsidering",
            "correcting", "searching", "suppressing", "judging", "realizing",
            "withholding", "changing_tactic", "observing"}
TACTICS = {"share", "clarify", "reassure", "protect", "challenge", "set_boundary",
           "invite", "tease", "correct", "confront", "persuade", "admit", "withhold"}
REL_KEYS = {"trust", "familiarity", "irritation", "concern", "authority", "vulnerability",
            "playfulness", "distance", "suspicion", "expected_knowledge"}
EMOTIONS = {"fear", "anger", "amusement", "tenderness", "curiosity", "shame", "relief"}
BODY = {"fatigue", "exertion", "breath_reserve", "tension", "suppression_effort"}
NUMERIC_PATHS = {
    **{"emotional_state."+x: (0,1) for x in EMOTIONS},
    **{"body_state."+x: (0,1) for x in BODY},
    **{"relationship."+x: (0,1) for x in REL_KEYS},
    "mental_state.attention":(0,1), "mental_state.load":(0,1),
    "knowledge_state.certainty":(0,1), "stakes":(0,1),
    "mask.display_target":(0,1), "mask.effort":(0,1),
    "listener_model.monitoring":(0,1), "listener_model.resistance":(0,1),
    "interaction_state.urgency":(0,1), "interaction_state.elapsed_s":(0,3600),
    "subtext.disclosure":(0,1), "subtext.conflict":(0,1),
    "environment.listener_distance_m":(.25,20),
}

def digest(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",",":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def words(text):
    return list(re.finditer(r"[\w]+(?:['’][\w]+)*", text, re.UNICODE))

def new_state(session_id="mari-default", listener_id="unspecified"):
    if not session_id or not listener_id:
        raise ValueError("session and listener identity required")
    return {
        "schema":SCHEMA, "session_id":session_id, "turn":0, "time_s":0.0,
        "identity":{"subject":"Mari404", "adult":True, "lineage":"G1-1/Mari Voice v1",
                    "anchor_sha256":ANCHOR, "profile_sha256":PROFILE},
        "character":{"continuity_law":"Mari becomes otherwise without becoming someone else",
                     "commitments":[], "shared_history":[], "listener_histories":{}},
        "body_state":{"fatigue":0.0,"exertion":0.0,"breath_reserve":.85,
                      "tension":.08,"suppression_effort":0.0},
        "vocal_configuration":{"support":.85,"pressure":.2,"projection":.4,
                               "attack_softness":.25,"precision":.5,"breathiness":0.0},
        "mental_state":{"attention":.8,"load":0.0,"thought":"known"},
        "emotional_state":{k:0.0 for k in sorted(EMOTIONS)},
        "objective":{"action":"share","target":"communicate the proposition"},
        "obstacle":{"kind":"none","description":""}, "tactic":"share", "stakes":.2,
        "subtext":{"private_meaning":"","disclosure":1.0,"conflict":0.0},
        "mask":{"active":False,"dimension":"fear","display_target":0.0,"effort":0.0},
        "leakage":{"pressure":0.0,"threshold":.72,"active":False,"cause":None},
        "listener_model":{"id":listener_id,"monitoring":.2,"resistance":0.0,
                          "known_propositions":[],"misunderstandings":[]},
        "relationship":{k:(.5 if k in {"trust","distance"} else 0.0) for k in sorted(REL_KEYS)},
        "knowledge_state":{"certainty":.7,"known":{},"beliefs":{},"suspicions":{},
                           "withheld":[],"unresolved":[]},
        "language":"English", "dialect_register":{"dialect":"anchor","register":"conversational"},
        "speech_behavior":{"lexical_policy":"preserve_supplied_text","nonlexical_permission":False},
        "prosody":{}, "microbehavior":[], "nonlexical_behavior":[],
        "interaction_state":{"phase":"speaking","urgency":0.0,"elapsed_s":0.0,
                             "interrupted_intention":None,"repair_open":False},
        "medium":"ordinary_conversation", "performance_grammar":"naturalistic",
        "environment":{"listener_distance_m":1.5,"noise_level":"quiet"},
        "capture":{"mode":"dry_mono","microphone_distance_m":.3},
        "processing":{"effects":[]}, "constraints":{"random_disfluency":False,"fallback":False},
        "temporal_trajectory":[], "continuity_links":{"parent_state_hash":None,"journal_head":None},
        "previous_vocal_state":None,
    }

def validate_state(s):
    expected=set(new_state())
    if set(s)!=expected or s.get("schema")!=SCHEMA:
        raise ValueError("state schema/fields mismatch")
    if s["identity"]!=new_state()["identity"]:
        raise ValueError("frozen carrier identity violation")
    if s["language"]!="English" or s["medium"]!="ordinary_conversation":
        raise ValueError("out of ordinary-English conversation scope")
    if s["tactic"] not in TACTICS or s["mental_state"]["thought"] not in THOUGHTS:
        raise ValueError("unsupported tactic/thought")
    if s["turn"]<0 or s["time_s"]<0 or not s["listener_model"]["id"]:
        raise ValueError("invalid continuity coordinates")
    for path,(lo,hi) in NUMERIC_PATHS.items():
        x=get_path(s,path)
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not lo<=x<=hi:
            raise ValueError("invalid state coordinate: "+path)
    digest(s)
    return s

def get_path(s,path):
    v=s
    for p in path.split("."):v=v[p]
    return v

def put(s,path,value):
    parts=path.split(".");v=s
    for p in parts[:-1]:v=v[p]
    v[parts[-1]]=value

def numeric_patch(s,patch):
    for p,v in patch.items():
        if p not in NUMERIC_PATHS:
            raise ValueError("unknown/uncontrolled coordinate: "+p)
        lo,hi=NUMERIC_PATHS[p]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:
            raise ValueError("coordinate outside lawful range: "+p)
        put(s,p,float(v))

def compile_direction(direction, at_word=0):
    """Bounded language interface. Unknown clauses fail rather than becoming style prompts.

    General scene understanding is supplied through the typed scene API with source
    spans; this recognizer does not claim unrestricted language understanding.
    """
    d=direction.strip().lower().rstrip(".")
    entries={
        "mari is trying not to sound scared": {"kind":"mask","dimension":"fear","internal":.75,"display":.1,"effort":.8},
        "she is trying not to sound scared": {"kind":"mask","dimension":"fear","internal":.75,"display":.1,"effort":.8},
        "mari is searching for a word": {"kind":"thought","mode":"searching"},
        "mari remembers": {"kind":"thought","mode":"remembering"},
        "mari is deciding": {"kind":"thought","mode":"deciding"},
        "mari reconsiders": {"kind":"thought","mode":"reconsidering"},
        "mari realizes what happened": {"kind":"thought","mode":"realizing"},
        "mari already knows the answer": {"kind":"thought","mode":"known"},
        "mari corrects herself": {"kind":"thought","mode":"correcting"},
        "mari chooses not to say it": {"kind":"thought","mode":"withholding"},
        "mari is exhausted": {"kind":"set","values":{"body_state.fatigue":.8}},
    }
    if d not in entries:raise ValueError("unresolved direction; supply source-grounded typed scene events")
    return {"id":"direction-"+digest([direction,at_word])[:12],"at_word":at_word,
            "source":{"kind":"direction","text":direction},**entries[d]}

def apply_event(s,e,policy=None):
    k=e["kind"]
    old_certainty=s['knowledge_state']['certainty']
    if k=="set":numeric_patch(s,e["values"])
    elif k=="thought":
        mode=e["mode"]
        if mode not in THOUGHTS:raise ValueError("unknown thought operation")
        s["mental_state"]["thought"]=mode
        # Completion is an event, never a keyword in the spoken text.
        if mode in {"realizing","correcting"}:
            s["knowledge_state"]["certainty"]=max(.85,s["knowledge_state"]["certainty"])
            s["mental_state"]["load"]*=.35
        elif mode in {"searching","remembering","reconsidering","deciding"}:
            s["mental_state"]["load"]=max(.55,s["mental_state"]["load"])
        elif mode=="withholding":s["subtext"]["disclosure"]=.1
    elif k=="knowledge":
        status=e["status"];prop=e["proposition"]
        if status not in {"known","beliefs","suspicions","unresolved"} or not prop:raise ValueError("bad knowledge event")
        confidence=float(e["confidence"])
        if not 0<=confidence<=1:raise ValueError("bad confidence")
        if status=='unresolved':
            if prop not in s['knowledge_state']['unresolved']:s['knowledge_state']['unresolved'].append(prop)
        else:
            s["knowledge_state"]["certainty"]=confidence
            s["knowledge_state"][status][prop]={"confidence":confidence,"source":e["source"]}
            if prop in s["knowledge_state"]["unresolved"]:s["knowledge_state"]["unresolved"].remove(prop)
        if policy=='epistemic_focus_v2' and prop!=s['knowledge_state'].get('assertion',{}).get('proposition'):
            # Knowing an unrelated fact does not strengthen this assertion.
            s['knowledge_state']['certainty']=old_certainty
    elif k=='assertion':
        if policy!='epistemic_focus_v2':raise ValueError('assertion focus requires epistemic_focus_v2')
        prop=e.get('proposition');mode=e.get('mode','assert')
        if not isinstance(prop,str) or not prop.strip() or mode not in {'assert','admit_unknown','ask'}:raise ValueError('invalid assertion focus')
        confidence=e.get('confidence')
        if confidence is None:
            evidence=next((s['knowledge_state'][status][prop] for status in ['known','beliefs','suspicions'] if prop in s['knowledge_state'][status]),None)
            if evidence is None:raise ValueError('assertion needs proposition evidence or explicitly sourced commitment')
            confidence=evidence['confidence']
        if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('invalid assertion confidence')
        s['knowledge_state']['assertion']={'proposition':prop,'mode':mode,'source':copy.deepcopy(e['source'])}
        s['knowledge_state']['certainty']=float(confidence)
    elif k=="action":
        if e["tactic"] not in TACTICS:raise ValueError("unknown tactic")
        s["objective"]={"action":e.get("objective",e["tactic"]),"target":e["target"]}
        s["tactic"]=e["tactic"]
        s["obstacle"]={"kind":e.get("obstacle_kind","social"),"description":e.get("obstacle","")}
    elif k=="mask":
        if e["dimension"] not in EMOTIONS:raise ValueError("invalid mask dimension")
        numeric_patch(s,{"emotional_state."+e["dimension"]:e["internal"],
                         "mask.display_target":e["display"],"mask.effort":e["effort"]})
        s["mask"].update(active=True,dimension=e["dimension"])
        s["listener_model"]["monitoring"]=max(.8,s["listener_model"]["monitoring"])
    elif k=="unmask":s["mask"]["active"]=False;s["mask"]["effort"]=0.0
    elif k=="listener":
        old=s["listener_model"]["id"];new=e["listener_id"]
        if not new:raise ValueError("empty listener")
        s["character"]["listener_histories"][old]={"relationship":copy.deepcopy(s["relationship"]),
                                                "model":copy.deepcopy(s["listener_model"])}
        stored=s["character"]["listener_histories"].get(new)
        fresh=new_state(listener_id=new)
        s["relationship"]=copy.deepcopy(stored["relationship"] if stored else fresh["relationship"])
        s["listener_model"]=copy.deepcopy(stored["model"] if stored else fresh["listener_model"])
        numeric_patch(s,{"relationship."+k:v for k,v in e.get("relationship",{}).items()})
    elif k=="feedback":
        feedback=e["feedback"]
        if feedback=="resistance":
            s["listener_model"]["resistance"]=clamp(s["listener_model"]["resistance"]+.35)
            s["relationship"]["irritation"]=clamp(s["relationship"]["irritation"]+.15)
            s["tactic"]="clarify" if s["tactic"]=="persuade" else "set_boundary"
            s["mental_state"]["thought"]="changing_tactic"
        elif feedback=="understood":
            s["listener_model"]["resistance"]*=.4;s["interaction_state"]["repair_open"]=False
            s["relationship"]["trust"]=clamp(s["relationship"]["trust"]+.08)
            s["emotional_state"]["relief"]=clamp(s["emotional_state"]["relief"]+.25)
        elif feedback=="misunderstood":s["interaction_state"]["repair_open"]=True;s["tactic"]="clarify"
        else:raise ValueError("unsupported listener feedback")
    elif k=="interrupt":
        s["interaction_state"].update(phase="interrupted",interrupted_intention=copy.deepcopy(s["objective"]))
    elif k=="resume":
        if s["interaction_state"]["phase"]!="interrupted":raise ValueError("resume without interruption")
        s["interaction_state"]["phase"]="speaking"
        s["objective"]=s["interaction_state"]["interrupted_intention"]
        s["interaction_state"]["interrupted_intention"]=None
        s["mental_state"]["thought"]="correcting"
    elif k=="commitment":
        if e["proposition"] not in s["character"]["commitments"]:
            s["character"]["commitments"].append(e["proposition"])
    elif k=="nonlexical":
        if e["behavior"] not in {"laugh","sigh","acknowledgment"}:raise ValueError("unsupported nonlexical event")
        s["nonlexical_behavior"].append({"behavior":e["behavior"],"cause":e["id"],"at_word":e["at_word"]})
    else:raise ValueError("unknown event kind: "+str(k))
    if policy=='epistemic_focus_v2' and k=='thought' and e.get('mode') in {'realizing','correcting'}:
        # Completing a thought may reveal uncertainty or an error. Its
        # epistemic result must come from evidence, not the operation label.
        s['knowledge_state']['certainty']=old_certainty
    validate_state(s)

def advance(s,dt,speaking=True):
    b=s["body_state"];em=s["emotional_state"];m=s["mask"]
    # Residue is time-dependent and does not reset at each synthesis call.
    for k,tau in {"fear":25,"anger":45,"amusement":14,"shame":60,"relief":25}.items():
        em[k]*=math.exp(-dt/tau)
    b["suppression_effort"]=m["effort"] if m["active"] else b["suppression_effort"]*math.exp(-dt/2)
    desired=clamp(.08+.35*em["fear"]+.30*em["anger"]+.25*b["suppression_effort"]+.15*b["exertion"])
    b["tension"]+=(desired-b["tension"])*(1-math.exp(-dt/1.2))
    b["breath_reserve"]=clamp(b["breath_reserve"]+dt*((-.025-.03*b["exertion"]) if speaking else .20))
    b["fatigue"]=clamp(b["fatigue"]+dt*(.0007 if speaking else -.004))
    s["time_s"]+=dt

def realize_state(s,at_word,active_causes):
    """Derived vocal-production intent; support status is separate in the adapter."""
    b=s["body_state"];em=s["emotional_state"];r=s["relationship"];l=s["listener_model"]
    thought=s["mental_state"]["thought"];certainty=s["knowledge_state"]["certainty"]
    search={"remembering":.42,"deciding":.35,"searching":.65,"reconsidering":.48,
            "discovering":.3,"judging":.15,"withholding":.30}.get(thought,0.0)
    irritation=max(em["anger"],r["irritation"])
    closeness=r["trust"]*r["familiarity"]*(1-r["distance"])
    urgency=s["interaction_state"]["urgency"]
    threat=em["fear"]+.4*b["tension"]
    mask=s["mask"]
    leakage=clamp(threat-mask["effort"]*.65) if mask["active"] else 0.0
    s["leakage"].update(pressure=leakage,active=leakage>.72,cause=active_causes[-1] if leakage>.72 and active_causes else None)
    support=clamp(b["breath_reserve"]*(1-.35*b["fatigue"])/(1+.35*b["exertion"]))
    pressure=clamp(.2+.25*irritation+.18*urgency+.12*b["tension"]-.12*b["fatigue"])
    projection=clamp(.4-.20*closeness+.13*urgency+.04*r["authority"])
    attack=clamp(.25+.35*em["tenderness"]+.12*r["concern"]-.15*irritation)
    precision=clamp(.5+.28*irritation+.13*l["resistance"]+.12*(s["tactic"] in {"clarify","correct","set_boundary"})-.08*b["fatigue"])
    s["vocal_configuration"]={"support":support,"pressure":pressure,"projection":projection,
                              "attack_softness":attack,"precision":precision,"breathiness":0.0}
    rate=clamp(1+.07*urgency-.09*search-.05*b["fatigue"]-.03*l["resistance"]+.015*r["expected_knowledge"],.86,1.10)
    delay=.26*search+.05*l["monitoring"]*(1-s["subtext"]["disclosure"])
    # Suppression costs cognition without turning private feeling into a preset.
    delay+=.04*b["suppression_effort"]*em["fear"]
    if thought in {"realizing","correcting"}:delay=.045;rate=min(1.06,rate+.025)
    finality=clamp(certainty*(1-.35*search)+.08*(s["tactic"]=="set_boundary"))
    constraints={"rate":rate,"gain_db":clamp((projection-.4)*7+(pressure-.2)*1.5,-3,2),
                 "onset_delay_s":min(.24,delay),"finality":finality,
                 "attack_softness":attack,"precision":precision,"support":support,
                 "breathiness":0.0,"phonatory_tension":b["tension"],
                 "phrase_capacity_words":max(6,int(12+14*support-5*b["exertion"])),
                 "emphasis":.05*em["amusement"]*(1-r["distance"])}
    s["prosody"]=copy.deepcopy(constraints)
    # Only event-boundary behaviors are scheduled. No RNG creates a tic.
    micro=[]
    if delay>.02 and active_causes:micro.append({"kind":"decision_latency","cause":active_causes[-1],"seconds":delay})
    if leakage>.72 and active_causes:micro.append({"kind":"suppression_leak","cause":active_causes[-1],"pressure":leakage})
    s["microbehavior"]=micro
    return {"at_word":at_word,"time_s":s["time_s"],"thought":thought,
            "causes":list(active_causes),"controls":constraints,"microbehavior":copy.deepcopy(micro),
            "state_hash":digest(s)}

def compile_scene(text,scene=None,prior=None,timeline=None,policy=None):
    if not isinstance(text,str) or not words(text):raise ValueError("spoken text required")
    if policy not in {None,'bounded_thought_recovery_v1','epistemic_focus_v2'}:raise ValueError('unknown temporal policy')
    def evolve(state,dt,speaking=True):
        advance(state,dt,speaking)
        if policy and state['mental_state']['thought'] in {'realizing','correcting'}:
            # Resolved cognition has a brief integration phase. Knowledge and
            # relational residue persist independently after that phase ends.
            # Time constant is an explicit engineering hypothesis, not a
            # measurement of Mari's cognitive physiology.
            state['mental_state']['load']*=math.exp(-dt/.4)
            if dt>0 and state['mental_state']['load']<.05:state['mental_state']['thought']='known'
    # Native renderer interprets these as directives, so raw input cannot smuggle them.
    if re.search(r"[\[\]<>]",text):raise ValueError("renderer markup is not spoken text")
    scene=copy.deepcopy(scene or {})
    allowed={"session_id","listener_id","events","directions","elapsed_s","metadata"}
    if set(scene)-allowed:raise ValueError("unknown scene field")
    original=copy.deepcopy(prior if prior is not None else new_state(scene.get("session_id","mari-default"),scene.get("listener_id","unspecified")))
    validate_state(original)
    if scene.get("session_id",original["session_id"])!=original["session_id"]:raise ValueError("session mismatch")
    s=copy.deepcopy(original)
    if scene.get("listener_id",s["listener_model"]["id"])!=s["listener_model"]["id"]:
        raise ValueError("listener changes require explicit listener event")
    elapsed=scene.get("elapsed_s",0.0)
    if not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or not 0<=elapsed<=3600:raise ValueError("invalid elapsed time")
    evolve(s,elapsed,speaking=False)
    events=scene.get("events",[])+[compile_direction(x["text"],x.get("at_word",0)) for x in scene.get("directions",[])]
    n=len(words(text));ids=set()
    if timeline is not None:
        if set(timeline)!={'words','duration_s','source_audio_sha256'} or len(timeline['words'])!=n or not timeline['source_audio_sha256']:
            raise ValueError('incomplete realization timeline')
        duration=timeline['duration_s']
        if not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<=0:
            raise ValueError('invalid realized duration')
        last_end=0.0
        for t in timeline['words']:
            if set(t)!={'start','end'} or any(not isinstance(t[k],(int,float)) or not math.isfinite(t[k]) for k in t):
                raise ValueError('invalid word clock')
            if not last_end<=t['start']<t['end']<=duration+.001:
                raise ValueError('nonmonotonic word clock')
            last_end=t['end']
        advance(s,timeline['words'][0]['start'],speaking=False)
    for e in events:
        if not isinstance(e.get("at_word"),int) or not 0<=e["at_word"]<=n:raise ValueError("event boundary out of range")
        if not e.get("id") or e["id"] in ids:raise ValueError("missing/duplicate event id")
        if not isinstance(e.get("source"),dict) or not e["source"].get("text"):raise ValueError("event source required")
        ids.add(e["id"])
    # Stable sort preserves causal ordering of events at the same location.
    events.sort(key=lambda e:e["at_word"])
    grouped={}
    for e in events:grouped.setdefault(e["at_word"],[]).append(e)
    active=[];knots=[];journal=[];state_samples=[]
    head=original["continuity_links"]["journal_head"]
    for i in range(n+1):
        for e in grouped.get(i,[]):
            before=digest(s);apply_event(s,e,policy=policy);active.append(e["id"])
            if policy and e['kind']=='thought' and e.get('mode') in {'realizing','correcting'}:
                s['mental_state']['load']=max(.12,s['mental_state']['load'])
            row={"event":e,"before":before,"after":digest(s),"parent":head}
            row["hash"]=digest(row);head=row["hash"];journal.append(row)
        knot=realize_state(s,i,active)
        if policy and not any(e['kind'] in {'thought','mask','feedback','resume'} for e in grouped.get(i,[])):
            # A searching state can persist, but entering it is one event. Do
            # not schedule another onset hesitation at each subsequent word.
            knot['controls']['onset_delay_s']=0.
            knot['microbehavior']=[x for x in knot['microbehavior'] if x['kind']!='decision_latency']
            s['prosody']['onset_delay_s']=0.;s['microbehavior']=copy.deepcopy(knot['microbehavior'])
            knot['state_hash']=digest(s)
        knots.append(knot)
        if i in grouped or i==0 or i==n:state_samples.append({"at_word":i,"state":copy.deepcopy(s)})
        if i<n:
            speaking=s["interaction_state"]["phase"]=="speaking"
            if timeline is None:evolve(s,.30/knot["controls"]["rate"],speaking=speaking)
            else:
                clock=timeline['words'][i]
                evolve(s,clock['end']-clock['start'],speaking=speaking)
                next_start=timeline['words'][i+1]['start'] if i+1<n else timeline['duration_s']
                evolve(s,max(0,next_start-clock['end']),speaking=False)
    s["turn"]+=1
    s["previous_vocal_state"]=copy.deepcopy(original["vocal_configuration"])
    s["temporal_trajectory"]=[{"at_word":k["at_word"],"thought":k["thought"],"state_hash":k["state_hash"]} for k in knots]
    s["continuity_links"]={"parent_state_hash":digest(original),"journal_head":head}
    validate_state(s)
    plan={"schema":"mari-causal-performance/1.0","text":text,"words":[m.group() for m in words(text)],
          "scene":scene,"initial_state":original,"final_state":s,"trajectory":knots,
          "state_samples":state_samples,"journal":journal,"event_count":len(events),
          "renderer_instruction":None,"coefficient_status":"bounded_engineering_hypotheses",
          "acoustic_identity_frozen":True}
    if timeline is not None:
        plan.update(schema='mari-causal-performance/1.1',realization_timeline=copy.deepcopy(timeline))
    if policy:plan.update(schema='mari-causal-performance/1.2',temporal_policy=policy)
    plan["plan_hash"]=digest(plan)
    return plan

def replay(text,scene,prior=None):
    return compile_scene(text,scene,prior)

def verify_plan(plan):
    bare={k:v for k,v in plan.items() if k!="plan_hash"}
    if digest(bare)!=plan.get("plan_hash"):raise ValueError("plan hash mismatch")
    if compile_scene(plan["text"],plan["scene"],plan["initial_state"],timeline=plan.get('realization_timeline'),policy=plan.get('temporal_policy'))!=plan:
        raise ValueError("plan replay mismatch")
    return True
