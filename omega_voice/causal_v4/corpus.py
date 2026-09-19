"""Predeclared causal coverage, counterfactuals and held-out combinations."""
from .runtime import compile_scene,compile_direction

def ev(kind,i=0,**kw):
    return {"id":f"{kind}-{i}","kind":kind,"at_word":i,
            "source":{"kind":"authored_scene","text":kw.pop("cause","specified causal situation")},**kw}
def setv(**kw):return ev("set",values={k.replace("__","."):v for k,v in kw.items()})
def thought(mode,i=0):return ev("thought",i,mode=mode,cause="thought operation: "+mode)
def action(tactic):return ev("action",tactic=tactic,target="the immediate conversational outcome")

def cases():
    common="I understand what happened. Give me a moment to check the sequence."
    rows=[
      ("neutral",common,[]),
      ("observation","The light is still on in the room across the street.",[thought("observing")]),
      ("reasoning","If the door was already open, someone must have arrived earlier.",[thought("deciding")]),
      ("realization","The reflection moved first. That means the window was open.",[thought("searching"),thought("realizing",5)]),
      ("correction","I said the second door. I meant the one beside it.",[thought("known"),thought("correcting",5)]),
      ("uncertainty",common,[thought("searching"),setv(knowledge_state__certainty=.3)]),
      ("certainty",common,[thought("known"),setv(knowledge_state__certainty=.95)]),
      ("practical_action","Set the cup down here. I'll clear a space beside the lamp.",[action("clarify")]),
      ("reassurance","You can leave that with me. Let's deal with the immediate problem.",[action("reassure"),setv(relationship__concern=.7)]),
      ("tenderness","You don't have to explain it yet. I can stay a little longer.",[setv(emotional_state__tenderness=.75,relationship__trust=.9,relationship__familiarity=.8,relationship__distance=.1)]),
      ("dry_amusement","You found the one drawer that was supposed to stay closed.",[action("tease"),setv(emotional_state__amusement=.45,relationship__playfulness=.5)]),
      ("genuine_amusement","You were trying to be helpful. I know. That makes it better.",[setv(emotional_state__amusement=.8),ev("nonlexical",5,behavior="laugh",cause="shared unexpected absurdity")]),
      ("contained_irritation","We already agreed on this. Tell me what has actually changed.",[setv(emotional_state__anger=.45),action("set_boundary")]),
      ("stronger_anger","You knew the door was broken. You sent them through it anyway.",[setv(emotional_state__anger=.9),action("confront")]),
      ("urgency","Leave the bag. Come through the door with me now.",[setv(interaction_state__urgency=.85),action("protect")]),
      ("authority","Stay beside the wall. I'll tell you when there is room to move.",[setv(relationship__authority=.8),action("protect")]),
      ("curiosity","What made you notice that part of the picture first?",[setv(emotional_state__curiosity=.75),thought("discovering")]),
      ("guardedness",common,[setv(relationship__suspicion=.7,relationship__trust=.2,subtext__disclosure=.3),thought("withholding")]),
      ("increased_trust",common,[ev("listener",listener_id="trusted",relationship={"trust":.9,"familiarity":.7,"distance":.15})]),
      ("familiarity",common,[setv(relationship__trust=.85,relationship__familiarity=.95,relationship__distance=.1,relationship__expected_knowledge=.8)]),
      ("disagreement","I see why you think that. I don't think the evidence fits.",[action("challenge"),thought("judging")]),
      ("withholding","There is something else. It can wait until we get outside.",[thought("withholding"),setv(subtext__conflict=.8,subtext__disclosure=.2)]),
      ("fatigue",common,[setv(body_state__fatigue=.85,body_state__exertion=.3)]),
      ("interruption_recovery","Let me finish the part about the door. Then you can tell me.",[action("clarify"),ev("interrupt",3),ev("resume",6)]),
      ("concealed_fear","Take my hand. We have enough room to get through together.",[compile_direction("Mari is trying not to sound scared"),action("protect")]),
      ("remembering","I remember the bench being closer to the water than this.",[thought("remembering"),thought("realizing",6)]),
      ("reconsidering","That should work. Unless the weight is carried by the other side.",[thought("known"),thought("reconsidering",3)]),
    ]
    result=[{"id":name,"split":"development","text":text,"scene":{"events":events}} for name,text,events in rows]
    result += [
      {"id":"heldout_trust_under_pressure","split":"held_out","text":"Keep the blue folder with you. We'll sort out the rest when the lift arrives.","scene":{"events":[setv(relationship__trust=.86,relationship__familiarity=.71,relationship__distance=.2,interaction_state__urgency=.64,body_state__fatigue=.25),action("protect")]}},
      {"id":"heldout_tender_disagreement","split":"held_out","text":"I believe you. I also think we should check before we call anyone.","scene":{"events":[setv(emotional_state__tenderness=.63,relationship__concern=.5),action("challenge"),thought("deciding",5)]}},
      {"id":"heldout_relief_to_judgment","split":"held_out","text":"The latch is free. Good. Now tell me why you thought pulling harder would help.","scene":{"events":[setv(emotional_state__relief=.8),thought("judging",6),action("clarify")]}},
      {"id":"long_form","split":"held_out","text":"Before we begin, I want to check one thing. The route on the map crosses the old bridge, but the photograph shows a temporary barrier. That may be nothing more than maintenance. It could also mean the crossing is closed. We should find out before carrying everything down there. I remember a second path beside the garden. It takes longer, and the last stretch is narrow, but there is room to stop if we need to. Yes, that is the better choice. Leave the heavy case here for now. We can come back for it after we know the path is clear. I know you wanted to finish this tonight. So did I. Getting there together matters more than proving that the first plan was right.","scene":{"events":[thought("deciding"),thought("remembering",53),thought("realizing",80),setv(relationship__trust=.8,relationship__familiarity=.8,relationship__distance=.2)]}},
    ]
    # Typed scene events are authored test inputs. The corpus does not assert that
    # any keyword recognizer discovered these causes from the spoken wording.
    return result

def evaluate_structural_corpus():
    results=[]
    for c in cases():
        p=compile_scene(c["text"],c["scene"])
        results.append({"id":c["id"],"split":c["split"],"plan_hash":p["plan_hash"],
                        "words":len(p["words"]),"events":p["event_count"],"pass":True})
    # Explicit continuity episode: disagreement has residue, repair changes it.
    a=compile_scene("That wasn't what we agreed.",{"events":[setv(emotional_state__anger=.65),action("set_boundary")]})
    b=compile_scene("I understand. Thank you for telling me.",{"events":[ev("feedback",feedback="understood")]},a["final_state"])
    assert b["final_state"]["emotional_state"]["anger"]>0
    assert b["final_state"]["relationship"]["trust"]>a["final_state"]["relationship"]["trust"]
    results.append({"id":"multi_turn_continuity","split":"held_out","pass":True,"state_hash":b["plan_hash"]})
    return results
