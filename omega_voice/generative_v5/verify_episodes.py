"""Unseen within-sentence discovery and long-form state combinations."""
import argparse,pathlib,json
from .session import PerformanceSession,durable_json
from .episode import realize
from .alignment import ForcedAligner
from ..causal_v4.evaluate import Evaluator

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/causal_episodes';d.mkdir(exist_ok=True)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json')
 def e(kind,cause,**kw):return dict(id=kind+'-'+str(len(cause)),kind=kind,at_word=0,source={'text':cause},**kw)
 observation='Mari sees the package beside the window after recalling a different location.'
 units=[{'id':'recollection','text':'I thought the package was upstairs,','scene':{'events':[e('assertion','Mari has a provisional memory of the package location.',proposition='package location',confidence=.35),e('thought','Mari recalls where she left the package.',mode='remembering')]},'seed':95000},
 {'id':'discovery','text':'but I can see it beside the window.','boundary_cause':{'text':observation},'scene':{'events':[e('knowledge',observation,proposition='package location',status='known',confidence=.96),e('thought',observation,mode='realizing')]},'seed':95001}]
 long=[
 ('initial','I have the delivery note here, and the address is right. The tracking record says the parcel was left by the side entrance. I checked that door before coming in, and there was nothing beside it. Let me check the photograph before we ask anyone to go back outside.',[e('assertion','Mari is checking a provisional record against her observations.',proposition='parcel location',confidence=.45),e('thought','Mari compares the delivery record with what she observed.',mode='deciding')],None),
 ('new_evidence','There it is. That is the gate behind the workshop, not the entrance to this building. The driver has taken the picture from the other side of the fence. We can reach it from the courtyard without going back along the road.',[e('knowledge','The photograph reveals the parcel location at the workshop gate.',proposition='parcel location',status='known',confidence=.96),e('thought','Mari recognizes the gate in the newly opened photograph.',mode='realizing')],'Mari opens the delivery photograph and recognizes a different gate.'),
 ('listener_concern','I know you are tired. Leave the bags here and give me the key. I will bring the parcel in, then we can decide what needs to be unpacked tonight. The rest can wait until morning.',[e('set','Her familiar trusted listener says they are exhausted.',values={'relationship.trust':.87,'relationship.familiarity':.79,'relationship.distance':.18,'relationship.concern':.7,'emotional_state.tenderness':.45}),e('action','Mari takes responsibility for bringing the parcel in.',tactic='reassure',target='let the tired listener rest')],'The listener says they are exhausted and worried about the remaining work.'),
 ('clarification','The small silver key, please. That one opens the courtyard gate. Thank you. Stay here where it is warm; I should only be a few minutes.',[e('feedback','The listener offers the wrong key and asks which one Mari needs.',feedback='misunderstood'),e('action','Mari specifies the courtyard key so the listener can hand it over.',tactic='clarify',target='obtain the courtyard key')],'The listener offers the wrong key and asks for clarification.')]
 episodes={'mid_sentence':units,'heldout_long_form':[dict(id=name,text=text,scene={'events':events},seed=95100+i,**({'boundary_cause':{'text':cause}} if cause else {})) for i,(name,text,events,cause) in enumerate(long)]}
 results=[]
 durable_json(d/'PREDECLARED_CORPUS.json',episodes)
 for name,items in episodes.items():
  session=PerformanceSession(r,d/name/'session',ev,mode='physical',aligner=al,temporal_policy='embodied_continuity_v3',conditioning='previous_carrier',articulation=True)
  try:
   result=realize(session,items,d/name);row={'id':name,'passed':result['episode_quality_admitted'],'quality':result['quality']}
  except Exception as exc:row={'id':name,'passed':False,'error':str(exc)}
  results.append(row);durable_json(d/'RESULTS.json',results);print(name,row['passed'],row.get('error'),flush=True)
if __name__=='__main__':main()
