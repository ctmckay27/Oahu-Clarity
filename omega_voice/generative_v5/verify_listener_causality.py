"""Source-role, tactic and waveform counterfactuals for listener_causal_v6."""
import argparse,pathlib,json,subprocess,copy
import numpy as np,soundfile as sf
from .physical import realize as physical
from .articulation import realize as articulate
from .alignment import ForcedAligner
from .session import timeline_from_evaluation,durable_json
from ..causal_v4.runtime import compile_scene
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/listener_causal';d.mkdir(exist_ok=True)
 sources=[(r/'continuation/calibration/unchanged.wav','I understand what happened. Give me a moment to check the sequence.',4),(r/'continuation/context_session/continuous/third/carrier.wav','I will bring it to you. Keep the door open.',6)]
 cases=[]
 for i,(_,text,boundary) in enumerate(sources):
  scenes={'neutral':'There is a chair in the room.','listener_only':'The listener is worried.',
   'care_calm':'Mari cares about the listener. Mari wants to reassure the listener. The listener is not worried.',
   'care_worried':f'Mari cares about the listener. Mari wants to reassure the listener. At spoken word index {boundary}, the listener is worried.',
   'clarify_calm':'Mari clarifies the answer. The listener is not confused.',
   'clarify_confused':f'Mari clarifies the answer. At spoken word index {boundary}, the listener is confused.'}
  cases.extend({'source':i,'id':name,'scene':scene,'text':text}for name,scene in scenes.items())
 durable_json(d/'WAVE_CORPUS.json',cases)
 code='''import sys,json,spacy
from omega_voice.generative_v5.explicit_scene import ExplicitSceneCompiler
c=ExplicitSceneCompiler(spacy.load('en_core_web_sm'),policy='listener_causal_v6')
rows=[]
for case in json.load(sys.stdin):
 result=c.compile(case['scene'],case['text'])
 if not result['admitted']:raise ValueError(result['unresolved'])
 rows.append({'case':case,'result':result})
print(json.dumps(rows))
'''
 process=subprocess.run(['/tmp/mari-parserenv/bin/python','-c',code],input=json.dumps(cases),text=True,capture_output=True,check=True)
 compiled=json.loads(process.stdout);durable_json(d/'WAVE_SCENE_COMPILED.json',compiled)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');rows=[]
 for i,(source,text,boundary)in enumerate(sources):
  base=ev.evaluate(source,text);align=al.align(ev.load(source),text,sha(source),base['wer']==0);clock=timeline_from_evaluation(dict(base,alignment=align));raw,_=sf.read(source,dtype='int16');at=round(clock['words'][boundary]['start']*24000)
  for entry in [x for x in compiled if x['case']['source']==i]:
   name=entry['case']['id'];scene=entry['result']['scene'];plan=compile_scene(text,scene,timeline=clock,policy='listener_causal_v6');out=d/f'{i}_{name}.wav'
   mechanism=articulate if name.startswith('clarify')else physical
   receipt=mechanism(source,out,plan);quality=ev.evaluate(out,text)
   row={'source':i,'case':name,'source_sha256':sha(source),'audio_sha256':sha(out),'scene':scene,'plan':plan,'mechanism':receipt,'quality':quality,'full_completion':False}
   if name in {'care_worried','clarify_confused'}:
    calm=d/f"{i}_{'care_calm' if name=='care_worried' else 'clarify_calm'}.wav"
    x,_=sf.read(calm,dtype='int16');y,_=sf.read(out,dtype='int16');row.update(prefix_exact=bool(np.array_equal(x[:at],y[:at])),changed_after_observation=bool(np.any(x[at:]!=y[at:])),mari_fear_max=max(s['state']['emotional_state']['fear']for s in plan['state_samples']))
   if name=='listener_only':row['observation_without_chosen_response_exact']=sha(out)==sha(d/f'{i}_neutral.wav')
   rows.append(row);durable_json(d/'WAVE_RESULTS.json',rows);print(i,name,quality['wer'],quality['speaker_similarity'],row.get('prefix_exact'),flush=True)
   if quality['wer']!=0 or not quality['quality_screen_pass']:raise ValueError('listener causal realization failed quality')
  # Irrelevant recording metadata cannot change the chosen response.
  scene=copy.deepcopy(next(x['result']['scene']for x in compiled if x['case']['source']==i and x['case']['id']=='care_worried'));scene['metadata']={'chair_color':'blue'}
  p=compile_scene(text,scene,timeline=clock,policy='listener_causal_v6');out=d/f'{i}_irrelevant.wav';physical(source,out,p)
  if sha(out)!=sha(d/f'{i}_care_worried.wav'):raise ValueError('irrelevant scene metadata changed waveform')
 checks={'all_quality':all(x['quality']['quality_screen_pass']and x['quality']['wer']==0 for x in rows),'no_future_leak':all(x.get('prefix_exact',True)for x in rows),'observed_cause_reaches_waveform':all(x.get('changed_after_observation',True)for x in rows),'listener_fear_not_mari_fear':all(x.get('mari_fear_max',0)==0 for x in rows),'observation_alone_does_not_force_empathy':all(x.get('observation_without_chosen_response_exact',True)for x in rows),'irrelevant_exact':True}
 durable_json(d/'WAVE_ASSESSMENT.json',{'checks':checks,'all_pass':all(checks.values()),'scope':'bounded observed listener state, chosen tactic and acoustic consequences; perceived relational coherence not established','full_completion':False})
if __name__=='__main__':main()
