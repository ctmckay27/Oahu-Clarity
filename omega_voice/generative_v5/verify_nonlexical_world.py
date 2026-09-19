"""Explicit acknowledgment event without spelling a nonlexical sound."""
import argparse,json,pathlib,copy
import numpy as np
from .alignment import ForcedAligner
from .session import durable_json
from .nonlexical import compile_event,realize
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.runtime import new_state,ANCHOR
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/nonlexical_world';d.mkdir(exist_ok=True)
 source=r/'continuation/calibration/unchanged.wav';native=json.loads(source.with_suffix('.receipt.json').read_text());text=native['text']
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');q=ev.evaluate(source,text)
 if not q['quality_screen_pass'] or q['wer']!=0:raise ValueError('unqualified source')
 aligner=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');alignment=aligner.align(ev.load(source),text,sha(source),True)
 word=next(x for x in alignment['words'] if x['word']=='moment');char=word['characters'][0]
 if char['letter']!='M' or char['probability']<.35:raise ValueError('nasal source location not verified')
 f=np.load(r/'continuation/world_qualification/0_features.npz');indices=(f['t']>=char['start'])&(f['t']<=char['end'])&(f['f0']>0)
 if indices.sum()<2:raise ValueError('insufficient voiced nasal evidence')
 feature_path=d/'nasal_source.npz'
 np.savez_compressed(feature_path,f0=np.median(f['f0'][indices]),spectrum=np.exp(np.log(np.maximum(f['spectrum'][indices],1e-20)).mean(0)),aperiodicity=f['aperiodicity'][indices].mean(0))
 provenance={'role':'diagnostic measured nasal source; not permanent selected voice asset','source_audio':str(source),'source_audio_sha256':sha(source),'source_native_receipt_sha256':sha(source.with_suffix('.receipt.json')),
  'source_carrier_sha256':ANCHOR,'transcript':text,'word':word,'character_location':char,'caveat':'CTC character timing localizes expected nasal; phonetic/perceptual confirmation remains unqualified',
  'features_sha256':sha(feature_path),'world_analysis_sha256':sha(r/'continuation/world_qualification/0_features.npz'),'selected_analysis_frames':int(indices.sum())}
 durable_json(d/'SOURCE.json',provenance)
 base=new_state();base['knowledge_state']['known']['the gate is open']={'confidence':.95,'source':{'text':'Mari directly checked the gate.'}}
 cause={'id':'acknowledge-listener','behavior':'acknowledgment','intent':'receipt','proposition':'the gate is open','source':{'text':'The listener finishes reporting that the gate is open; Mari acknowledges having heard the report.'}}
 rows=[]
 for name,state,event in [('receipt',base,cause),('agreement',base,dict(cause,intent='agreement',source={'text':'Mari agrees with the listener, having directly checked the gate.'})),
  ('irrelevant_knowledge',copy.deepcopy(base),cause)]:
  if name=='irrelevant_knowledge':state['knowledge_state']['known']['capital of France']={'confidence':.99,'source':{'text':'Unrelated held knowledge.'}}
  plan=compile_event(state,event);p=d/(name+'.wav');result=realize(plan,feature_path,d/'SOURCE.json',p);rows.append({'id':name,'audio':str(p),'sha256':sha(p),'plan_hash':plan['plan_hash'],'full_completion':False})
  durable_json(d/'RESULTS.json',rows)
 durable_json(d/'STRUCTURAL_ASSESSMENT.json',{'irrelevant_fact_exact':sha(d/'receipt.wav')==sha(d/'irrelevant_knowledge.wav'),
  'agreement_changes_waveform':sha(d/'receipt.wav')!=sha(d/'agreement.wav'),'lexical_substitution':False,'perceptual_admission':False,'full_completion':False})
 print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
