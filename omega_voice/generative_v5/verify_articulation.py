"""Matched-text precision counterfactuals with unchanged carrier and word clock."""
import argparse,pathlib,json,numpy as np,soundfile as sf
from .articulation import realize
from .alignment import ForcedAligner
from .session import timeline_from_evaluation,durable_json
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.runtime import compile_scene
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/articulation';d.mkdir(exist_ok=True)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');rows=[]
 sources=[r/'continuation/context_session/continuous/third/carrier.wav',r/'continuation/interruption_recovery/preflight.wav']
 texts=['I will bring it to you. Keep the door open.','I checked the front room and the kitchen. The package must be upstairs beside the wardrobe.']
 for i,(source,text) in enumerate(zip(sources,texts)):
  base=ev.evaluate(source,text);alignment=al.align(ev.load(source),text,sha(source),base['wer']==0)
  timeline=timeline_from_evaluation(dict(base,alignment=alignment));event={'id':'listener-needs-precision','kind':'action','at_word':5,'tactic':'clarify','target':'make the requested location unambiguous','source':{'text':'The listener did not catch which location Mari meant and asks her to clarify.'}}
  scenes={'neutral':{},'clarify':{'events':[event]},'irrelevant':{'metadata':{'wall_color':'blue'}}}
  for name,scene in scenes.items():
   plan=compile_scene(text,scene,timeline=timeline,policy='embodied_continuity_v3');audio=d/f'{i}_{name}.wav';p=realize(source,audio,plan)
   q=ev.evaluate(audio,text);yy,_=sf.read(source,dtype='int16');zz,_=sf.read(audio,dtype='int16');boundary=round(timeline['words'][5]['start']*24000)
   row={'text':text,'condition':name,'source':str(source),'plan':plan,'physical':p,'quality':q,'prefix_exact':bool(np.array_equal(yy[:boundary],zz[:boundary])),'full_completion':False};rows.append(row);durable_json(d/'RESULTS.json',rows)
   print(i,name,q['wer'],q['speaker_similarity'],p.get('consonant_to_vowel_ratio_db_after'),row['prefix_exact'],flush=True)
   if not q['quality_screen_pass'] or q['wer']!=0:raise ValueError('articulation quality failed; preserve evidence without retuning')
 checks={'all_quality':all(x['quality']['quality_screen_pass'] and x['quality']['wer']==0 for x in rows),'no_future_effect':all(x['prefix_exact'] for x in rows)}
 for i in range(2):
  checks[str(i)+'_irrelevant_exact']=sha(d/f'{i}_neutral.wav')==sha(d/f'{i}_irrelevant.wav')
  p=rows[i*3+1]['physical'];checks[str(i)+'_consonant_ordering']=p['consonant_to_vowel_ratio_db_after']>p['consonant_to_vowel_ratio_db_before']
 durable_json(d/'ASSESSMENT.json',{'checks':checks,'full_completion':False,'perceived_precision':'unqualified','scope':'situated consonant-energy channel, not complete articulatory control'})
if __name__=='__main__':main()
