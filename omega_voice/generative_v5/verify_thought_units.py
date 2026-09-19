"""Complete thought units with same-speaker prior-audio context.

This is the successor to rejected text starvation. No future thought text is
present in an earlier generation call; only delivered Mari audio can condition
the next unit. Identity and contextual-continuation quality require measurement.
"""
import argparse,json,pathlib,soundfile as sf,numpy as np
from .native import render
from ..causal_v4.runtime import ANCHOR,compile_scene,digest
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/thought_units';d.mkdir(exist_ok=True)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');rows=[]
 def synth(name,text,seed,reference=None):
  p=d/(name+'.wav');receipt=render(r,text,p,seed,reference=reference);q=ev.evaluate(p,text)
  row={'id':name,'text':text,'audio':str(p),'quality':q,'renderer':receipt,'full_completion':False};rows.append(row);(d/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,q['asr_text'],q['wer'],q['speaker_similarity'],flush=True)
  return p,q
 prefix='I thought the package was upstairs.';first,q=synth('prefix',prefix,94800)
 if not q['quality_screen_pass'] or q['wer']!=0:raise ValueError('earlier thought failed quality; no conditioning admission')
 reference={'audio':str(first),'text':prefix,'sha256':sha(first),'source_carrier_sha256':ANCHOR,'role':'same-speaker preceding delivered thought; no voice selection or style instruction'}
 endings=['I can see it beside the window.','It is still missing from the cupboard.'];parts=[]
 for i,text in enumerate(endings):
  p,q=synth(str(i)+'_context',text,94801,reference);parts.append((p,q))
 fresh,fq=synth('fresh_control',endings[0],94801)
 checks={'all_context_quality':all(q['quality_screen_pass'] and q['wer']==0 for _,q in parts),'preceding_audio_affects_next':sha(parts[0][0])!=sha(fresh)}
 for i,(p,q) in enumerate(parts):
  if not q['quality_screen_pass'] or q['wer']!=0:continue
  aa,sr=sf.read(first,dtype='int16');bb,sr2=sf.read(p,dtype='int16');assert sr==sr2==24000
  joined=d/(str(i)+'_episode.wav');sf.write(joined,np.concatenate([aa,bb]),sr,subtype='PCM_16')
  whole=ev.evaluate(joined,prefix+' '+endings[i]);checks[str(i)+'_episode_quality']=whole['quality_screen_pass'] and whole['wer']==0
  (d/(str(i)+'_episode.json')).write_text(json.dumps({'quality':whole,'prefix_sha256':sha(first),'continuation_sha256':sha(p),'join':'unmodified PCM concatenation at complete thought boundary; no inserted random pause or crossfade','full_completion':False},indent=2))
 (d/'ASSESSMENT.json').write_text(json.dumps({'checks':checks,'full_completion':False,'scope':'thought-unit information isolation and same-speaker acoustic context; cognitive naturalness not implied'},indent=2)+'\n');print(checks)
if __name__=='__main__':main()
