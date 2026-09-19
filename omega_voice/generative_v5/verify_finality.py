"""Held-out native counterfactuals with identical text/seed/identity."""
import argparse,json,pathlib
import numpy as np
from .native import render,write_trajectory
from .trajectory import compile_finality,alignment_error
from .acoustics import measure
from ..causal_v4.runtime import compile_scene
from ..causal_v4.evaluate import Evaluator

TEXTS=[
 'I checked the receipt. It arrived on Tuesday.',
 'The spare key is under the blue bowl.',
 'I thought the light was broken. The switch is on the other wall.',
 'You can leave the folder here. I will take care of it.',
]

def event(value,at=0):
 return {'id':'evidence-update-'+str(at),'at_word':at,'kind':'knowledge',
         'status':'known' if value>.7 else 'beliefs','proposition':'spoken proposition',
         'confidence':value,'source':{'kind':'controlled scene','text':
         'Mari has verified the claim directly.' if value>.7 else 'Mari has incomplete evidence and treats the claim as provisional.'}}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();root=pathlib.Path(a.root)
 d=root/'continuation/heldout_finality';d.mkdir(exist_ok=True)
 ev=Evaluator(root/'models/whisper',root/'models/ecapa',root/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 bank=np.load(root/'continuation/calibration/finality_bank.npz')['directions'];records=[]
 for index,text in enumerate(TEXTS):
  seed=91600+index;base=d/f'{index}_carrier.wav'
  if not base.exists():render(root,text,base,seed)
  base_eval=ev.evaluate(base,text);base_eval['acoustics']=measure(base)
  if not base_eval['alignment']['exact_words']:raise RuntimeError('carrier words failed')
  for name,scene in [('verified',{'events':[event(.98)]}),('provisional',{'events':[event(.15)]})]:
   plan=compile_scene(text,scene);alignment=base_eval['alignment'];duration=base_eval['audio']['duration_s']
   trials=[]
   for attempt in range(3):
    stem=d/f'{index}_{name}_{attempt}';weights,packet=compile_finality(plan,alignment,duration)
    write_trajectory(stem.with_suffix('.mtraj'),bank,weights)
    if not stem.with_suffix('.wav').exists():render(root,text,stem.with_suffix('.wav'),seed,stem.with_suffix('.mtraj'))
    result=ev.evaluate(stem.with_suffix('.wav'),text);result['acoustics']=measure(stem.with_suffix('.wav'))
    error=alignment_error(packet['segments'],result['alignment'])
    trial={'audio':str(stem.with_suffix('.wav')),'packet':packet,'result':result,'alignment_error_s':error}
    trials.append(trial)
    if error<=.16 or not result['quality_screen_pass']:break
    alignment=result['alignment'];duration=result['audio']['duration_s']
   row={'text':text,'condition':name,'plan':plan,'baseline':base_eval,'trials':trials,
        'synchronized':trials[-1]['alignment_error_s']<=.16}
   records.append(row);(d/'RESULTS.json').write_text(json.dumps(records,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
   print(index,name,'synchronized',row['synchronized'],'slope',trials[-1]['result']['acoustics']['final_slope_semitones_per_s'],'WER',result['wer'],flush=True)
if __name__=='__main__':main()
