"""Native onset counterfactuals on text excluded from calibration."""
import argparse,json,pathlib
import numpy as np
from .native import render,write_trajectory
from .attack_calibration import attack_measure
from .acoustics import measure
from ..causal_v4.evaluate import Evaluator
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/attack_heldout';d.mkdir(exist_ok=True)
 bank=np.load(r/'continuation/attack_calibration/attack_bank.npz')['directions'];ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');rows=[]
 for i,text in enumerate(['All right. I will check the lock.','I know you have had a difficult day.']):
  for name,sign in [('base',0),('firm',1),('soft',-1)]:
   out=d/f'{i}_{name}.wav';weights=np.zeros((100,1),np.float32)
   for frame in range(4):weights[frame,0]=sign*.7*max(0,1-(frame+1)*.08/.32)
   trajectory=d/f'{i}_{name}.mtraj';write_trajectory(trajectory,bank,weights,onset=np.array([sign*.7]))
   if not out.exists():render(r,text,out,seed=92600+i,trajectory=trajectory)
   result=ev.evaluate(out,text);result.update(attack=attack_measure(out),acoustics=measure(out))
   rows.append({'text':text,'condition':name,'result':result,'audio':str(out)})
   (d/'RESULTS.json').write_text(json.dumps(rows,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
   print(i,name,result['attack']['initial_to_body_db'],'speaker',result['speaker_similarity'],'WER',result['wer'],flush=True)
if __name__=='__main__':main()
