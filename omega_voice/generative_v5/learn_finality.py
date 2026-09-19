"""Identify a local boundary-contour direction from measured paired speech.

The acoustic label is falling vs rising boundary contour. Mapping certainty to
this physical channel is a separate Mari behavioral hypothesis, never a claim
that question punctuation itself establishes thought or certainty.
"""
import argparse,json,pathlib,numpy as np
from .native import read_sequence,write_trajectory,render
from .acoustics import measure
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/calibration'
 rows=[];deltas=[];neutral=[]
 for i in range(4):
  pair={}
  for kind in ['statement','question']:
   p=d/f'pair_{i}_{kind}.wav';m=measure(p);seq=read_sequence(p.with_suffix('.qseq'))
   # The last half-second of voiced output, excluding decoder-tail silence.
   first=max(0,int((m['voiced_end_s']-.55)/.08));last=min(len(seq),int(m['voiced_end_s']/.08)+1)
   pair[kind]={'acoustics':m,'frame_window':[first,last],'activation':seq[first:last].mean(0)}
  contrast=pair['statement']['acoustics']['final_slope_semitones_per_s']-pair['question']['acoustics']['final_slope_semitones_per_s']
  accepted=contrast < -2
  if accepted:
   deltas.append(pair['statement']['activation']-pair['question']['activation']);neutral.append((pair['statement']['activation']+pair['question']['activation'])/2)
  rows.append({'pair':i,'fall_minus_rise_slope':contrast,'accepted_physical_contrast':accepted,**{k:{n:v for n,v in p.items() if n!='activation'} for k,p in pair.items()}})
 report={'method':'paired same-voice same-words punctuation contrasts filtered by observed boundary F0; conditional causal interpretation remains separate','pairs':rows,'accepted':len(deltas),'status':'MEASURED_CONTRASTS'}
 (d/'FINALITY_CALIBRATION.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2),flush=True)
 if len(deltas)<2:raise RuntimeError('insufficient measured finality contrasts; do not invent calibrated axis')
 v=np.mean(deltas,axis=0);n=np.mean(neutral,axis=0);v-=v.mean(axis=1,keepdims=True)
 v-=n*((v*n).sum(1)/(n*n).sum(1))[:,None]
 # Act inside the engine's inspected late representation layers. This is a
 # hypothesis under test, not inherited validation from shipped emotion vectors.
 v[:21]=0;v[26:]=0
 np.savez(d/'finality_bank.npz',directions=v[None],paired_deltas=np.stack(deltas))
 report['layer_norms']=np.linalg.norm(v,axis=1).tolist();(d/'FINALITY_CALIBRATION.json').write_text(json.dumps(report,indent=2)+'\n')
 text='I understand what happened. Give me a moment to check the sequence.'
 # Original failed counterfactual: same words/seed and identity, changed local
 # finality. Direction only enters the last ~0.6 s of each phrase.
 weights=np.zeros((100,1),np.float32);weights[17:25,0]=.7;weights[42:52,0]=.7
 for name,sign in [('commit',1),('open',-1)]:
  p=d/(name+'.mtraj');write_trajectory(p,v[None],weights*sign)
  out=render(r,text,d/(name+'.wav'),trajectory=p)
  print(name,out['audio'],flush=True)
if __name__=='__main__':main()
