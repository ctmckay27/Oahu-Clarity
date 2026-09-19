"""Identify an onset-pressure actuator from isolated same-waveform references.

Amplitude-envelope references provide a measurable calibration target, not a
claim to have measured glottal physiology. Native free-generation must confirm
the effect and identity/naturalness before this actuator can be admitted.
"""
import argparse,json,pathlib
import numpy as np,soundfile as sf,librosa
from .reference_calibration import encode_reference
from .native import render,read_sequence
from ..causal_v4.renderer import sha

def attack_measure(path):
 y,sr=sf.read(path,dtype='float32');y=y.mean(1) if y.ndim==2 else y
 rms=librosa.feature.rms(y=y,frame_length=480,hop_length=120)[0]
 active=np.flatnonzero(rms>max(.001,float(rms.max())*.12))
 if not len(active):raise ValueError('no speech onset')
 start=float(active[0]*120/sr)
 def level(a,b):
  x=y[int(a*sr):int(b*sr)];return float(np.sqrt(np.mean(x*x)+1e-12))
 early=level(start,start+.12);body=level(start+.12,start+.36)
 return {'onset_s':start,'initial_to_body_db':float(20*np.log10(early/body)),
         'initial_rms':early,'body_rms':body,'duration_s':len(y)/sr,
         'scope':'acoustic onset envelope only, not a physiological tension estimate'}

def reference(source,out,sign):
 y,sr=sf.read(source,dtype='float32');m=attack_measure(source);t=np.arange(len(y))/sr
 elapsed=np.maximum(0,t-m['onset_s']);envelope=1+sign*.30*np.exp(-elapsed/.12)
 envelope[t<m['onset_s']]=1
 z=y*envelope;z*=np.sqrt(np.sum(y*y)/np.sum(z*z));sf.write(out,z,sr,subtype='PCM_16')
 return {'source_sha256':sha(source),'audio_sha256':sha(out),'sign':sign,'acoustics':attack_measure(out),'purpose':'isolated diagnostic onset-envelope reference; not selected voice asset'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();root=pathlib.Path(a.root);d=root/'continuation/attack_calibration';d.mkdir(exist_ok=True)
 texts=['The package is ready.','You left it by the window.','The meeting starts at seven.','That is the correct address.']
 rows=[];deltas=[];means=[]
 for i,text in enumerate(texts):
  source=root/f'continuation/calibration/pair_{i}_statement.wav';pair={}
  for label,sign in [('firm',1),('soft',-1)]:
   p=d/f'{i}_{label}';ref=p.with_suffix('.reference.wav');codes=p.with_suffix('.codes');rec=reference(source,ref,sign)
   if not codes.exists():encode_reference(root,ref,text,codes)
   replay=p.with_suffix('.replay.wav')
   if not replay.exists():render(root,text,replay,90500+i,capture=True,teacher_codes=codes)
   seq=read_sequence(replay.with_suffix('.qseq'));end=min(len(seq),max(1,int((rec['acoustics']['onset_s']+.32)/.08)))
   pair[label]={'reference':rec,'replay':attack_measure(replay),'activation':seq[:end].mean(0)}
  contrast=pair['firm']['replay']['initial_to_body_db']-pair['soft']['replay']['initial_to_body_db'];accepted=contrast>.5
  if accepted:deltas.append(pair['firm']['activation']-pair['soft']['activation']);means.append((pair['firm']['activation']+pair['soft']['activation'])/2)
  rows.append({'pair':i,'replayed_firm_minus_soft_db':contrast,'accepted':accepted,**{k:{n:v for n,v in item.items() if n!='activation'} for k,item in pair.items()}})
  (d/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n');print('attack contrast',i,contrast,flush=True)
 if len(deltas)<3:raise RuntimeError('onset reference contrast did not survive codec; do not invent actuator')
 v=np.mean(deltas,0);n=np.mean(means,0);v-=v.mean(1,keepdims=True);v-=n*((v*n).sum(1)/(n*n).sum(1))[:,None];v[:21]=0;v[26:]=0
 np.savez(d/'attack_bank.npz',directions=v[None],paired_deltas=np.stack(deltas))
 print('ATTACK_BANK_READY',np.linalg.norm(v,axis=1).tolist(),flush=True)
if __name__=='__main__':main()
