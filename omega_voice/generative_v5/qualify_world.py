"""Full carrier spectral-envelope reconstruction before phonation control.

This diagnostic does not select a replacement speaker or admit a new backend.
The low-dimensional neural reconstruction failed identity; WORLD retains the
measured spectral envelope and aperiodicity as separate source parameters.
"""
import argparse,pathlib,json,time,importlib.metadata
import numpy as np,soundfile as sf,pyworld as pw
from ..causal_v4.renderer import sha
from ..causal_v4.evaluate import Evaluator
from .session import durable_json

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/world_qualification';d.mkdir(exist_ok=True)
 if importlib.metadata.version('pyworld')!='0.3.5':raise ValueError('unqualified WORLD wrapper version')
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 cases=[r/'continuation/calibration/unchanged.wav',r/'continuation/anchor_context/heldout_short_anchor_icl.wav']
 rows=[]
 for i,p in enumerate(cases):
  receipt=json.loads(p.with_suffix('.receipt.json').read_text());text=receipt['text'];x,sr=sf.read(p,dtype='float64')
  if sr!=24000 or x.ndim!=1:raise ValueError('wrong carrier format')
  start=time.monotonic();f0,t=pw.harvest(x,sr,f0_floor=70,f0_ceil=500,frame_period=5.0);f0=pw.stonemask(x,f0,t,sr)
  spectrum=pw.cheaptrick(x,f0,t,sr);aperiodicity=pw.d4c(x,f0,t,sr)
  np.savez_compressed(d/f'{i}_features.npz',f0=f0,t=t,spectrum=spectrum,aperiodicity=aperiodicity)
  y=pw.synthesize(f0,spectrum,aperiodicity,sr,5.0);repeat=pw.synthesize(f0,spectrum,aperiodicity,sr,5.0)
  if len(y)<len(x) or not np.isfinite(y).all() or np.max(np.abs(y))>=.98:raise ValueError('unqualified WORLD output')
  out=d/f'{i}_reconstruction.wav'
  if out.exists():raise FileExistsError(out)
  sf.write(out,y[:len(x)],sr,subtype='PCM_16');q=ev.evaluate(out,text)
  rows.append({'source':str(p),'source_sha256':sha(p),'source_text':text,'audio':str(out),'quality':q,
   'reconstruction_repeat_exact':bool(np.array_equal(y,repeat)),'analysis_frames':len(f0),'spectral_bins':spectrum.shape[1],
   'tail_padding_removed_samples':len(y)-len(x),'elapsed_s':time.monotonic()-start,'feature_intervention':None,'full_completion':False})
  durable_json(d/'RESULTS.json',rows);print(i,q['wer'],q['speaker_similarity'],q['quality_screen_pass'],flush=True)
 durable_json(d/'ASSESSMENT.json',{'all_reconstruction_quality':all(x['quality']['quality_screen_pass'] and x['quality']['wer']==0 for x in rows),
   'reproducible':all(x['reconstruction_repeat_exact'] for x in rows),'full_completion':False,
   'scope':'reconstruction only; phonatory control, naturalness, nonlexical behavior and character unqualified'})
 durable_json(d/'DEPENDENCIES.json',{'pyworld':'0.3.5','module':str(pathlib.Path(pw.__file__)),'module_sha256':sha(pw.__file__),
   'numpy':np.__version__,'implementation_sha256':sha(__file__),'primary_sources':['https://github.com/mmorise/World','https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder']})
if __name__=='__main__':main()
