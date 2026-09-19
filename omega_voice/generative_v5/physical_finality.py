"""Measured-word contour realization after native timing feedback oscillated.

This realizes an acoustic consequence of commitment, not a standalone test of
cognitive presence. Pitch changes remain within a causal word boundary window.
"""
import json,pathlib,shutil
import numpy as np,soundfile as sf,parselmouth
from parselmouth.praat import call
from ..causal_v4.runtime import verify_plan
from ..causal_v4.evaluate import normalized
from ..causal_v4.renderer import sha,inspect_audio
from .trajectory import compile_finality

def realize(source,out,plan):
 source=pathlib.Path(source);out=pathlib.Path(out)
 if out.exists():raise FileExistsError(out)
 if not verify_plan(plan):raise ValueError('unverified plan')
 clock=plan.get('realization_timeline')
 if not clock or clock['source_audio_sha256']!=sha(source):raise ValueError('actual source word clock required')
 alignment={'exact_words':True,'source_sha256':sha(source),'words':[dict(t,word=w) for t,w in zip(clock['words'],normalized(plan['text']))]}
 _,packet=compile_finality(plan,alignment,clock['duration_s'])
 record={'role':'sample-aligned physical contour diagnostic','plan_hash':plan['plan_hash'],'input_sha256':sha(source),
  'implementation_sha256':sha(pathlib.Path(__file__)),'method':'PSOLA local zero-mean contour; blend remains inside causal target window',
  'full_completion':False,'segments':packet['segments']}
 if not any(abs(s['strength'])>1e-8 for s in packet['segments']):
  shutil.copyfile(source,out);record.update(zero_control_exact_parity=True,audio_sha256=sha(out))
  out.with_suffix('.finality.json').write_text(json.dumps(record,indent=2)+'\n');return record
 sound=parselmouth.Sound(str(source));y,sr=sf.read(source,dtype='float64')
 if sr!=24000 or y.ndim!=1:raise ValueError('24k mono carrier required')
 manipulation=call(sound,'To Manipulation',.01,70,500);tier=call(manipulation,'Extract pitch tier')
 points=[(call(tier,'Get time from index',i),call(tier,'Get value at index',i)) for i in range(1,call(tier,'Get number of points')+1)]
 windows=[]
 for segment in packet['segments']:
  if abs(segment['strength'])<1e-8:continue
  word=segment['word']
  # A long gap must not select phonation from the following word. Final tails
  # may extend past an ASR token end; intermediate release has a strict cap.
  upper=min(clock['words'][word+1]['start']-.02,clock['words'][word]['end']+.08) if word+1<len(clock['words']) else clock['duration_s']
  times=[t for t,hz in points if segment['start_s']<=t<=upper]
  if len(times)<3:raise ValueError('boundary lacks sufficient voiced evidence')
  end=max(times);start=max(segment['start_s'],end-.55)
  if end-start<.08:raise ValueError('voiced boundary too short for controlled contour')
  windows.append(dict(segment,acoustic_start_s=start,acoustic_end_s=end,release_end_s=min(upper,end+.04)))
 call(tier,'Remove points between',sound.xmin,sound.xmax)
 for t,hz in points:
  change=0
  for w in windows:
   start,end=w['acoustic_start_s'],w['acoustic_end_s']
   if start<=t<=end:
    u=(t-start)/(end-start);change-=w['strength']/.7*(9*u*u-6*u)
  call(tier,'Add point',t,hz*2**(change/12))
 call([tier,manipulation],'Replace pitch tier');result=call(manipulation,'Get resynthesis (overlap-add)')
 z=np.asarray(result.values[0],dtype=np.float64)
 if len(z)!=len(y):raise ValueError('PSOLA changed duration; refusing output')
 t=np.arange(len(y))/sr;mix=np.zeros(len(y))
 for w in windows:
  start,end=w['acoustic_start_s'],w['release_end_s'];fade=min(.04,(end-start)/4)
  envelope=np.minimum(np.clip((t-start)/fade,0,1),np.clip((end-t)/fade,0,1));mix=np.maximum(mix,envelope)
 output=np.where(mix>0,y*(1-mix)+z*mix,y)
 if not np.isfinite(output).all() or np.max(np.abs(output))>=.98:raise ValueError('contour output outside quality bound')
 sf.write(out,output,sr,subtype='PCM_16');_,_,info=inspect_audio(out)
 record.update(audio=info,windows=windows,duration_unchanged=True,unmodified_samples_outside_windows=True)
 out.with_suffix('.finality.json').write_text(json.dumps(record,indent=2)+'\n');return record
