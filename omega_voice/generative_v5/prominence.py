"""Local vocal consequence of resolved semantic contrast.

Bounded pitch excursion and pressure envelope on the contrasted spoken word.
No global affect, text change, pause, random timing or identity adjustment.
This is an acoustic approximation requiring independent emphasis verification.
"""
import json,pathlib,shutil
import numpy as np,soundfile as sf,parselmouth
from parselmouth.praat import call
from ..causal_v4.runtime import verify_plan
from ..causal_v4.renderer import sha,inspect_audio

def realize(source,out,plan):
 source=pathlib.Path(source);out=pathlib.Path(out)
 if out.exists():raise FileExistsError(out)
 if not verify_plan(plan) or plan.get('temporal_policy')!='contrast_focus_v7':raise ValueError('resolved contrast policy required')
 clock=plan.get('realization_timeline')
 if not clock or clock['source_audio_sha256']!=sha(source):raise ValueError('verified source word clock required')
 windows=[]
 for knot in plan['trajectory'][:-1]:
  strength=knot['controls']['emphasis'];i=knot['at_word']
  if strength>0:
   t=clock['words'][i];windows.append({'word':i,'text':plan['words'][i],'start':t['start'],'end':t['end'],'strength':strength,'causes':knot['causes']})
 receipt={'role':'semantic contrast prominence diagnostic','plan_hash':plan['plan_hash'],'source_sha256':sha(source),'implementation_sha256':sha(__file__),'windows':windows,'coefficient_status':'bounded engineering hypotheses, not measurements','mechanism':'situated raised pitch excursion and pressure envelope; source samples fixed outside the contrast word','full_completion':False,'perceptual_emphasis_admitted':False}
 if not windows:
  shutil.copyfile(source,out);receipt.update(zero_control_exact_parity=True,output_sha256=sha(out));out.with_suffix('.prominence.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
 y,sr=sf.read(source,dtype='float64')
 if sr!=24000 or y.ndim!=1:raise ValueError('24k mono carrier required')
 sound=parselmouth.Sound(str(source));manipulation=call(sound,'To Manipulation',.01,70,500);tier=call(manipulation,'Extract pitch tier')
 points=[(call(tier,'Get time from index',i),call(tier,'Get value at index',i)) for i in range(1,call(tier,'Get number of points')+1)]
 for w in windows:
  voiced=[t for t,hz in points if w['start']<=t<=w['end']]
  if len(voiced)<3 or max(voiced)-min(voiced)<.06:raise ValueError('contrast lacks voiced evidence for controlled prominence')
 call(tier,'Remove points between',sound.xmin,sound.xmax)
 def shape(t,w):
  u=(t-w['start'])/(w['end']-w['start'])
  return np.sin(np.pi*np.clip(u,0,1))**2*((u>=0)&(u<=1))
 for t,hz in points:
  change=sum(2.5*w['strength']*shape(t,w)for w in windows)
  call(tier,'Add point',t,hz*2**(change/12))
 call([tier,manipulation],'Replace pitch tier');result=call(manipulation,'Get resynthesis (overlap-add)');z=np.asarray(result.values[0],dtype=np.float64)
 if len(z)!=len(y):raise ValueError('prominence changed source clock')
 times=np.arange(len(y))/sr;mix=np.zeros(len(y));gain=np.zeros(len(y))
 for w in windows:
  envelope=shape(times,w);gain+=2.4*w['strength']*envelope;fade=min(.025,(w['end']-w['start'])/4)
  mix=np.maximum(mix,np.minimum(np.clip((times-w['start'])/fade,0,1),np.clip((w['end']-times)/fade,0,1)))
 output=np.where(mix>0,(y*(1-mix)+z*mix)*10**(gain/20),y)
 if not np.isfinite(output).all() or np.max(np.abs(output))>=.98:raise ValueError('prominence exceeds audio quality bound; no hidden limiting')
 sf.write(out,output,sr,subtype='PCM_16');_,_,info=inspect_audio(out);receipt.update(audio=info,output_sha256=sha(out),source_samples_outside_focus_exact=True,duration_unchanged=True)
 out.with_suffix('.prominence.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
