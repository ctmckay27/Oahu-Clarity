"""Time consequence of scoped prominence, calibrated on fixed matched text.

Unlike global rate/pauses, only the explicitly contrasted lexical unit changes
duration. All later original samples shift intact; earlier speech stays exact.
"""
import pathlib,json,math,copy
import numpy as np,soundfile as sf,parselmouth
from parselmouth.praat import call
from ..causal_v4.runtime import verify_plan
from ..causal_v4.renderer import sha

def realize(source,out,plan,calibration):
 source=pathlib.Path(source);out=pathlib.Path(out);calibration=pathlib.Path(calibration)
 if out.exists():raise FileExistsError(out)
 if not verify_plan(plan) or plan.get('temporal_policy')!='contrast_focus_v7':raise ValueError('verified semantic contrast required')
 clock=plan.get('realization_timeline')
 if not clock or clock['source_audio_sha256']!=sha(source):raise ValueError('actual source clock required')
 if sha(calibration)!='cc38d2a2d23689ab735a92e5076e8dbd7a11466dac83fd8c66e03de3b8c888c3':raise ValueError('calibration bytes differ from selected matched-reference evidence')
 evidence=json.loads(calibration.read_text());g=evidence['geometry']['duration_ratio']
 if evidence['pairs']!=16 or evidence['admitted_clocks']!=16:raise ValueError('unselected matched calibration')
 # Fixed reference-strength mapping, declared before observing Mari outputs.
 reference_strength=.55;median=g['median'];cap=g['q75'];windows=[]
 y,sr=sf.read(source,dtype='int16')
 if sr!=24000 or y.ndim!=1:raise ValueError('24k mono required')
 for knot in plan['trajectory'][:-1]:
  strength=knot['controls']['emphasis'];i=knot['at_word']
  if strength<=0:continue
  t=clock['words'][i];start=round(t['start']*sr);end=round(t['end']*sr);ratio=min(cap,math.exp(math.log(median)*strength/reference_strength));extra=round((end-start)*(ratio-1))
  if end-start<round(.08*sr):raise ValueError('contrast word too short for bounded duration control')
  windows.append({'word':i,'start_sample':start,'end_sample':end,'extra_samples':extra,'duration_ratio':ratio,'strength':strength})
 parts=[];cursor=0;total_extra=0;mapping=[]
 for w in windows:
  a,b=w['start_sample'],w['end_sample'];extra=w['extra_samples'];word=y[a:b].astype(float)/32768
  # Local context supports pitch extraction; no context sample is replaced.
  margin=round(.08*sr);lo=max(0,a-margin);hi=min(len(y),b+margin);sound=parselmouth.Sound(y[lo:hi].astype(float)/32768,sampling_frequency=sr)
  manipulation=call(sound,'To Manipulation',.005,70,500);duration=call(manipulation,'Extract duration tier');call(duration,'Remove points between',sound.xmin,sound.xmax)
  start=(a-lo)/sr;end=(b-lo)/sr;length=end-start;ramp=min(.025,length*.1);height=1+(extra/sr)/(length-ramp)
  knots=[(0.,1.),(start,1.),(start+ramp,height),(end-ramp,height),(end,1.),(sound.xmax,1.)]
  for t,v in sorted(set(knots)):call(duration,'Add point',t,v)
  call([duration,manipulation],'Replace duration tier');z=call(manipulation,'Get resynthesis (overlap-add)').values[0];target_length=b-a+extra;offset=a-lo;segment=z[offset:offset+target_length]
  if len(segment)!=target_length:raise ValueError('duration resynthesis lost requested lexical unit')
  fade=min(round(.012*sr),len(word)//8);alpha=np.linspace(0,1,fade,endpoint=False)
  # These blends stay inside the changed word; no past/future source sample
  # is repainted. Start/end preserve the existing captured boundary phase.
  segment[:fade]=word[:fade]*(1-alpha)+segment[:fade]*alpha
  segment[-fade:]=segment[-fade:]*(1-alpha)+word[-fade:]*alpha
  if not np.isfinite(segment).all()or np.max(np.abs(segment))>=.98:raise ValueError('duration output outside quality bounds')
  pcm=np.rint(segment*32768).astype(np.int16);parts.extend([y[cursor:a],pcm]);mapping.append(dict(w,delivered_start_sample=a+total_extra,delivered_end_sample=b+total_extra+extra));total_extra+=extra;cursor=b
 parts.append(y[cursor:]);out.parent.mkdir(exist_ok=True,parents=True);sf.write(out,np.concatenate(parts),sr,subtype='PCM_16')
 mapped=copy.deepcopy(clock)
 for i,t in enumerate(mapped['words']):
  before=sum(w['extra_samples']for w in windows if w['word']<i)/sr;own=sum(w['extra_samples']for w in windows if w['word']==i)/sr;t['start']+=before;t['end']+=before+own
 mapped['duration_s']+=total_extra/sr;mapped['source_audio_sha256']=sha(out)
 receipt={'role':'scoped contrast duration diagnostic','input_sha256':sha(source),'output_sha256':sha(out),'implementation_sha256':sha(__file__),'calibration_sha256':sha(calibration),'calibration_source':'fixed matched synthetic speech, two texts/all four voices; not Mari identity or human physiology','reference_strength':reference_strength,'median_ratio':median,'cap_ratio':cap,'windows':mapping,'added_samples':total_extra,'mapped_timeline':mapped,'untouched_source_samples_preserved_in_order':True,'random_pauses_added':False,'full_completion':False};out.with_suffix('.duration.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
