"""Causal local acoustic realization where latent/reference control failed.

This is a bounded acoustic approximation, not a measured glottal model. Tension
changes voiced harmonic slope; attack changes only an observed phonation onset.
No noise, disfluency, formant shift, pitch shift, or arbitrary timing is added.
Quality and perceptual qualification remain separate from signal causality.
"""
import json,pathlib,shutil
import numpy as np
import librosa,soundfile as sf
from scipy.signal import lfilter
from ..causal_v4.runtime import compile_scene,verify_plan,digest
from ..causal_v4.renderer import sha,inspect_audio

def control_curves(plan,times):
 if not verify_plan(plan) or 'realization_timeline' not in plan:raise ValueError('verified actual word clock required')
 timeline=plan['realization_timeline'];neutral=compile_scene(plan['text'],timeline=timeline)
 starts=np.array([w['start'] for w in timeline['words']]);indices=np.clip(np.searchsorted(starts,times,side='right')-1,0,len(starts)-1)
 curves={}
 for name in ['phonatory_tension','attack_softness','gain_db']:
  deltas=np.array([a['controls'][name]-b['controls'][name] for a,b in zip(plan['trajectory'],neutral['trajectory'])])
  target=deltas[indices];target[times<starts[0]]=0
  # Causal relaxation: a future state must never alter an earlier sample.
  alpha=1-np.exp(-float(times[1]-times[0])/.03) if len(times)>1 else 1
  curves[name]=lfilter([alpha],[1,-(1-alpha)],target)
 return curves

def realize(source,out,plan):
 source=pathlib.Path(source);out=pathlib.Path(out)
 if out.exists():raise FileExistsError(out)
 if plan.get('realization_timeline',{}).get('source_audio_sha256')!=sha(source):raise ValueError('timeline belongs to another waveform')
 y,sr=sf.read(source,dtype='float64')
 if y.ndim!=1 or sr!=24000:raise ValueError('24k mono carrier required')
 hop=120;nfft=2048;times=np.arange(1+len(y)//hop)*hop/sr;curves=control_curves(plan,times)
 record={'role':'physical mechanism diagnostic; not completed Mari performance',
         'input_sha256':sha(source),'plan_hash':plan['plan_hash'],'method':'causal voiced harmonic slope and observed-onset envelope; no style prompt',
         'scope':'acoustic approximation; no physiological measurement or character-presence claim',
         'unrealized':['articulation gestures','nonlexical behavior','complete character specificity']}
 if all(not np.any(c) for c in curves.values()):
  shutil.copyfile(source,out);record.update(zero_control_exact_parity=True,audio_sha256=sha(out))
  out.with_suffix('.physical.json').write_text(json.dumps(record,indent=2)+'\n');return record
 X=librosa.stft(y,n_fft=nfft,hop_length=hop)
 f0,voiced,_=librosa.pyin(y,sr=sr,fmin=70,fmax=500,frame_length=nfft,hop_length=hop)
 if X.shape[1]!=len(times) or len(f0)!=len(times):raise RuntimeError('physical analysis frame mismatch')
 freq=librosa.fft_frequencies(sr=sr,n_fft=nfft)[:,None];f0safe=np.nan_to_num(f0,nan=200)[None,:]
 harmonics=np.rint(freq/f0safe);distance=np.abs(freq-harmonics*f0safe)
 harmonic_mask=np.exp(-.5*(distance/(2*sr/nfft))**2)*(harmonics>=1)*(freq<=5000)*voiced[None,:]
 # A centered analysis window overlaps earlier samples. Feed its control from
 # the window's beginning so a later event cannot leak backwards through ISTFT.
 spectral_curves=control_curves(plan,times-nfft/(2*sr))
 slope=np.clip(4*spectral_curves['phonatory_tension'],-1.5,1.5)
 octaves=np.clip(np.log2(np.maximum(freq,70)/500),-2,3)
 gain=10**(harmonic_mask*octaves*slope[None,:]/20)
 # Keep frame energy independent of the spectral tilt. Projection is explicit.
 ratio=np.sqrt((np.sum(np.abs(X)**2,axis=0)+1e-15)/(np.sum(np.abs(X*gain)**2,axis=0)+1e-15))
 Z=X*gain*ratio[None,:]
 z=librosa.istft(Z,hop_length=hop,length=len(y))
 rms=librosa.feature.rms(y=y,frame_length=480,hop_length=hop)[0]
 active=rms>max(.001,float(rms.max())*.12);onsets=[];last_active=-1000
 for i,a in enumerate(active):
  if a:
   if i-last_active>int(.14*sr/hop):onsets.append(i)
   last_active=i
 sample_times=np.arange(len(y))/sr;envelope=np.ones(len(y));attack_events=[]
 for frame in onsets:
  # Use the active state at the observed onset; never extrapolate a future beat.
  at=frame*hop/sr;delta=float(curves['attack_softness'][min(frame,len(times)-1)])
  if abs(delta)<1e-8:continue
  elapsed=sample_times-at;mask=(elapsed>=0)&(elapsed<.36)
  envelope[mask]*=1-np.clip(delta*.85,-.25,.4)*np.exp(-elapsed[mask]/.10)
  attack_events.append({'time_s':at,'softness_delta':delta})
 indices=np.clip(np.searchsorted(times,sample_times,side='right')-1,0,len(times)-1)
 # Interpolation between the preceding and following control knots anticipates
 # a future event. Hold past values, then smooth forwards at audio sample rate.
 sample_gain=curves['gain_db'][indices];alpha=1-np.exp(-1/(sr*.003))
 sample_gain=lfilter([alpha],[1,-(1-alpha)],sample_gain)
 projection=10**(sample_gain/20)
 z*=envelope*projection
 active_curve=np.maximum.reduce([np.abs(c) for c in curves.values()])
 changed=active_curve[indices]>1e-12
 z=np.where(changed,z,y)
 if not np.isfinite(z).all() or np.max(np.abs(z))>=.98:raise ValueError('physical output safety/quality bound exceeded; no clipping or silent limiting')
 sf.write(out,z,sr,subtype='PCM_16');_,_,info=inspect_audio(out)
 record.update(audio=info,attack_events=attack_events,source_tension_slope_db_octave={'min':float(slope.min()),'max':float(slope.max())},
               realization_controls_hash=digest({k:v.tolist() for k,v in curves.items()}),pitch_and_duration_changed=False,
               spectral_control_window_delay_s=nfft/(2*sr),implementation_sha256=sha(pathlib.Path(__file__)))
 out.with_suffix('.physical.json').write_text(json.dumps(record,indent=2)+'\n');return record
