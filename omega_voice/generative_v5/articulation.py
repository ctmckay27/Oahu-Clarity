"""Bounded consonant-energy realization of source-grounded precision.

This is one acoustic correlate of clearer articulation, not an articulatory
tract simulation or evidence of perceived character. Existing unvoiced detail
is modulated; no consonant, noise, hesitation, or timing is synthesized.
"""
import pathlib,json,shutil
import numpy as np,librosa,soundfile as sf
from scipy.signal import lfilter
from ..causal_v4.runtime import compile_scene,verify_plan,digest
from ..causal_v4.renderer import sha,inspect_audio

def analyze(y,sr):
 hop=120;nfft=1024
 X=librosa.stft(y,n_fft=nfft,hop_length=hop);power=np.abs(X)**2
 _,voiced,prob=librosa.pyin(y,sr=sr,fmin=70,fmax=500,frame_length=2048,hop_length=hop)
 frequency=librosa.fft_frequencies(sr=sr,n_fft=nfft)
 high=frequency>=1500
 energy=power.sum(0);active=energy>max(1e-6,float(energy.max())*.001)
 fraction=power[high].sum(0)/(energy+1e-15)
 # This is deliberately reported as an unvoiced-consonant acoustic proxy.
 # Strong harmonic vowels, silence, and low-energy background are excluded.
 consonant=active & (~voiced) & (fraction>.45) & (np.nan_to_num(prob)<.3)
 vowel=active & voiced & (np.nan_to_num(prob)>.6)
 return X,frequency,consonant,vowel,hop,nfft

def realize(source,out,plan):
 source=pathlib.Path(source);out=pathlib.Path(out)
 if out.exists():raise FileExistsError(out)
 if not verify_plan(plan) or plan.get('realization_timeline',{}).get('source_audio_sha256')!=sha(source):raise ValueError('verified actual word clock required')
 y,sr=sf.read(source,dtype='float64')
 if y.ndim!=1 or sr!=24000:raise ValueError('24k mono required')
 neutral=compile_scene(plan['text'],timeline=plan['realization_timeline'],policy=plan.get('temporal_policy'))
 deltas=np.array([a['controls']['precision']-b['controls']['precision'] for a,b in zip(plan['trajectory'],neutral['trajectory'])])
 receipt={'role':'consonant-energy diagnostic; no full-articulation or character claim','input_sha256':sha(source),
   'plan_hash':plan['plan_hash'],'implementation_sha256':sha(pathlib.Path(__file__)),
   'mechanism':'existing unvoiced consonant prominence from causal precision; bounded to plus/minus 3dB',
   'research_basis':['https://pubmed.ncbi.nlm.nih.gov/14759028/','https://www.ee.iitb.ac.in/~spilab/papers/2007/paper_arjayan_ica2007.pdf'],
   'hypothesis_boundary':'clear-speech correlates motivate this channel; irritation/character perception must be independently qualified'}
 if np.max(np.abs(deltas))<1e-12:
  shutil.copyfile(source,out);receipt.update(zero_control_exact_parity=True,audio_sha256=sha(out))
 else:
  X,f,consonant,vowel,hop,nfft=analyze(y,sr)
  if consonant.sum()<3 or vowel.sum()<3:raise ValueError('insufficient observed consonant/vowel regions; no substitute actuator')
  times=np.arange(X.shape[1])*hop/sr
  starts=np.array([w['start'] for w in plan['realization_timeline']['words']]);window_start=times-nfft/(2*sr)
  indices=np.clip(np.searchsorted(starts,window_start,side='right')-1,0,len(starts)-1)
  controls=deltas[indices];controls[window_start<starts[0]]=0
  alpha=1-np.exp(-hop/(sr*.025));controls=lfilter([alpha],[1,-(1-alpha)],controls)
  db=np.clip(8*controls,-3,3)*consonant
  # Frequency taper avoids a hard crossover. It cannot create new excitation.
  band=np.clip((f-1000)/1000,0,1)[:,None];gain=10**(band*db[None,:]/20)
  z=librosa.istft(X*gain,hop_length=hop,length=len(y))
  sample_times=np.arange(len(y))/sr
  source_indices=np.clip(np.searchsorted(starts,sample_times,side='right')-1,0,len(starts)-1)
  active=deltas[source_indices]!=0;active[sample_times<starts[0]]=False
  z=np.where(active,z,y)
  if not np.isfinite(z).all() or np.max(np.abs(z))>=.98:raise ValueError('articulation output exceeds quality bound')
  sf.write(out,z,sr,subtype='PCM_16')
  Z=librosa.stft(z,n_fft=nfft,hop_length=hop);high=f>=1500
  def ratio(A):return float(10*np.log10((np.mean(np.abs(A[high][:,consonant])**2)+1e-15)/(np.mean(np.abs(A[:,vowel])**2)+1e-15)))
  receipt.update(consonant_proxy_frames=int(consonant.sum()),vowel_proxy_frames=int(vowel.sum()),
    consonant_to_vowel_ratio_db_before=ratio(X),consonant_to_vowel_ratio_db_after=ratio(Z),
    maximum_requested_consonant_gain_db=float(db.max()),minimum_requested_consonant_gain_db=float(db.min()),
    duration_preserved=len(z)==len(y),control_hash=digest(db.tolist()),audio=inspect_audio(out)[2])
 out.with_suffix('.articulation.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt
