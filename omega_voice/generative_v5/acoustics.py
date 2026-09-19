"""Acoustic observations, kept separate from inferred intention/personality."""
import numpy as np,librosa,soundfile as sf
def measure(path):
 y,sr=sf.read(path,dtype='float32');y=y.mean(1) if y.ndim==2 else y
 x=librosa.resample(y,orig_sr=sr,target_sr=16000)
 f0,voiced,prob=librosa.pyin(x,sr=16000,fmin=70,fmax=500,frame_length=1024,hop_length=160)
 rms=librosa.feature.rms(y=x,frame_length=1024,hop_length=160)[0][:len(f0)]
 good=np.flatnonzero(np.isfinite(f0)&(rms>max(.001,float(rms.max())*.04)))
 if len(good)<8:raise ValueError('insufficient voiced frames')
 t=good*.01;st=12*np.log2(f0[good]/100);end=t[-1]
 final=(t>=end-.55);pre=(t>=end-1.05)&(t<end-.55)
 slope=float(np.polyfit(t[final],st[final],1)[0]) if final.sum()>=4 else None
 delta=float(np.median(st[final])-np.median(st[pre])) if pre.sum()>=4 else None
 return {'duration_s':len(y)/sr,'voiced_end_s':float(end),'median_f0_hz':float(np.median(f0[good])),'final_slope_semitones_per_s':slope,'final_vs_pre_semitones':delta,'voiced_range_semitones':float(np.percentile(st,90)-np.percentile(st,10)),'voiced_fraction':float(len(good)/len(f0)),'interpretation':'acoustic observations only; pitch movement is not itself certainty or character'}
