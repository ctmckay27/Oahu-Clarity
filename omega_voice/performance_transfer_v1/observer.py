"""Low-level acoustic observer for performance-transfer diagnostics.

This observer measures consequences; it does not define good acting and does not
score Mari perceptual authenticity.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import soundfile as sf

def _frames(y,sr,win_s=.04,hop_s=.01):
    win=max(32,round(win_s*sr));hop=max(16,round(hop_s*sr))
    if len(y)<win:return []
    return [y[i:i+win] for i in range(0,len(y)-win+1,hop)]

def _f0(frame,sr,lo=70.0,hi=400.0):
    x=frame.astype(np.float64)-float(np.mean(frame))
    rms=float(np.sqrt(np.mean(x*x)))
    if rms<1e-4:return math.nan
    x*=np.hanning(len(x))
    ac=np.correlate(x,x,mode="full")[len(x)-1:]
    if ac[0]<=1e-12:return math.nan
    lo_lag=max(1,int(sr/hi));hi_lag=min(len(ac)-1,int(sr/lo))
    if hi_lag<=lo_lag:return math.nan
    seg=ac[lo_lag:hi_lag+1];j=int(np.argmax(seg))+lo_lag
    if ac[j]/ac[0]<.25:return math.nan
    return sr/j

def acoustic_summary(path: str|Path):
    y,sr=sf.read(path,dtype="float32",always_2d=True)
    if y.shape[1]!=1: y=y.mean(1,keepdims=True)
    y=y[:,0]
    frames=_frames(y,sr)
    rms=np.asarray([float(np.sqrt(np.mean(f*f))) for f in frames],dtype=float)
    f0=np.asarray([_f0(f,sr) for f in frames],dtype=float)
    voiced=f0[np.isfinite(f0)]
    def q(a,p):return None if not len(a) else float(np.quantile(a,p))
    return {
        "sample_rate_hz":sr,
        "duration_s":len(y)/sr,
        "peak":float(np.max(np.abs(y))) if len(y) else 0.0,
        "rms_mean":float(np.mean(rms)) if len(rms) else 0.0,
        "rms_std":float(np.std(rms)) if len(rms) else 0.0,
        "rms_p10":q(rms,.10),"rms_p90":q(rms,.90),
        "energy_dynamic_range":None if not len(rms) else float(q(rms,.90)-q(rms,.10)),
        "voiced_fraction":len(voiced)/max(1,len(f0)),
        "f0_median_hz":q(voiced,.50),
        "f0_std_hz":None if not len(voiced) else float(np.std(voiced)),
        "f0_p10_hz":q(voiced,.10),"f0_p90_hz":q(voiced,.90),
    }

def transfer_observation(baseline,conditioned,donor):
    b=acoustic_summary(baseline);c=acoustic_summary(conditioned);d=acoustic_summary(donor)
    def delta(k):
        return None if b[k] is None or c[k] is None else c[k]-b[k]
    return {
        "baseline":b,"conditioned":c,"donor":d,
        "conditioned_minus_baseline":{
            "duration_s":delta("duration_s"),
            "rms_std":delta("rms_std"),
            "energy_dynamic_range":delta("energy_dynamic_range"),
            "voiced_fraction":delta("voiced_fraction"),
            "f0_median_hz":delta("f0_median_hz"),
            "f0_std_hz":delta("f0_std_hz"),
        },
        "interpretation_limit":"These measurements prove acoustic consequence only. They do not establish that donor performance was faithfully transferred or that acting improved.",
    }
