"""One reference-calibrated local prominence gesture; no prose direction."""
import pathlib,json
import numpy as np
from parselmouth.praat import call
from ..causal_v4.renderer import sha

def load(path,expected_hash):
 p=pathlib.Path(path)
 if not expected_hash or sha(p)!=expected_hash:raise ValueError('unselected gesture calibration')
 c=json.loads(p.read_text())
 for name,key in [('REFERENCES.json','source_sha256'),('PREDECLARED.json','selection_sha256'),('PAIRS.json','pairs_sha256')]:
  if sha(p.parent/name)!=c[key]:raise ValueError('gesture calibration evidence changed')
 phase=np.asarray(c['normalized_time']);curve=np.asarray(c['pitch_residual_delta_st']);level=c['relative_level_delta_dB']
 if c['schema']!='mari-coupled-prominence-calibration/1.0'or not c['no_Mari_fit']or c['observer_training_text_overlap']or c['pairs']<16:raise ValueError('unqualified calibration scope')
 if phase.shape!=(33,)or curve.shape!=(33,)or not np.isfinite(curve).all()or not np.allclose(phase,np.linspace(0,1,33))or max(abs(curve))>6 or abs(curve[0])+abs(curve[-1])>1e-8 or not 0<=level<=3:raise ValueError('gesture outside bounded calibration neighborhood')
 return c

def install_pitch(manipulation,start,end,strength,calibration):
 tier=call(manipulation,'Extract pitch tier');n=call(tier,'Get number of points');points=[(call(tier,'Get time from index',i),call(tier,'Get value at index',i))for i in range(1,n+1)]
 scale=strength/calibration['reference_strength'];phase=calibration['normalized_time'];curve=calibration['pitch_residual_delta_st'];call(tier,'Remove points between',0.,1e6)
 for t,f in points:
  delta=float(np.interp((t-start)/(end-start),phase,curve))*scale if start<=t<=end else 0.
  call(tier,'Add point',t,f*2**(delta/12))
 call([tier,manipulation],'Replace pitch tier');return {'pitch_peak_increment_st':float(max(curve)*scale),'coupled_level_peak_dB':float(calibration['relative_level_delta_dB']*scale),'endpoint_contour_preserved':True}

def apply_level(segment,strength,calibration):
 phase=np.linspace(0,1,len(segment));shape=np.interp(phase,calibration['normalized_time'],calibration['pitch_residual_delta_st']);peak=max(abs(shape))
 if peak<=0:return segment
 activation=np.maximum(0,shape)/peak;db=calibration['relative_level_delta_dB']*strength/calibration['reference_strength']*activation
 return segment*10**(db/20)
