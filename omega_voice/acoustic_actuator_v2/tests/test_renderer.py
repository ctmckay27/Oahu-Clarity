import numpy as np
from omega_voice.acoustic_actuator_v2.renderer import _apply_expanded
def test_expanded_controls_change_waveform():
 sr=24000;y=np.sin(2*np.pi*180*np.arange(sr)/sr).astype("float32")*.1
 a,_=_apply_expanded(y,sr,{"onset_energy":.47,"consonant_energy":.47,"finality":.48,"timing_flex":.05})
 b,_=_apply_expanded(y,sr,{"onset_energy":.57,"consonant_energy":.63,"finality":.82,"timing_flex":.19})
 assert len(a)!=len(b) or not np.allclose(a[:min(len(a),len(b))],b[:min(len(a),len(b))])
 assert np.max(np.abs(a))<1 and np.max(np.abs(b))<1
