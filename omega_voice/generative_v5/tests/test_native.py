import json,struct
import numpy as np
import pytest
from omega_voice.generative_v5.native import read_sequence,write_trajectory
from omega_voice.generative_v5.trajectory import compile_finality,boundaries
from omega_voice.causal_v4.runtime import compile_scene

def alignment(text):
 from omega_voice.causal_v4.evaluate import normalized
 return {'exact_words':True,'source_sha256':'diagnostic','words':[
  {'word':w,'start':i*.3,'end':i*.3+.25} for i,w in enumerate(normalized(text))]}

def test_truncated_capture_is_not_evidence(tmp_path):
 p=tmp_path/'x.qseq';p.write_bytes(struct.pack('<3I',0x3151534d,29,2048)+np.zeros(29*2048-3,dtype='<f4').tobytes())
 with pytest.raises(ValueError,match='truncated'):read_sequence(p)

def test_trajectory_envelope(tmp_path):
 p=tmp_path/'x';bank=np.zeros((1,29,2048));weight=np.zeros((20,1))
 for bad in [float('nan'),float('inf'),2.01]:
  weight[2,0]=bad
  with pytest.raises(ValueError):write_trajectory(p,bank,weight)
 weight[2,0]=0;bank[0,22,3]=101
 with pytest.raises(ValueError):write_trajectory(p,bank,weight)

def test_irrelevant_metadata_has_identical_controls():
 text='The key is here. I checked.';a=alignment(text)
 first=compile_scene(text,{'metadata':{'wall_color':'blue'}})
 second=compile_scene(text,{'metadata':{'wall_color':'green'}})
 x,_=compile_finality(first,a,2);y,_=compile_finality(second,a,2)
 assert np.array_equal(x,y) and not x.any()

def test_local_knowledge_overrides_prior_global_uncertainty():
 text='It could be here. I found it.';a=alignment(text)
 events=[{'id':'uncertain','at_word':0,'kind':'set','values':{'knowledge_state.certainty':.15},'source':{'text':'Location is unverified.'}},
         {'id':'found','at_word':4,'kind':'knowledge','status':'known','confidence':.98,'proposition':'location','source':{'text':'Mari sees the key.'}}]
 p=compile_scene(text,{'events':events});x,packet=compile_finality(p,a,2.5)
 assert [s['word'] for s in packet['segments']]==[3,6]
 assert packet['segments'][0]['strength']<0<packet['segments'][1]['strength']
 assert packet['segments'][1]['start_s']>=a['words'][4]['start']

def test_prior_turn_knowledge_reaches_next_native_trajectory():
 p=compile_scene('I found it.',{'events':[{'id':'k','at_word':0,'kind':'knowledge','status':'known','confidence':.98,'proposition':'location','source':{'text':'Mari sees the key.'}}]})
 text='It is here.';nextp=compile_scene(text,prior=p['final_state'])
 x,_=compile_finality(nextp,alignment(text),1)
 assert x.max()>0 and nextp['final_state']['turn']==2

def test_alignment_mismatch_fails_closed():
 text='It is here.';a=alignment(text);a['words'][1]['word']='was'
 with pytest.raises(ValueError):compile_finality(compile_scene(text),a,1)
