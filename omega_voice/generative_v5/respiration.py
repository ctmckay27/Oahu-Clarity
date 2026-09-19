"""Explicit quiet-inhalation planning under a normalized respiratory budget.

This is a constrained synthetic-body model, not measured human physiology.
No CTC blank is declared a breath. When a known phrase exceeds available air,
an explicit inhalation receives its own time interval at a verified quiet
boundary. No noise or disfluency is injected. Audible breath timbre remains
unimplemented and must not be inferred from this timing mechanism.
"""
import copy,pathlib,re,json
import numpy as np,soundfile as sf
from ..causal_v4.runtime import compile_scene,verify_plan,words,digest
from ..causal_v4.renderer import sha
from .session import durable_json

POLICY='respiratory_budget_v5'
FLOOR=.18

def phrase_boundaries(text):
 tokens=words(text);boundaries=[0]
 for i in range(1,len(tokens)):
  separator=text[tokens[i-1].end():tokens[i].start()]
  if re.search(r'[,.;:!?]',separator):boundaries.append(i)
 return boundaries+[len(tokens)]

def quiet_boundary(y,sr,clock,index):
 if index==0:return 0
 lo=round(clock['words'][index-1]['end']*sr);hi=round(clock['words'][index]['start']*sr)
 half=round(.010*sr)
 if hi-lo<2*half:raise ValueError('respiratory boundary has no verified quiet interval')
 candidates=np.arange(lo+half,hi-half+1,max(1,round(.002*sr)))
 level=np.array([np.sqrt(np.mean(y[t-half:t+half]**2)) for t in candidates]);j=int(level.argmin())
 if level[j]>max(.001,float(np.max(np.abs(y)))*.01):raise ValueError('respiratory boundary would cut active articulation')
 return int(candidates[j])

def realize(source,out,text,scene,prior,timeline,policy=POLICY):
 if policy not in {POLICY,'listener_causal_v6'}:raise ValueError('unsupported respiratory policy')
 source=pathlib.Path(source);out=pathlib.Path(out)
 if out.exists():raise FileExistsError(out)
 if sha(source)!=timeline['source_audio_sha256']:raise ValueError('respiratory clock belongs to another recording')
 raw,sr=sf.read(source,dtype='int16');y=raw.astype(float)/32768
 if sr!=24000 or raw.ndim!=1:raise ValueError('24k mono respiratory carrier required')
 actual=copy.deepcopy(scene or {});actual.setdefault('events',[])
 if any(e.get('kind')=='inhale' for e in actual['events']):raise ValueError('preplanned inhalation requires its own delivery receipt')
 clock=copy.deepcopy(timeline);boundaries=phrase_boundaries(text);events=[];insertions=[];offset=0
 for index,end in zip(boundaries[:-1],boundaries[1:]):
  plan=compile_scene(text,actual,prior,timeline=clock,policy=policy)
  state=plan['state_samples'][index]['state'];body=state['body_state']
  # Anticipate only the currently available linguistic phrase, using current
  # body demand. Later scene state is not allowed to color this earlier breath.
  seconds=sum(w['end']-w['start'] for w in clock['words'][index:end])
  cost=seconds*(.025+.03*body['exertion'])
  if cost>.85-FLOOR:raise ValueError('phrase exceeds modeled respiratory capacity; structural phrase division required')
  if body['breath_reserve']-cost>=FLOOR:continue
  target=min(.85,FLOOR+cost+.12);volume=target-body['breath_reserve']
  if volume<=0:raise ValueError('respiratory planner cannot repair this phase')
  intake_rate=2*(1-.25*body['fatigue'])
  frames=int(np.ceil(max(.18,volume/intake_rate)*sr));duration=frames/sr
  location=quiet_boundary(y,sr,timeline,index)
  cause={'text':'Available respiratory reserve cannot support the currently available phrase within the modeled lower bound.',
   'mechanism':'normalized respiratory conservation; quiet inhalation',
   'prior_state_hash':digest(state),'available_reserve':body['breath_reserve'],'predicted_phrase_cost':cost,
   'known_phrase_word_interval':[index,end],'coefficient_status':'engineering hypothesis, not measured physiology'}
  event={'kind':'inhale','id':'respiratory-intake-'+str(index),'at_word':index,'duration_s':duration,'reserve_increment':volume,'source':cause}
  actual['events'].append(event);events.append(event);insertions.append({'source_sample':location,'frames':frames,'at_word':index,'cause':event['id']})
  # Exact insertion transform, not an invented re-alignment.
  for word in clock['words'][index:]:word['start']+=duration;word['end']+=duration
  clock['duration_s']+=duration;offset+=frames
 plan=compile_scene(text,actual,prior,timeline=clock,policy=policy)
 if min(s['state']['body_state']['breath_reserve'] for s in plan['state_samples'])<FLOOR-1e-7:
  raise ValueError('unresolved respiratory deficit after causal planning')
 parts=[];cursor=0
 for item in insertions:
  at=item['source_sample'];parts.extend([raw[cursor:at],np.zeros(item['frames'],dtype=np.int16)]);cursor=at
 parts.append(raw[cursor:]);output=np.concatenate(parts);sf.write(out,output,sr,subtype='PCM_16')
 if not insertions and sha(out)!=sha(source):raise ValueError('no-breath route must retain exact carrier bytes')
 clock['source_audio_sha256']=sha(out);plan=compile_scene(text,actual,prior,timeline=clock,policy=policy);verify_plan(plan)
 receipt={'role':'quiet respiratory-budget diagnostic, not completed Mari performance','full_completion':False,
  'source_sha256':sha(source),'audio_sha256':sha(out),'implementation_sha256':sha(__file__),
  'source_timeline':timeline,'delivered_timeline':clock,'source_scene':scene,'delivered_scene':actual,'plan':plan,
  'insertions':insertions,'added_samples':offset,'unchanged_existing_pcm':True,'audible_breath_synthesis':False,
  'respiratory_coefficients':{'floor':FLOOR,'target_limit':.85,'reserve_use_per_s':'.025+.03*exertion','intake_per_s':'2*(1-.25*fatigue)'},
  'unqualified':['naturalness of quiet inhalation timing','audible breath timbre','measured physiological calibration','character presence']}
 durable_json(out.with_suffix('.respiration.json'),receipt);return receipt
