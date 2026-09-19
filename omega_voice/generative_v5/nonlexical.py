"""Source-grounded vocal events, separate from lexical TTS.

The first diagnostic actuator is a closed-mouth acknowledgment using measured
Mari nasal phonation. It is not a spelling sent to TTS. Its interaction meaning
and perceived naturalness remain unqualified until an independent route passes.
"""
import copy,json,pathlib
import numpy as np,soundfile as sf
from ..causal_v4.runtime import validate_state,apply_event,advance,digest,ANCHOR
from ..causal_v4.renderer import sha
from .session import durable_json

def compile_event(prior,event):
 validate_state(prior)
 if set(event)!={'id','behavior','intent','proposition','source'} or not event['id'] or not isinstance(event['source'],dict) or not event['source'].get('text'):
  raise ValueError('explicit complete event cause required')
 if event['behavior']!='acknowledgment' or event['intent'] not in {'receipt','agreement'}:
  raise ValueError('nonlexical actuator not implemented for this behavior')
 if not isinstance(event['proposition'],str) or not event['proposition']:raise ValueError('heard proposition required')
 knowledge=prior['knowledge_state'];evidence=knowledge['known'].get(event['proposition']) or knowledge['beliefs'].get(event['proposition'])
 if event['intent']=='agreement' and (evidence is None or evidence['confidence']<.7):raise ValueError('agreement requires prior affirmative evidence; acknowledgment is not assent')
 state=copy.deepcopy(prior);cause=dict(event,kind='nonlexical',at_word=0)
 apply_event(state,cause,policy='linguistic_scope_v4')
 heard=state['listener_model'].setdefault('heard_propositions',[])
 if event['proposition'] not in heard:heard.append(event['proposition'])
 # Bounded engineering timing hypotheses. No random pulse count or tic.
 duration=.24+.16*state['mental_state']['load']+.08*state['body_state']['fatigue']
 support=state['body_state']['breath_reserve']*(1-.35*state['body_state']['fatigue'])
 if support<.12:raise ValueError('insufficient respiratory reserve; breath planning required')
 trajectory={'duration_s':duration,'relative_pressure':min(1.,.65+.35*support),
  'terminal_f0_semitones':-1.2*evidence['confidence'] if event['intent']=='agreement' else 0.,
  'cause':event['id'],'physical_role':'brief supported nasal phonation, no breathiness or lexical substitution'}
 advance(state,duration,speaking=True,respiratory_rest=False);state['turn']+=1
 state['continuity_links']={'parent_state_hash':digest(prior),'journal_head':digest({'parent':prior['continuity_links']['journal_head'],'event':event})}
 plan={'schema':'mari-vocal-event/1.0','initial_state':copy.deepcopy(prior),'event':copy.deepcopy(event),'trajectory':trajectory,
  'final_state':state,'coefficient_status':'bounded engineering hypotheses; perceptual qualification required','full_completion':False}
 plan['plan_hash']=digest(plan);return plan

def realize(plan,source_features,source_provenance,out):
 import pyworld as pw
 if compile_event(plan['initial_state'],plan['event'])!=plan:raise ValueError('vocal event replay mismatch')
 out=pathlib.Path(out)
 if out.exists():raise FileExistsError(out)
 provenance=json.loads(pathlib.Path(source_provenance).read_text())
 if provenance['source_carrier_sha256']!=ANCHOR or sha(source_features)!=provenance['features_sha256']:raise ValueError('nonlexical source identity/provenance mismatch')
 features=np.load(source_features);f0=float(features['f0']);spectrum=features['spectrum'];ap=features['aperiodicity']
 trajectory=plan['trajectory'];dt=.005;duration=trajectory['duration_s'];t=np.arange(0,duration+dt,dt);u=np.clip(t/duration,0,1)
 # Continuous pressure gesture; source spectrum and aperiodicity retain the
 # measured carrier. No recorded laughter, style description or text token.
 envelope=np.sin(np.pi*u)**.8*trajectory['relative_pressure']
 contour=f0*2**(trajectory['terminal_f0_semitones']*u*u/12)
 spec=np.tile(spectrum,(len(t),1))*envelope[:,None]**2
 periodicity=np.tile(ap,(len(t),1));y=pw.synthesize(contour,np.ascontiguousarray(spec),np.ascontiguousarray(periodicity),24000,5.)
 y=y[:round(duration*24000)]
 if not np.isfinite(y).all() or np.max(np.abs(y))>=.98:raise ValueError('nonlexical output invalid')
 sf.write(out,y,24000,subtype='PCM_16')
 receipt={'role':'unqualified nonlexical mechanism diagnostic','audio_sha256':sha(out),'plan':plan,
  'source_provenance':provenance,'implementation_sha256':sha(__file__),'lexical_tts_called':False,
  'random_behavior':False,'full_completion':False,'perceptual_admission':False}
 durable_json(out.with_suffix('.receipt.json'),receipt);return receipt
