"""Source-derived physical vocal gestures, selected by explicit interaction cause.

These are phonological realization units, not emotional presets or a new voice.
The source carrier, actual capture interval, and remaining perceptual limits are
preserved independently. Future lexical text never enters event realization.
"""
import argparse,pathlib,json,copy,math
import numpy as np,soundfile as sf,parselmouth
from parselmouth.praat import call
from .nonlexical import compile_event
from .prominence_duration import deterministic_overlap_add
from .session import durable_json
from ..causal_v4.runtime import apply_event,advance,digest,validate_state,ANCHOR,PROFILE
from ..causal_v4.renderer import sha,inspect_audio

def build(root):
 r=pathlib.Path(root);d=r/'continuation/native_event_bank';d.mkdir(exist_ok=True);source=r/'continuation/ack_counterfactual';observed=json.loads((r/'continuation/acknowledgment_clock/RESULTS.json').read_text());rows=[]
 for row in observed:
  if not row['pass']:raise ValueError('unverified source event clock')
  kind=row['id'];p=source/(kind+'.wav');o=row['observation'];native_path=p.with_suffix('.receipt.json');native=json.loads(native_path.read_text())
  if sha(p)!=o['audio_sha256']or native['audio']['sha256']!=sha(p):raise ValueError('source native waveform changed')
  y,sr=sf.read(p,dtype='int16');z=y.astype(float)/32768;hop=round(.01*sr);half=round(.01*sr);threshold=max(.001,float(np.max(abs(z)))*.01);start=o['gesture_onset_s'];end=o['gesture_end_s'];before=range(max(half,round((start-.2)*sr)),round(start*sr)-half+1,hop);after=range(round(end*sr)+half,min(len(y)-half,round((end+.2)*sr))+1,hop)
  quiet=lambda q:float(np.sqrt(np.mean(z[q-half:q+half]**2)))<=threshold
  starts=[q for q in before if quiet(q)];ends=[q for q in after if quiet(q)]
  if not starts or not ends:raise ValueError('source has no verified quiet capture boundaries')
  a=max(starts);b=min(ends)
  if b>=round(o['lexical_phonation_onset_s']*sr):raise ValueError('capture includes following lexical phonation')
  path=d/(kind+'.wav');sf.write(path,y[a:b],sr,subtype='PCM_16');saved,_=sf.read(path,dtype='int16')
  if not np.array_equal(saved,y[a:b]):raise ValueError('capture altered source samples')
  regions=[dict(x,start=x['start']-a/sr,end=x['end']-a/sr)for x in o['event_regions']];rows.append({'intent':kind,'path':str(path),'sha256':sha(path),'source_audio_sha256':sha(p),'source_native_receipt_sha256':sha(native_path),'source_interval_samples':[a,b],'sample_rate':sr,'duration_s':len(saved)/sr,'voiced_regions':regions,'source_ASR_prefix':o['raw_ASR_prefix'],'source_episode_speaker_similarity':o['quality']['speaker_similarity'],'capture_boundaries_quiet':True,'capture_samples_exact':True})
 bank={'schema':'mari-native-vocal-gesture-bank/1.0','carrier_anchor_sha256':ANCHOR,'carrier_profile_sha256':PROFILE,'entries':rows,'source_clock_results_sha256':sha(r/'continuation/acknowledgment_clock/RESULTS.json'),'implementation_sha256':sha(__file__),'scope':'two source-derived nasal acknowledgment gestures from the existing selected native Mari carrier; no emotional preset, speaker search or character completion','meaning_or_isolated_identity_admitted':False,'full_completion':False};durable_json(d/'BANK.json',bank);return bank

def realize(prior,event,bank_path,out):
 validation=compile_event(prior,event);bank_path=pathlib.Path(bank_path);bank=json.loads(bank_path.read_text())
 if bank['schema']!='mari-native-vocal-gesture-bank/1.0'or bank['carrier_anchor_sha256']!=ANCHOR or bank['carrier_profile_sha256']!=PROFILE:raise ValueError('wrong gesture identity lineage')
 entry=next(x for x in bank['entries']if x['intent']==event['intent']);source=pathlib.Path(entry['path'])
 if sha(source)!=entry['sha256']:raise ValueError('selected physical gesture changed')
 out=pathlib.Path(out)
 if out.exists():raise FileExistsError(out)
 pcm,sr=sf.read(source,dtype='int16');y=pcm.astype(float)/32768;body=prior['body_state'];scale=1+.15*prior['mental_state']['load']+.08*body['fatigue'];support=body['breath_reserve']*(1-.35*body['fatigue']);pressure=(.65+.35*support)/(.65+.35*.85)
 if not 1<=scale<=1.23 or support<.12:raise ValueError('vocal gesture support/duration outside bounded neighborhood')
 seed=None
 if scale!=1:
  s=parselmouth.Sound(y,sampling_frequency=sr);m=call(s,'To Manipulation',.005,70,500);tier=call(m,'Extract duration tier');call(tier,'Remove points between',0.,s.xmax);call(tier,'Add point',0.,scale);call(tier,'Add point',s.xmax,scale);call([tier,m],'Replace duration tier');y,seed=deterministic_overlap_add(m,pcm);n=round(len(pcm)*scale)
  if len(y)<n:raise ValueError('event resynthesis truncated')
  y=y[:n]
 y*=pressure
 if not np.isfinite(y).all()or np.max(abs(y))>=.98:raise ValueError('event output quality failed')
 sf.write(out,y,sr,subtype='PCM_16');_,_,audio=inspect_audio(out);state=copy.deepcopy(prior);apply_event(state,dict(event,kind='nonlexical',at_word=0),policy='contrast_focus_v7');heard=state['listener_model'].setdefault('heard_propositions',[])
 if event['proposition']not in heard:heard.append(event['proposition'])
 t=0.;segments=[]
 for region in entry['voiced_regions']:
  a=max(t,region['start']*scale);b=min(audio['duration_s'],region['end']*scale)
  if a>t:advance(state,a-t,speaking=False,respiratory_rest=False)
  if b>a:advance(state,b-a,speaking=True,respiratory_rest=False);segments.append({'start':a,'end':b,'cause':event['id']})
  t=b
 if audio['duration_s']>t:advance(state,audio['duration_s']-t,speaking=False,respiratory_rest=False)
 state['continuity_links']={'parent_state_hash':digest(prior),'journal_head':digest({'parent':prior['continuity_links']['journal_head'],'event':event,'audio':audio['sha256']})};validate_state(state)
 receipt={'schema':'mari-source-derived-vocal-event/1.0','role':'diagnostic causal physical vocal gesture','source_bank_sha256':sha(bank_path),'source_entry':entry,'event':copy.deepcopy(event),'initial_state':copy.deepcopy(prior),'final_state':state,'source_validation_plan_hash':validation['plan_hash'],'duration_scale':scale,'relative_pressure':pressure,'support':support,'voiced_segments':segments,'resynthesis_seed':seed,'audio':audio,'future_lexical_context_used':False,'new_speaker_or_backend':False,'implementation_sha256':sha(__file__),'perceived_meaning_admitted':False,'full_completion':False};durable_json(out.with_suffix('.event.json'),receipt);return receipt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();bank=build(a.root);print([(x['intent'],x['duration_s'])for x in bank['entries']],flush=True)
if __name__=='__main__':main()
