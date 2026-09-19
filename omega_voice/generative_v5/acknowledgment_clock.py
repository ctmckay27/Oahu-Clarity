"""Measure vocal gesture and lexical continuation as distinct performed events."""
import copy,pathlib,json
import numpy as np,torch,parselmouth,soundfile as sf
from .session import durable_json
from .nonlexical import compile_event
from ..causal_v4.runtime import apply_event,advance,compile_scene,digest,validate_state
from ..causal_v4.evaluate import normalized
from ..causal_v4.renderer import sha,UnresolvedRealization

class EpisodeRejected(UnresolvedRealization):
 def __init__(self,message,report):
  self.report=report;super().__init__(message)

def observe_episode(path,event,continuation,evaluator,aligner,output_directory):
 path=pathlib.Path(path);form={'receipt':'Mm.','agreement':'Mm-hm.'}[event['intent']];quality=evaluator.evaluate(path,form+' '+continuation);expected=normalized(continuation);hyp=normalized(quality['asr_text']);prefix=hyp[:-len(expected)]if len(hyp)>len(expected)and hyp[-len(expected):]==expected else None
 allowed={'receipt':[['mm'],['mmm']],'agreement':[['mm','hm'],['mm','hmm'],['mhm'],['mmhmm']]}
 checks={'whole_episode_identity':quality['identity_pass'],'lexical_continuation_exact':prefix is not None,'expected_event_phonology':prefix in allowed[event['intent']]}
 if not all(checks.values()):raise EpisodeRejected('episode admission failed: '+', '.join(k for k,v in checks.items()if not v),{'quality':quality,'checks':checks,'observed_prefix':prefix,'expected_phonology':allowed[event['intent']],'full_completion':False})
 wave16=evaluator.load(path);words=normalized(continuation)
 with torch.inference_mode():
  emission,_=aligner.model(torch.as_tensor(wave16,dtype=torch.float32)[None]);emission=emission[0];star=torch.zeros((len(emission),1),dtype=emission.dtype);extended=torch.cat([emission,star],dim=-1);spans=aligner.aligner(extended,[[emission.shape[-1]]]+aligner.tokenizer(words))
 step=len(wave16)/16000/len(emission);lexical=[]
 for word,ss in zip(words,spans[1:]):
  duration=sum(s.end-s.start for s in ss);prob=sum(s.score*(s.end-s.start)for s in ss)/duration;lexical.append({'word':word,'start':float(ss[0].start*step),'end':float(ss[-1].end*step),'probability':float(prob)})
 if len(lexical)!=len(words)or min(x['probability']for x in lexical)<.35:raise UnresolvedRealization('lexical continuation clock rejected')
 y,sr=sf.read(path,dtype='float64');snd=parselmouth.Sound(y,sampling_frequency=sr);pitch=snd.to_pitch_ac(time_step=.005,pitch_floor=70,pitch_ceiling=500);hz=pitch.selected_array['frequency'];times=pitch.xs();active=np.flatnonzero(hz>0)
 if len(active)<24:raise UnresolvedRealization('no bounded prelexical phonation observed')
 groups=np.split(active,np.flatnonzero(np.diff(active)>.04/.005)+1);regions=[{'start':max(0.,float(times[g[0]]-.0025)),'end':float(times[g[-1]]+.0025),'voiced_frames':len(g),'pitch_start_hz':float(np.median(hz[g[:max(1,len(g)//3)]])),'pitch_end_hz':float(np.median(hz[g[-max(1,len(g)//3):]]))}for g in groups if len(g)>=8]
 lexical_regions=[x for x in regions if x['start']<=lexical[0]['end']and x['end']>=lexical[0]['start']]
 if not lexical_regions:raise UnresolvedRealization('first lexical phonation is not observable')
 lexical_onset=lexical_regions[0]['start'];regions=[x for x in regions if x['end']<=lexical_onset-.08]
 if not regions or regions[-1]['end']-regions[0]['start']>1.5:raise UnresolvedRealization('gesture/lexical separation is not observable')
 split_sample=round((regions[-1]['end']+.02)*sr);split=split_sample/sr
 output_directory=pathlib.Path(output_directory);output_directory.mkdir(exist_ok=True,parents=True);suffix=output_directory/(path.stem+'_lexical_suffix.wav');sf.write(suffix,y[split_sample:],sr,subtype='PCM_16')
 original,_=sf.read(path,dtype='int16');saved,_=sf.read(suffix,dtype='int16')
 if not np.array_equal(original[split_sample:],saved):raise ValueError('suffix artifact changed source samples')
 return {'audio_sha256':sha(path),'quality':quality,'lexical_continuation_exact':True,'raw_ASR_prefix':prefix,'event_regions':regions,'gesture_onset_s':regions[0]['start'],'gesture_end_s':regions[-1]['end'],'split_sample':split_sample,'split_s':split,'lexical_suffix_audio':str(suffix),'lexical_suffix_audio_sha256':sha(suffix),'lexical_phonation_onset_s':lexical_onset,'lexical_alignment':lexical,'lexical_min_probability':min(x['probability']for x in lexical),'alignment_model_sha256':aligner.receipt['sha256'],'clock_method':'MMS lexical suffix with unknown prefix; independently observed prelexical phonation; STAR is not an event classifier','event_meaning_admitted':False,'isolated_event_identity_admitted':False,'full_completion':False}

def reconstruct_episode(prior,event,continuation,observation):
 # The legacy plan validates source and prior knowledge. Its hypothesized
 # duration is deliberately not used to commit the actual performed event.
 validated=compile_event(prior,event);state=copy.deepcopy(prior);apply_event(state,dict(event,kind='nonlexical',at_word=0),policy='contrast_focus_v7');heard=state['listener_model'].setdefault('heard_propositions',[])
 if event['proposition']not in heard:heard.append(event['proposition'])
 t=0.;segments=[]
 for region in observation['event_regions']:
  if region['start']>t:advance(state,region['start']-t,speaking=False,respiratory_rest=False);segments.append({'start':t,'end':region['start'],'speaking':False,'inhalation_inferred':False})
  advance(state,region['end']-region['start'],speaking=True,respiratory_rest=False);segments.append({'start':region['start'],'end':region['end'],'speaking':True,'cause':event['id']});t=region['end']
 split=observation['split_s']
 if split<t:raise ValueError('event split precedes observed gesture')
 advance(state,split-t,speaking=False,respiratory_rest=False)
 if sha(observation['lexical_suffix_audio'])!=observation['lexical_suffix_audio_sha256']:raise ValueError('measured suffix artifact changed')
 timeline={'source_audio_sha256':observation['lexical_suffix_audio_sha256'],'duration_s':observation['quality']['audio']['duration_s']-split,'words':[{'start':w['start']-split,'end':w['end']-split}for w in observation['lexical_alignment']]}
 plan=compile_scene(continuation,prior=state,timeline=timeline,policy='contrast_focus_v7');final=copy.deepcopy(plan['final_state']);final['continuity_links']={'parent_state_hash':digest(prior),'journal_head':digest({'parent':prior['continuity_links']['journal_head'],'event':event,'observation':observation,'lexical_plan':plan['plan_hash']})};validate_state(final)
 if abs(final['time_s']-prior['time_s']-observation['quality']['audio']['duration_s'])>1e-8:raise ValueError('performed clock not conserved')
 return {'schema':'mari-native-acknowledgment-episode/1.0','initial_state':copy.deepcopy(prior),'event':copy.deepcopy(event),'source_validation_plan_hash':validated['plan_hash'],'observed_event_segments':segments,'lexical_plan':plan,'final_state':final,'original_audio_preserved':True,'unrealized_intent_channels':['hypothesized event duration','relative pressure','state-controlled terminal pitch'],'acoustic_phonetic_form_observed':True,'perceived_acknowledgment_meaning_verified':False,'full_completion':False}

def render_session_episode(session,request_id,event,continuation,aligner,seed,diagnostic):
 import fcntl
 from .native import render,anchor_reference
 from ..causal_v4.runtime import new_state
 if not diagnostic:raise UnresolvedRealization('ordinary-English completion admission remains closed')
 if session.mode!='physical'or session.conditioning!='selected_anchor'or session.temporal_policy!='contrast_focus_v7':raise ValueError('acknowledgment requires the selected-anchor physical continuity session')
 if not request_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'for c in request_id):raise ValueError('invalid request id')
 if not isinstance(continuation,str)or not normalized(continuation):raise ValueError('explicit lexical continuation required for contextual identity verification')
 with (session.directory/'session.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('session already rendering')
  try:
   session.recover_commit();previous=json.loads(session.state_path.read_text())if session.state_path.exists()else None
   if previous and sha(session.directory/previous['receipt'])!=previous['receipt_sha256']:raise ValueError('previous session evidence changed')
   listener=event.get('source',{}).get('listener_id')
   if not isinstance(listener,str)or not listener:raise ValueError('acknowledgment needs the identified listener')
   prior=previous['state']if previous else new_state(session.directory.name,listener)
   if prior['listener_model']['id']!=listener:raise ValueError('acknowledgment belongs to another listener')
   compile_event(prior,event)
   target=session.directory/request_id
   if target.exists():raise FileExistsError('request already exists')
   target.mkdir();form={'receipt':'Mm.','agreement':'Mm-hm.'}[event['intent']];text=form+' '+continuation;audio=target/'performance.wav';native=render(session.root,text,audio,seed=seed,reference=anchor_reference(session.root))
   try:
    observed=observe_episode(audio,event,continuation,session.evaluator,aligner,target);delivered=reconstruct_episode(prior,event,continuation,observed)
   except Exception as e:
    durable_json(target/'receipt.json',{'role':'rejected diagnostic native acknowledgment episode','error':repr(e),'failure_evidence':getattr(e,'report',None),'native':native,'quality_admitted':False,'state_committed':False,'full_completion':False});raise
   receipt={'role':'diagnostic source-grounded native acknowledgment with observed continuity','request_id':request_id,'parent_state_hash':digest(prior),'native':native,'observation':observed,'delivered_episode':delivered,'quality_admitted':True,'quality_scope':'whole-speaker identity, exact lexical continuation, observed phonetic event class and actual clock; raw nonlexical orthographic WER retained','unrealized_channels':delivered['unrealized_intent_channels'],'full_completion':False}
   current=json.loads(session.state_path.read_text())if session.state_path.exists()else None
   if current!=previous:raise RuntimeError('stale acknowledgment commit')
   saved={'scope':'diagnostic observed acknowledgment continuity','last_request':request_id,'state':delivered['final_state'],'audio_sha256':sha(audio),'parent_record_hash':digest(previous)if previous else None};session.commit(previous,saved,target/'receipt.json',receipt);return receipt
  finally:fcntl.flock(lock,fcntl.LOCK_UN)
