"""Transactional vocal event then causally continued speech through one interface."""
import pathlib,json,copy,fcntl
import numpy as np,soundfile as sf
from .native_event_bank import realize
from .session import PerformanceSession,durable_json
from ..causal_v4.runtime import new_state,validate_state,digest
from ..causal_v4.evaluate import normalized
from ..causal_v4.renderer import sha,UnresolvedRealization

def render_episode(session,request_id,event,continuation,bank_path,scene=None,seed=88000,diagnostic=False):
 if not diagnostic:raise UnresolvedRealization('ordinary-English completion admission remains closed')
 if session.mode!='physical'or session.conditioning!='selected_anchor'or session.temporal_policy!='contrast_focus_v7':raise ValueError('selected-anchor physical continuity session required')
 if not request_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'for c in request_id):raise ValueError('invalid request id')
 if not isinstance(continuation,str)or not normalized(continuation):raise ValueError('lexical continuation required for contextual verification')
 with(session.directory/'session.lock').open('a')as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('session already rendering')
  try:
   session.recover_commit();previous=json.loads(session.state_path.read_text())if session.state_path.exists()else None
   if previous and sha(session.directory/previous['receipt'])!=previous['receipt_sha256']:raise ValueError('prior continuity evidence changed')
   listener=event.get('source',{}).get('listener_id')
   if not isinstance(listener,str)or not listener:raise ValueError('identified listener required')
   prior=previous['state']if previous else new_state(session.directory.name,listener)
   if prior['listener_model']['id']!=listener:raise ValueError('wrong listener')
   target=session.directory/request_id
   if target.exists():raise FileExistsError('request already exists')
   target.mkdir();stage=PerformanceSession(session.root,target/'staging',session.evaluator,mode='physical',aligner=session.aligner,temporal_policy=session.temporal_policy,scene_compiler=session.scene_compiler,conditioning='selected_anchor',articulation=session.articulation,cold_start=session.cold_start,respiration=session.respiration,prominence=session.prominence,prominence_calibration=session.prominence_calibration)
   try:
    event_audio=stage.directory/'event.wav';event_receipt=realize(prior,event,bank_path,event_audio);stage.commit(None,{'scope':'private uncommitted episode stage','last_request':'event','state':event_receipt['final_state'],'audio_sha256':sha(event_audio),'parent_record_hash':digest(previous)if previous else None},stage.directory/'event.receipt.json',event_receipt)
    lexical=stage.render_turn('lexical',continuation,scene,seed=seed,diagnostic=True);lexical_audio=stage.directory/'lexical/performance.wav';a,sr=sf.read(event_audio,dtype='int16');b,bsr=sf.read(lexical_audio,dtype='int16')
    if sr!=bsr or sr!=24000:raise ValueError('component sample rates differ')
    output=target/'performance.wav';sf.write(output,np.concatenate([a,b]),sr,subtype='PCM_16');form={'receipt':'Mm.','agreement':'Mm-hm.'}[event['intent']];quality=session.evaluator.evaluate(output,form+' '+continuation);hyp=normalized(quality['asr_text']);expected=normalized(continuation);exact_suffix=len(hyp)>=len(expected)and hyp[-len(expected):]==expected
    if not quality['identity_pass']or quality['audio']['clipping_fraction']!=0 or not exact_suffix:raise UnresolvedRealization('joined contextual identity, speech or technical gate failed')
    saved=json.loads(stage.state_path.read_text());final=copy.deepcopy(saved['state']);final['continuity_links']={'parent_state_hash':digest(prior),'journal_head':digest({'parent':prior['continuity_links']['journal_head'],'event_audio':sha(event_audio),'event_cause':event,'lexical_plan':lexical['delivered_plan']['plan_hash']})};validate_state(final)
    if abs(final['time_s']-prior['time_s']-quality['audio']['duration_s'])>1e-7:raise ValueError('episode actual clock mismatch')
    joined,_=sf.read(output,dtype='int16')
    if not(np.array_equal(joined[:len(a)],a)and np.array_equal(joined[len(a):],b)):raise ValueError('joined samples differ from verified components')
    receipt={'schema':'mari-banked-acknowledgment-episode/1.0','role':'diagnostic physical vocal event followed by state-conditioned ordinary speech','request_id':request_id,'event':event_receipt,'lexical':lexical,'quality':quality,'raw_ASR_prefix':hyp[:-len(expected)],'lexical_continuation_exact':exact_suffix,'actual_event_samples':len(a),'components_exact':True,'parent_state_hash':digest(prior),'final_state':final,'future_text_affects_event':False,'quality_admitted':True,'event_meaning_and_isolated_identity_unverified':True,'implementation_sha256':sha(__file__),'full_completion':False}
   except Exception as e:
    durable_json(target/'receipt.json',{'role':'rejected diagnostic physical acknowledgment episode','error':repr(e),'state_committed':False,'full_completion':False});raise
   current=json.loads(session.state_path.read_text())if session.state_path.exists()else None
   if current!=previous:raise RuntimeError('stale whole-episode commit')
   session.commit(previous,{'scope':'diagnostic event-to-speech continuity','last_request':request_id,'state':final,'audio_sha256':quality['audio']['sha256'],'parent_record_hash':digest(previous)if previous else None},target/'receipt.json',receipt);return receipt
  finally:fcntl.flock(lock,fcntl.LOCK_UN)
