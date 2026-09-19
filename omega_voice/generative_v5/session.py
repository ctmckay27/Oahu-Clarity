"""Recoverable performance-session integration with explicit admission status.

The current numeric actuator covers boundary contour only. This implementation
can execute diagnostic stateful turns; ordinary-English completion admission
remains closed until the separate nineteen-condition assessment actually passes.
"""
import copy,json,pathlib,fcntl,os
import numpy as np
from ..causal_v4.runtime import compile_scene,digest
from ..causal_v4.renderer import sha,UnresolvedRealization
from .native import render,write_trajectory
from .trajectory import compile_finality,alignment_error

def durable_json(path,value):
 path=pathlib.Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:
  json.dump(value,f,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x));f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(path)
 fd=os.open(path.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)

def timeline_from_evaluation(result):
 a=result['alignment']
 if not a['exact_words']:raise ValueError('cannot commit an unverified spoken transcript')
 return {'source_audio_sha256':a['source_sha256'],'duration_s':result['audio']['duration_s'],
         'words':[{'start':float(w['start']),'end':float(w['end'])} for w in a['words']]}

def unresolved_channels(plan):
 fresh=compile_scene(plan['text'],timeline=plan.get('realization_timeline'))
 channels=set()
 for actual,base in zip(plan['trajectory'],fresh['trajectory']):
  for key in ['precision','attack_softness','phonatory_tension','support','breathiness','emphasis','rate','gain_db','onset_delay_s']:
   if abs(actual['controls'][key]-base['controls'][key])>.01:channels.add(key)
 for e in plan['journal']:
  if e['event']['kind'] in ['nonlexical','interrupt','resume']:channels.add(e['event']['kind'])
 return sorted(channels)

class PerformanceSession:
 def __init__(self,root,directory,evaluator,bank=None,bank_sha256=None,mode='native',aligner=None):
  self.root=pathlib.Path(root);self.directory=pathlib.Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
  if mode not in ['native','physical']:raise ValueError('unknown realization mechanism')
  if mode=='physical' and aligner is None:raise ValueError('physical session requires independently qualified alignment')
  self.evaluator=evaluator;self.mode=mode;self.aligner=aligner
  self.bank_path=pathlib.Path(bank) if bank else None
  if mode=='native':
   if not self.bank_path or sha(self.bank_path)!=bank_sha256:raise ValueError('unselected control-bank bytes')
   self.bank=np.load(self.bank_path)['directions']
  self.state_path=self.directory/'session_state.json'

 def recover_commit(self):
  pending=self.directory/'pending_commit.json'
  if not pending.exists():return
  journal=json.loads(pending.read_text());receipt=self.directory/journal['receipt']
  if not receipt.is_file() or sha(receipt)!=journal['receipt_sha256']:raise RuntimeError('incomplete or changed session receipt; refusing recovery')
  current=json.loads(self.state_path.read_text()) if self.state_path.exists() else None
  if current!=journal['next'] and digest(current)!=journal['parent_record_hash']:raise RuntimeError('session journal conflicts with current state')
  durable_json(self.state_path,journal['next']);pending.unlink()

 def commit(self,previous,saved,receipt_path,receipt):
  # The accepted waveform and semantic receipt exist before state advances.
  durable_json(receipt_path,receipt)
  pending=self.directory/'pending_commit.json'
  saved=dict(saved,receipt=str(receipt_path.relative_to(self.directory)),receipt_sha256=sha(receipt_path))
  durable_json(pending,{'parent_record_hash':digest(previous),'next':saved,
                       'receipt':saved['receipt'],'receipt_sha256':saved['receipt_sha256']})
  self.recover_commit()

 def render_turn(self,request_id,text,scene=None,seed=88000,diagnostic=False):
  with (self.directory/'session.lock').open('a') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:raise RuntimeError('session already rendering; simultaneous turn admission denied')
   try:return self._render_turn(request_id,text,scene,seed,diagnostic)
   finally:fcntl.flock(lock,fcntl.LOCK_UN)

 def _render_turn(self,request_id,text,scene=None,seed=88000,diagnostic=False):
  if not diagnostic:raise UnresolvedRealization('nineteen-condition ordinary-English completion admission is not yet satisfied')
  if not request_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in request_id):raise ValueError('invalid request id')
  self.recover_commit()
  previous=json.loads(self.state_path.read_text()) if self.state_path.exists() else None
  if previous and sha(self.directory/previous['receipt'])!=previous['receipt_sha256']:raise RuntimeError('committed session evidence changed')
  prior=previous['state'] if previous else None
  request_hash=digest({'text':text,'scene':scene or {},'seed':seed,'parent':digest(prior) if prior else None,'mechanism':self.mode})
  target=self.directory/request_id
  if target.exists():raise FileExistsError('request already exists; inspect its persisted receipt rather than regenerate or double-commit')
  target.mkdir();draft=compile_scene(text,scene,prior)
  carrier=target/'carrier.wav';render(self.root,text,carrier,seed)
  base=self.evaluator.evaluate(carrier,text)
  if not base['quality_screen_pass']:raise UnresolvedRealization('carrier quality gate failed')
  if self.mode=='physical':return self._render_physical(request_id,request_hash,target,carrier,base,text,scene,prior,previous)
  reference=base;trials=[]
  for attempt in range(3):
   plan=compile_scene(text,scene,prior,timeline=timeline_from_evaluation(reference))
   weights,packet=compile_finality(plan,reference['alignment'],reference['audio']['duration_s'])
   control=target/f'control_{attempt}.mtraj';write_trajectory(control,self.bank,weights)
   audio=target/f'performance_{attempt}.wav';render(self.root,text,audio,seed,trajectory=control)
   result=self.evaluator.evaluate(audio,text);error=alignment_error(packet['segments'],result['alignment'])
   trials.append({'audio':str(audio),'evaluation':result,'packet':packet,'alignment_error_s':float(error)})
   if not result['quality_screen_pass']:break
   if error<=.16:break
   reference=result
  final=trials[-1];admitted=final['evaluation']['quality_screen_pass'] and final['alignment_error_s']<=.16
  receipt={'request_id':request_id,'request_hash':request_hash,'parent_state_hash':digest(prior) if prior else None,
           'role':'diagnostic stateful realization, not completed Mari performance','trials':trials,'quality_admitted':bool(admitted),
           'unresolved_channels':unresolved_channels(draft),'full_completion':False,'bank_sha256':sha(self.bank_path)}
  if admitted:
   delivered=compile_scene(text,scene,prior,timeline=timeline_from_evaluation(final['evaluation']))
   receipt['delivered_plan']=delivered
   # Reject concurrent state advancement before committing the new actual clock.
   current=json.loads(self.state_path.read_text()) if self.state_path.exists() else None
   if current!=previous:raise RuntimeError('session advanced concurrently; refusing stale state commit')
   saved={'scope':'diagnostic session continuity','last_request':request_id,'state':delivered['final_state'],
          'audio_sha256':final['evaluation']['audio']['sha256'],'parent_record_hash':digest(previous) if previous else None}
   self.commit(previous,saved,target/'receipt.json',receipt)
  else:durable_json(target/'receipt.json',receipt)
  if not admitted:raise UnresolvedRealization('native diagnostic quality/alignment gate failed; state not committed')
  return receipt

 def _render_physical(self,request_id,request_hash,target,carrier,base,text,scene,prior,previous):
  from .physical_finality import realize as finality
  from .physical import realize as body
  from .alignment import AlignmentRejected
  receipt={'request_id':request_id,'request_hash':request_hash,'parent_state_hash':digest(prior) if prior else None,
   'role':'diagnostic stateful physical realization, not completed Mari performance','mechanism':'sample-aligned physical',
   'full_completion':False,'quality_admitted':False,'baseline':base}
  try:
   alignment=self.aligner.align(self.evaluator.load(carrier),text,sha(carrier),base['wer']==0)
   aligned=dict(base,asr_alignment=base['alignment'],alignment=alignment)
   timeline=timeline_from_evaluation(aligned);plan=compile_scene(text,scene,prior,timeline=timeline)
   contour=target/'contour.wav';contour_receipt=finality(carrier,contour,plan)
   # Both physical mechanisms preserve sample count. Rebind provenance, never
   # alter a measured boundary to make a failed alignment test pass.
   body_clock=dict(timeline,source_audio_sha256=sha(contour));body_plan=compile_scene(text,scene,prior,timeline=body_clock)
   audio=target/'performance.wav';body_receipt=body(contour,audio,body_plan)
   result=self.evaluator.evaluate(audio,text)
   if result['audio']['frames']!=base['audio']['frames']:raise UnresolvedRealization('physical mechanism changed the measured clock')
   observed=self.aligner.align(self.evaluator.load(audio),text,sha(audio),result['wer']==0)
   error=max(abs(a[k]-b[k]) for a,b in zip(alignment['words'],observed['words']) for k in ['start','end'])
   admitted=result['quality_screen_pass'] and result['wer']==0 and error<=.08
   unresolved=set(unresolved_channels(plan))-{'phonatory_tension','attack_softness','gain_db'}
   receipt.update(quality_admitted=bool(admitted),unresolved_channels=sorted(unresolved),alignment=alignment,
    trials=[{'audio':str(audio),'evaluation':result,'alignment_error_s':error,'observed_alignment':observed,
             'finality':contour_receipt,'body':body_receipt}])
   if not admitted:raise UnresolvedRealization('physical diagnostic quality gate failed')
   delivered=compile_scene(text,scene,prior,timeline=timeline_from_evaluation(dict(result,alignment=observed)))
   receipt['delivered_plan']=delivered
   current=json.loads(self.state_path.read_text()) if self.state_path.exists() else None
   if current!=previous:raise RuntimeError('session advanced concurrently; refusing stale state commit')
   saved={'scope':'diagnostic session continuity','last_request':request_id,'state':delivered['final_state'],
          'audio_sha256':sha(audio),'parent_record_hash':digest(previous) if previous else None}
  except Exception as error:
   receipt.update(quality_admitted=False,error=str(error),state_committed=False)
   if isinstance(error,AlignmentRejected):receipt['rejected_alignment']=error.report
   durable_json(target/'receipt.json',receipt)
   raise
  # Commit failures must retain the exact receipt already referenced by the
  # durable journal, so recovery can finish once without a changed hash.
  self.commit(previous,saved,target/'receipt.json',receipt)
  return receipt
