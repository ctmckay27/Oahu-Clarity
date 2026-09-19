"""Recoverable performance-session integration with explicit admission status.

The current numeric actuator covers boundary contour only. This implementation
can execute diagnostic stateful turns; ordinary-English completion admission
remains closed until the separate nineteen-condition assessment actually passes.
"""
import copy,json,pathlib,fcntl,os,shutil
import numpy as np
from ..causal_v4.runtime import compile_scene,digest,ANCHOR
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
 fresh=compile_scene(plan['text'],timeline=plan.get('realization_timeline'),policy=plan.get('temporal_policy'))
 channels=set()
 for actual,base in zip(plan['trajectory'],fresh['trajectory']):
  for key in ['precision','attack_softness','phonatory_tension','support','breathiness','emphasis','rate','gain_db','onset_delay_s']:
   if abs(actual['controls'][key]-base['controls'][key])>.01:channels.add(key)
 for e in plan['journal']:
  if e['event']['kind'] in ['nonlexical','interrupt','resume']:channels.add(e['event']['kind'])
 return sorted(channels)

class PerformanceSession:
 def __init__(self,root,directory,evaluator,bank=None,bank_sha256=None,mode='native',aligner=None,temporal_policy=None,scene_compiler=None,conditioning='independent',articulation=False,cold_start='profile_only',respiration=False,prominence=False,prominence_calibration=None):
  self.root=pathlib.Path(root);self.directory=pathlib.Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
  if mode not in ['native','physical']:raise ValueError('unknown realization mechanism')
  if mode=='physical' and aligner is None:raise ValueError('physical session requires independently qualified alignment')
  self.evaluator=evaluator;self.mode=mode;self.aligner=aligner;self.temporal_policy=temporal_policy
  self.scene_compiler=scene_compiler
  if conditioning not in {'independent','previous_carrier','selected_anchor'}:raise ValueError('unknown conditioning policy')
  if conditioning=='previous_carrier' and mode!='physical':raise ValueError('previous-carrier context qualified only for physical diagnostic sessions')
  self.conditioning=conditioning
  if articulation and mode!='physical':raise ValueError('consonant precision requires physical mechanism')
  self.articulation=bool(articulation)
  if respiration and (mode!='physical' or temporal_policy not in {'respiratory_budget_v5','listener_causal_v6','contrast_focus_v7'}):raise ValueError('respiratory planning requires physical mechanism and respiratory_budget_v5 or successor')
  self.respiration=bool(respiration)
  if type(prominence) is not bool or (prominence and (mode!='physical' or temporal_policy!='contrast_focus_v7')):raise ValueError('prominence requires physical contrast_focus_v7')
  self.prominence=prominence;self.prominence_calibration=pathlib.Path(prominence_calibration) if prominence_calibration else None
  if prominence and (self.prominence_calibration is None or sha(self.prominence_calibration)!='cc38d2a2d23689ab735a92e5076e8dbd7a11466dac83fd8c66e03de3b8c888c3'):raise ValueError('selected prominence calibration required')
  if not prominence and prominence_calibration is not None:raise ValueError('unused prominence calibration')
  if cold_start not in {'profile_only','selected_anchor_icl'}:raise ValueError('unknown cold-start conditioning')
  self.cold_start=cold_start
  self.bank_path=pathlib.Path(bank) if bank else None
  if mode=='native':
   if not self.bank_path or sha(self.bank_path)!=bank_sha256:raise ValueError('unselected control-bank bytes')
   self.bank=np.load(self.bank_path)['directions']
  self.state_path=self.directory/'session_state.json'

 def compile(self,text,scene=None,prior=None,timeline=None):
  return compile_scene(text,scene,prior,timeline=timeline,policy=self.temporal_policy)

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
  source_scene=scene;scene_record=None
  if isinstance(scene,str) or (isinstance(scene,dict) and 'direction' in scene):
   if self.scene_compiler is None:raise ValueError('natural-language scene requires an explicit selected compiler')
   if self.temporal_policy!=self.scene_compiler.temporal_policy:raise ValueError('scene compiler requires its declared temporal policy')
   if isinstance(scene,dict):
    if set(scene)-{'direction','context'}:raise ValueError('unknown directed scene field')
    scene_record=self.scene_compiler.compile(scene['direction'],text,prior,context=scene.get('context'))
   else:scene_record=self.scene_compiler.compile(scene,text,prior)
   if not scene_record['admitted']:raise UnresolvedRealization('unresolved scene reality: '+json.dumps(scene_record['unresolved']))
   scene=scene_record['scene']
  reference=None
  if self.conditioning=='selected_anchor' or (not previous and self.cold_start=='selected_anchor_icl'):
   from .native import anchor_reference
   reference=anchor_reference(self.root)
  if previous and self.conditioning=='previous_carrier':
   parent_path=self.directory/previous['receipt'];parent=json.loads(parent_path.read_text());prior_audio=parent_path.parent/'carrier.wav'
   if not parent.get('quality_admitted') or 'baseline' not in parent or not prior_audio.exists() or sha(prior_audio)!=parent['baseline']['audio']['sha256']:raise UnresolvedRealization('previous native carrier context cannot be verified')
   if prior['interaction_state']['phase']=='interrupted' and parent.get('delivery',{}).get('kind')!='verified_interrupted_native_stream':raise UnresolvedRealization('full prior carrier must not condition recovery after partial delivery')
   reference={'audio':str(prior_audio),'text':parent['delivered_plan']['text'],'sha256':sha(prior_audio),'source_carrier_sha256':ANCHOR,
    'role':'preceding verified native carrier; physical state carried separately to avoid repeated physical processing'}
  request_hash=digest({'text':text,'source_scene':source_scene or {},'scene':scene or {},'compiler':scene_record['provenance'] if scene_record else None,'seed':seed,'parent':digest(prior) if prior else None,'mechanism':self.mode,'temporal_policy':self.temporal_policy,'conditioning':self.conditioning,'reference':reference,'consonant_precision':self.articulation,'cold_start':self.cold_start,'respiration':self.respiration,'prominence':self.prominence,'prominence_calibration_sha256':sha(self.prominence_calibration) if self.prominence else None})
  target=self.directory/request_id
  if target.exists():raise FileExistsError('request already exists; inspect its persisted receipt rather than regenerate or double-commit')
  target.mkdir();draft=self.compile(text,scene,prior)
  if scene_record:durable_json(target/'scene_compilation.json',scene_record)
  carrier=target/'carrier.wav';render(self.root,text,carrier,seed,reference=reference)
  base=self.evaluator.evaluate(carrier,text)
  durable_json(target/'carrier.evaluation.json',base)
  if not base['quality_screen_pass']:
   durable_json(target/'receipt.json',{'request_id':request_id,'request_hash':request_hash,'parent_state_hash':digest(prior) if prior else None,
     'role':'rejected diagnostic carrier','quality_admitted':False,'state_committed':False,'full_completion':False,
     'baseline':base,'error':'carrier quality gate failed'})
   raise UnresolvedRealization('carrier quality gate failed')
  if self.mode=='physical':return self._render_physical(request_id,request_hash,target,carrier,base,text,scene,prior,previous,scene_record)
  reference=base;trials=[]
  for attempt in range(3):
   plan=self.compile(text,scene,prior,timeline=timeline_from_evaluation(reference))
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
           'unresolved_channels':unresolved_channels(draft),'full_completion':False,'bank_sha256':sha(self.bank_path),
           'scene_compilation':scene_record}
  if admitted:
   delivered=self.compile(text,scene,prior,timeline=timeline_from_evaluation(final['evaluation']))
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

 def admit_interrupted_delivery(self,request_id,plan,audio,stream_receipt,heard_words,diagnostic=False):
  """Commit only an independently verified prefix delivered by StreamingTurn.

  The original intention is retained in the receipt. It cannot enter the
  acoustic reference or advance events beyond the delivered prefix.
  """
  from .interaction import commit_interrupted_prefix
  from ..causal_v4.runtime import verify_plan
  if not diagnostic:raise UnresolvedRealization('ordinary-English completion admission remains closed')
  if self.mode!='physical' or self.aligner is None:raise ValueError('interrupted admission requires verified physical-session alignment')
  if not request_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in request_id):raise ValueError('invalid request id')
  with (self.directory/'session.lock').open('a') as lock:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:raise RuntimeError('session already rendering')
   try:
    self.recover_commit();previous=json.loads(self.state_path.read_text()) if self.state_path.exists() else None
    prior=previous['state'] if previous else None
    if previous and sha(self.directory/previous['receipt'])!=previous['receipt_sha256']:raise RuntimeError('committed evidence changed')
    if not verify_plan(plan) or plan['temporal_policy']!=self.temporal_policy:raise ValueError('unverified source plan or policy')
    expected=self.compile(plan['text'],plan['scene'],prior)
    if expected['initial_state']!=plan['initial_state']:raise ValueError('interruption belongs to a different session state')
    audio=pathlib.Path(audio);stream_receipt=pathlib.Path(stream_receipt);stream=json.loads(stream_receipt.read_text())
    if not stream.get('interruption') or stream['audio_sha256']!=sha(audio) or stream['delivered_samples_after_interrupt']!=0:raise ValueError('unverified interruption boundary')
    command=stream['command']
    if command[command.index('--text')+1]!=plan['text']:raise ValueError('streamed text differs from source plan')
    from ..causal_v4.runtime import words
    if not 1<=heard_words<=len(plan['words']):raise ValueError('invalid delivered word count')
    prefix=plan['text'][:words(plan['text'])[heard_words-1].end()]
    result=self.evaluator.evaluate(audio,prefix)
    if not result['quality_screen_pass'] or result['wer']!=0:raise UnresolvedRealization('interrupted prefix failed independent quality/transcription')
    if result['audio']['frames']!=stream['delivered_samples'] or stream['interruption']['heard_samples']!=stream['delivered_samples']:raise ValueError('delivered sample boundary mismatch')
    alignment=self.aligner.align(self.evaluator.load(audio),prefix,sha(audio),True)
    delivered=commit_interrupted_prefix(plan,heard_words,timeline_from_evaluation(dict(result,alignment=alignment)),stream['interruption']['source'])
    target=self.directory/request_id
    if target.exists():raise FileExistsError('interruption request already exists')
    target.mkdir();shutil.copyfile(audio,target/'carrier.wav');shutil.copyfile(stream_receipt,target/'stream.receipt.json')
    receipt={'request_id':request_id,'role':'diagnostic actual interrupted delivery','full_completion':False,
      'quality_admitted':True,'parent_state_hash':digest(prior) if prior else None,'baseline':result,'alignment':alignment,
      'intended_plan':plan,'delivered_plan':delivered,'unresolved_channels':unresolved_channels(delivered),
      'delivery':{'kind':'verified_interrupted_native_stream','source_receipt_sha256':sha(stream_receipt),
                  'heard_words':heard_words,'delivered_samples':stream['delivered_samples'],'future_reference_excluded':True}}
    saved={'scope':'diagnostic interrupted session continuity','last_request':request_id,'state':delivered['final_state'],
      'audio_sha256':sha(audio),'parent_record_hash':digest(previous) if previous else None}
    self.commit(previous,saved,target/'receipt.json',receipt);return receipt
   finally:fcntl.flock(lock,fcntl.LOCK_UN)

 def _render_physical(self,request_id,request_hash,target,carrier,base,text,scene,prior,previous,scene_record=None):
  from .physical_finality import realize as finality
  from .physical import realize as body
  from .alignment import AlignmentRejected
  receipt={'request_id':request_id,'request_hash':request_hash,'parent_state_hash':digest(prior) if prior else None,
   'role':'diagnostic stateful physical realization, not completed Mari performance','mechanism':'sample-aligned physical',
   'full_completion':False,'quality_admitted':False,'baseline':base,'scene_compilation':scene_record,'requested_scene':copy.deepcopy(scene or {}),
   'conditioning_policy':self.conditioning,'native_carrier_receipt_sha256':sha(carrier.with_suffix('.receipt.json'))}
  try:
   alignment=self.aligner.align(self.evaluator.load(carrier),text,sha(carrier),base['wer']==0)
   aligned=dict(base,asr_alignment=base['alignment'],alignment=alignment)
   timeline=timeline_from_evaluation(aligned);respiratory_receipt=None
   expected_frames=base['audio']['frames']
   prominence_receipt=None
   if self.prominence:
    from .prominence_duration import realize as emphasize
    prominence_plan=self.compile(text,scene,prior,timeline=timeline)
    prominent=target/'prominence.wav';prominence_receipt=emphasize(carrier,prominent,prominence_plan,self.prominence_calibration)
    receipt['prominence']=prominence_receipt
    quality=self.evaluator.evaluate(prominent,text)
    lexical_verified=self._verify_lexical_quality(prominent,text,quality,prominence_receipt['mapped_timeline'])
    if not lexical_verified:raise UnresolvedRealization('scoped prominence quality gate failed')
    observed_prominence=self.aligner.align(self.evaluator.load(prominent),text,sha(prominent),True)
    clock_error=max(abs(a[k]-b[k]) for a,b in zip(prominence_receipt['mapped_timeline']['words'],observed_prominence['words']) for k in ['start','end'])
    receipt['prominence_evaluation']={'quality':quality,'alignment':observed_prominence,'clock_error_s':clock_error}
    if clock_error>.08:raise UnresolvedRealization('prominence clock mapping failed')
    carrier=prominent;timeline=timeline_from_evaluation(dict(quality,alignment=observed_prominence));expected_frames+=prominence_receipt['added_samples']
   if self.respiration:
    from .respiration import realize as breathe
    respiratory_carrier=target/'respiratory_carrier.wav'
    respiratory_receipt=breathe(carrier,respiratory_carrier,text,scene,prior,timeline,policy=self.temporal_policy)
    receipt['respiration']=respiratory_receipt
    carrier=respiratory_carrier;timeline=respiratory_receipt['delivered_timeline'];scene=respiratory_receipt['delivered_scene']
    expected_frames+=respiratory_receipt['added_samples']
   plan=self.compile(text,scene,prior,timeline=timeline)
   contour=target/'contour.wav';contour_receipt=finality(carrier,contour,plan)
   # Both physical mechanisms preserve sample count. Rebind provenance, never
   # alter a measured boundary to make a failed alignment test pass.
   body_clock=dict(timeline,source_audio_sha256=sha(contour));body_plan=self.compile(text,scene,prior,timeline=body_clock)
   audio=target/'performance.wav';body_output=target/'body.wav' if self.articulation else audio
   body_receipt=body(contour,body_output,body_plan)
   articulation_receipt=None
   if self.articulation:
    from .articulation import realize as articulate
    precision_plan=self.compile(text,scene,prior,timeline=dict(timeline,source_audio_sha256=sha(body_output)))
    articulation_receipt=articulate(body_output,audio,precision_plan)
   result=self.evaluator.evaluate(audio,text)
   if result['audio']['frames']!=expected_frames:raise UnresolvedRealization('physical mechanism changed the measured clock')
   lexical_verified=self._verify_lexical_quality(audio,text,result,dict(timeline,source_audio_sha256=sha(audio)))
   observed=self.aligner.align(self.evaluator.load(audio),text,sha(audio),lexical_verified)
   error=max(abs(a[k]-b[k]) for a,b in zip(timeline['words'],observed['words']) for k in ['start','end'])
   admitted=lexical_verified and error<=.08
   unresolved=set(unresolved_channels(plan))-{'phonatory_tension','attack_softness','gain_db'}
   if self.articulation and 'precision' in unresolved:
    unresolved.remove('precision');unresolved.add('perceived_articulatory_precision')
   if prominence_receipt:
    unresolved.discard('emphasis');unresolved.add('perceived_dramatic_appropriateness_of_contrast')
   if respiratory_receipt:
    unresolved.discard('support');unresolved.add('respiratory_physiology_and_perceived_timing')
   receipt.update(quality_admitted=bool(admitted),unresolved_channels=sorted(unresolved),alignment=alignment,
    trials=[{'audio':str(audio),'evaluation':result,'alignment_error_s':error,'observed_alignment':observed,
             'finality':contour_receipt,'body':body_receipt,'consonant_precision':articulation_receipt}])
   if not admitted:raise UnresolvedRealization('physical diagnostic quality gate failed')
   delivered=self.compile(text,scene,prior,timeline=timeline_from_evaluation(dict(result,alignment=observed)))
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

 def _verify_lexical_quality(self,audio,text,quality,timeline):
  """Keep raw ASR visible; resolve context errors with full clause coverage.

  The original full-audio identity/WER<=.12 screen remains mandatory. The
  supplementary strict transcription gate requires WER0 either on the whole
  utterance or on every measured clause, covering all samples and all words.
  """
  if not quality['quality_screen_pass']:return False
  if quality['wer']==0:return True
  from .clause_asr import verify_clauses
  try:report=verify_clauses(audio,text,timeline,self.evaluator)
  except ValueError as error:
   quality['clause_ASR_verification']={'admitted':False,'error':str(error)};return False
  quality['clause_ASR_verification']=report
  return bool(report['admitted']and report['all_samples_covered_once']and report['all_words_covered_once'])
