"""Interruptible native PCM delivery with auditable heard/unspoken boundaries.

An output callback is the delivery boundary. Interrupting stops both delivery
and generation. State commitment is limited to actually delivered words;
future scene events cannot leak into the resumed state.
"""
import copy,json,os,pathlib,subprocess,threading,time,wave
from .native import verify_runtime
from ..causal_v4.runtime import words,compile_scene,PROFILE,ANCHOR
from ..causal_v4.renderer import sha

class StreamingTurn:
 def __init__(self,root,text,output,on_audio,seed=88000):
  self.root=pathlib.Path(root);self.text=text;self.output=pathlib.Path(output);self.on_audio=on_audio;self.seed=seed
  self.lock=threading.RLock();self.stopped=False;self.interruption=None;self.process=None;self.delivered=[];self.error=None

 def interrupt(self,source):
  if not isinstance(source,dict) or not source.get('text'):raise ValueError('interruption requires its observed cause')
  with self.lock:
   if self.stopped:return
   self.interruption={'source':copy.deepcopy(source),'heard_samples':sum(map(len,self.delivered))//2,'requested_monotonic_s':time.monotonic()}
   self.stopped=True
   if self.process and self.process.poll() is None:self.process.terminate()

 def run(self):
  if self.output.exists():raise FileExistsError(self.output)
  # Validate text and runtime before a byte can be delivered.
  compile_scene(self.text);lock=verify_runtime(self.root);p=self.root/'recovered/production_release/production';e=self.root/'continuation/engine'
  if sha(p/'MARI_VOICE_V1_PROFILE.bin')!=PROFILE or sha(p/'MARI_VOICE_V1_ANCHOR.wav')!=ANCHOR:raise ValueError('identity mismatch')
  cmd=[str(e/'qwen_tts'),'-d',str(self.root/'models/qwen-custom'),'--load-voice',str(p/'MARI_VOICE_V1_PROFILE.bin'),'--xvector-only','-l','English','--text',self.text,'--seed',str(self.seed),'--temperature','.42','--top-k','40','--top-p','.95','--rep-penalty','1.05','-j4','--stdout','--stream-chunk','2']
  env={k:v for k,v in os.environ.items() if not k.startswith(('MARI_','QWEN_'))}
  self.output.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic()
  with self.output.with_suffix('.log').open('wb') as log:
   with self.lock:
    if self.stopped:raise RuntimeError('turn interrupted before generation began')
    self.process=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=log)
   try:
    while True:
     chunk=self.process.stdout.read(1920) # 40 ms delivery packets, signed 16-bit mono
     if not chunk:break
     with self.lock:
      if self.stopped:break
      if len(chunk)%2:raise RuntimeError('malformed PCM packet')
      self.delivered.append(chunk);self.on_audio(chunk,self)
    self.process.wait(timeout=10)
   except BaseException as exc:
    self.error=str(exc)
    if self.process.poll() is None:self.process.kill();self.process.wait()
    raise
  if not self.interruption and self.process.returncode:raise RuntimeError('native streaming renderer failed')
  with wave.open(str(self.output),'wb') as f:
   f.setnchannels(1);f.setsampwidth(2);f.setframerate(24000);f.writeframes(b''.join(self.delivered))
  receipt={'scope':'native streaming delivery; no resumed-phonation equivalence claim','command':cmd,'runtime':lock,
           'audio_sha256':sha(self.output),'delivered_samples':sum(map(len,self.delivered))//2,
           'interruption':self.interruption,'process_returncode':self.process.returncode,'elapsed_s':time.monotonic()-started,
           'delivered_samples_after_interrupt':0}
  self.output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt

def commit_interrupted_prefix(plan,heard_words,timeline,source):
 if not 1<=heard_words<=len(plan['words']):raise ValueError('heard prefix must be verified')
 matches=words(plan['text']);prefix=plan['text'][:matches[heard_words-1].end()]
 scene=copy.deepcopy(plan['scene']);scene['events']=[e for e in scene.get('events',[]) if e['at_word']<heard_words]
 scene['directions']=[e for e in scene.get('directions',[]) if e.get('at_word',0)<heard_words]
 scene['events'].append({'id':'delivery-interruption','at_word':heard_words,'kind':'interrupt','source':source})
 result=compile_scene(prefix,scene,plan['initial_state'],timeline=timeline,policy=plan.get('temporal_policy'))
 return result
