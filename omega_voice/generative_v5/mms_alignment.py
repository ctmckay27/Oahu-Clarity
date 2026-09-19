"""Pinned alignment-trained model, independent of renderer and ASR admission."""
import json,pathlib
import torch,torchaudio
from .alignment import AlignmentRejected
from ..causal_v4.evaluate import normalized
from ..causal_v4.renderer import sha

class MMSAligner:
 def __init__(self,receipt):
  self.receipt=json.loads(pathlib.Path(receipt).read_text());p=pathlib.Path(self.receipt['path'])
  if sha(p)!=self.receipt['sha256']:raise ValueError('MMS weights changed')
  self.bundle=torchaudio.pipelines.MMS_FA;self.model=self.bundle.get_model(with_star=False,dl_kwargs={'model_dir':str(p.parent),'progress':False}).eval();self.tokenizer=self.bundle.get_tokenizer();self.aligner=self.bundle.get_aligner()

 def align(self,wave16,text,source_hash,transcript_verified):
  if not transcript_verified:raise ValueError('alignment cannot verify transcript')
  words=normalized(text);tokens=self.tokenizer(words)
  with torch.inference_mode():emission,_=self.model(torch.as_tensor(wave16,dtype=torch.float32)[None]);spans=self.aligner(emission[0],tokens)
  step=len(wave16)/(16000*emission.shape[1]);results=[]
  for w,ss in zip(words,spans):
   if len(w)!=len(ss):raise ValueError('romanized token mismatch')
   total=sum(s.end-s.start for s in ss);score=sum(s.score*(s.end-s.start)for s in ss)/total
   results.append({'word':w,'start':float(ss[0].start*step),'end':float(ss[-1].end*step),'probability':float(score),'characters':[{'letter':c,'start':float(s.start*step),'end':float(s.end*step),'probability':float(s.score)}for c,s in zip(w,ss)]})
  minimum=min(w['probability']for w in results);report={'source_sha256':source_hash,'words':results,'exact_words':True,'method':'MMS romanized forced alignment of independently supplied transcript','transcript_verification_separate':True,'model_sha256':self.receipt['sha256'],'frame_period_s':step,'minimum_word_probability':minimum,'admitted':minimum>=.35,'threshold':.35}
  if not report['admitted']:raise AlignmentRejected(report)
  return report
