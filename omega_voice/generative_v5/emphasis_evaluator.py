"""Pinned EmphaClass acoustic prominence observer; requires qualification.

This measures word emphasis, never the appropriateness of a dramatic choice,
Mari identity, cognition, naturalness, or character presence.
"""
import json,pathlib
import numpy as np
from ..causal_v4.renderer import sha

class EmphasisEvaluator:
 def __init__(self,root):
  import torch,torchaudio,transformers
  from transformers import Wav2Vec2ForAudioFrameClassification
  self.torch=torch;self.ta=torchaudio;torch.set_num_threads(3)
  self.root=pathlib.Path(root);self.directory=pathlib.Path('/tmp/mari-emphaclass')
  self.manifest=json.loads((self.root/'continuation/emphasis_inspection/MODEL_PROVENANCE.json').read_text())
  if not self.manifest['complete']:raise ValueError('incomplete emphasis weights')
  for row in self.manifest['files']:
   if sha(row['path'])!=row['sha256']:raise ValueError('emphasis provenance mismatch')
  self.model,info=Wav2Vec2ForAudioFrameClassification.from_pretrained(self.directory,local_files_only=True,output_loading_info=True,torch_dtype=torch.float32,weights_only=True)
  if any(info.get(k)for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('inexact emphasis loader '+repr(info))
  self.model.eval();self.provenance={'source_revision':'29ba1fa9b1bca35c9f47338cbb18b82d48ee576b','manifest':self.manifest,'loads':info,'torch':torch.__version__,'torchaudio':torchaudio.__version__,'transformers':transformers.__version__,'preprocessing':'official mono raw waveform, torchaudio resample16000, no normalization, no truncation','source_sha256':sha(__file__),'scope':'English acoustic emphasis observer pending qualification; not a character judge'}
 def infer(self,path):
  torch=self.torch;y,sr=self.ta.load(path);y=y.mean(0,keepdim=True)
  if sr!=16000:y=self.ta.functional.resample(y,sr,16000)
  if y.shape[1]<1600 or y.shape[1]>16000*20 or not torch.isfinite(y).all():raise ValueError('unsupported emphasis audio')
  with torch.inference_mode():logits=self.model(y).logits[0];prob=logits.softmax(-1)[:,1].numpy();labels=logits.argmax(-1).numpy()
  intervals=[];start=None
  for i,on in enumerate(labels):
   if on==1 and start is None:start=i*.02
   if on!=1 and start is not None:intervals.append([start,i*.02]);start=None
  if start is not None:intervals.append([start,len(labels)*.02])
  return {'audio_sha256':sha(path),'seconds':y.shape[1]/16000,'frame_s':.02,'emphasized_intervals':intervals,'positive_frame_fraction':float(np.mean(labels==1)),'probability':prob.tolist(),'interpretation':'acoustic emphasis only'}

def word_observations(result,alignment):
 if result['audio_sha256']!=alignment['source_sha256'] or not alignment['admitted']:raise ValueError('unverified emphasis word clock')
 out=[];p=np.asarray(result['probability'])
 for i,w in enumerate(alignment['words']):
  start,end=w['start'],w['end'];overlap=sum(max(0,min(end,b)-max(start,a))for a,b in result['emphasized_intervals'])
  indices=np.arange(len(p))*.02+.01;mask=(indices>=start)&(indices<end)
  out.append({'index':i,'word':w['word'],'start':start,'end':end,'overlap_fraction':overlap/(end-start),'emphasized':overlap>(end-start)*.5,'mean_probability':float(p[mask].mean())if mask.any()else None})
 return out
