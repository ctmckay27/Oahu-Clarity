"""Local pinned Voxtral audio-QA qualification, separate from voice rendering."""
import pathlib,json,time,os,importlib.metadata
import numpy as np,soundfile as sf
from ..causal_v4.renderer import sha
from .session import durable_json

class VoxtralProbe:
 def __init__(self,root):
  import torch,transformers
  from transformers import VoxtralForConditionalGeneration,AutoProcessor
  self.torch=torch;torch.set_num_threads(3);self.root=pathlib.Path(root);d=pathlib.Path('/tmp/mari-voxtral');e=self.root/'continuation/voxtral_inspection';manifest=json.loads((e/'MODEL_PROVENANCE.json').read_text())
  if manifest['revision']!='3060fe34b35ba5d44202ce9ff3c097642914f8f3' or not manifest['complete']:raise ValueError('unselected audio-QA model')
  for f in manifest['files']:
   if sha(f['path'])!=f['sha256']:raise ValueError('audio-QA model hash mismatch')
  if transformers.__version__!='4.54.1' or importlib.metadata.version('mistral-common')!='1.8.1':raise ValueError('unselected audio-QA dependency')
  os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';torch.manual_seed(97300)
  self.processor=AutoProcessor.from_pretrained(d,local_files_only=True,trust_remote_code=False)
  self.model,info=VoxtralForConditionalGeneration.from_pretrained(d,local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':'cpu'},attn_implementation='eager',low_cpu_mem_usage=True,output_loading_info=True,use_safetensors=True)
  if any(info.get(k)for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('inexact Voxtral weight load '+repr(info))
  self.model.eval();self.provenance={'manifest':manifest,'loads':info,'torch':torch.__version__,'transformers':transformers.__version__,'mistral-common':importlib.metadata.version('mistral-common'),'device':'cpu','dtype':'bfloat16','attention':'eager','source_sha256':sha(__file__),'role':'unqualified local audio understanding observer; no voice generation or character admission'}
 def infer(self,path,prompt):
  path=pathlib.Path(path);y,sr=sf.read(path,dtype='float32')
  if y.ndim!=1 or not np.isfinite(y).all() or not .1<=len(y)/sr<=30:raise ValueError('unsupported audio-QA waveform')
  if not isinstance(prompt,str)or not prompt.strip():raise ValueError('question required')
  start=time.monotonic();conversation=[{'role':'user','content':[{'type':'audio','path':str(path.resolve())},{'type':'text','text':prompt}]}]
  inputs=self.processor.apply_chat_template(conversation);inputs=inputs.to('cpu',dtype=self.torch.bfloat16)
  with self.torch.inference_mode():outputs=self.model.generate(**inputs,max_new_tokens=160,do_sample=False,use_cache=True)
  result=self.processor.batch_decode(outputs[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
  return {'audio_sha256':sha(path),'question':prompt,'response':result,'elapsed_s':time.monotonic()-start,'input_tokens':int(inputs.input_ids.shape[1]),'new_tokens':int(outputs.shape[1]-inputs.input_ids.shape[1]),'evaluator_provenance':self.provenance,'character_admitted':False}
