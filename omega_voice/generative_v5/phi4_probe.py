"""Official Phi audio input, exact weights, and inspectable inference receipts."""
import pathlib,json,time,hashlib,importlib.metadata,collections
import numpy as np,soundfile as sf,torch
from transformers import AutoModelForCausalLM,AutoProcessor
from .session import durable_json
from ..causal_v4.renderer import sha

class Phi4Probe:
 def __init__(self,root):
  self.root=pathlib.Path(root);e=self.root/'continuation/phi4_evaluator';d=pathlib.Path('/tmp/mari-phi4');m=json.loads((e/'MODEL_PROVENANCE.json').read_text())
  if not m['complete']or m['revision']!='93f923e1a7727d1c4f446756212d9d3e8fcc5d81':raise ValueError('unselected evaluator source')
  for row in m['files']:
   if sha(d/row['name'])!=row['sha256']:raise ValueError('evaluator file changed: '+row['name'])
  torch.set_num_threads(4);torch.manual_seed(98600)
  self.processor=AutoProcessor.from_pretrained(d,trust_remote_code=True,local_files_only=True)
  self.model,info=AutoModelForCausalLM.from_pretrained(d,trust_remote_code=True,local_files_only=True,device_map='cpu',torch_dtype=torch.bfloat16,_attn_implementation='eager',low_cpu_mem_usage=True,output_loading_info=True)
  durable_json(e/'LOADING_INFO.json',info)
  if any(info.get(k)for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('incomplete evaluator tensor load')
  self.model.eval();dtypes=collections.Counter(str(p.dtype)for p in self.model.parameters());meta=[n for n,p in self.model.named_parameters()if p.is_meta]
  if meta:raise ValueError('unmaterialized evaluator weights')
  self.provenance={'repository':m['repository'],'revision':m['revision'],'model_manifest_sha256':sha(e/'MODEL_PROVENANCE.json'),'source_sha256':sha(__file__),'device':'cpu','dtype_counts':dict(dtypes),'attention':'eager','loading_info':info,'dependencies':{n:importlib.metadata.version(n)for n in ['torch','transformers','peft','accelerate','torchvision','backoff','soundfile','numpy']},'audio_interface':'official processor audios=[(actual decoded PCM, actual sample rate)] and audio_1 placeholder','evaluator_not_qualified_yet':True}
  durable_json(e/'RUNTIME_PROVENANCE.json',self.provenance)

 def infer(self,path,question,max_new_tokens=100):
  x,sr=sf.read(path,dtype='float32')
  if x.ndim!=1 or not np.isfinite(x).all()or not len(x):raise ValueError('invalid input audio')
  prompt='<|user|><|audio_1|>'+question+'<|end|><|assistant|>';inputs=self.processor(text=prompt,audios=[(x,sr)],return_tensors='pt')
  audio=inputs.get('input_audio_embeds')
  if audio is None or not audio.numel()or not torch.isfinite(audio).all():raise ValueError('actual audio absent from evaluator')
  audit={'audio_sha256':sha(path),'decoded_PCM_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'sample_rate':sr,'frames':len(x),'prompt':prompt,'processor_tensor_shapes':{k:list(v.shape)for k,v in inputs.items()if torch.is_tensor(v)},'audio_features_sha256':hashlib.sha256(audio.float().numpy().tobytes()).hexdigest(),'audio_features_min':float(audio.min()),'audio_features_max':float(audio.max()),'input_mode':inputs['input_mode'].tolist(),'generation':{'do_sample':False,'max_new_tokens':max_new_tokens}}
  inputs={k:(v.to(torch.bfloat16)if torch.is_tensor(v)and v.is_floating_point()else v)for k,v in inputs.items()};started=time.monotonic()
  # Official prepare_inputs_for_generation defaults this to None, whereas
  # forward requires an integer. Generation needs only final-token logits.
  audit['generation']['num_logits_to_keep']=1
  with torch.inference_mode():output=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,num_logits_to_keep=1)
  generated=output[:,inputs['input_ids'].shape[1]:];response=self.processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
  return dict(audit,response=response,elapsed_s=time.monotonic()-started,generated_tokens=generated.shape[1],truncated=generated.shape[1]>=max_new_tokens)
