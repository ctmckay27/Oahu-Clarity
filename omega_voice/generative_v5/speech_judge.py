"""Pinned speech-specific pairwise critic. Qualification precedes any use.

Naturalness, prosody, pacing and articulation are its declared scope. Its scores
never establish Mari identity, cognitive truth, or character-specific presence.
"""
import argparse,json,pathlib,hashlib,re,time
import numpy as np,torch,soundfile as sf
from scipy.signal import resample_poly
from math import gcd

def sha(path):
 with pathlib.Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def conversation(text,a,b):
 return [
 {'role':'system','content':[{'type':'text','text':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'}]},
 {'role':'user','content':[
  {'type':'text','text':"We are comparing the naturalness of two Text-to-Speech models' outputs. The models need to generate the target text."},
  {'type':'text','text':'Target text: '+text},{'type':'text','text':'Output A:'},{'type':'audio','audio':a},
  {'type':'text','text':'Output B:'},{'type':'audio','audio':b},
  {'type':'text','text':'Analysis the two output above, and score them with number from 1 to 10.'},
  {'type':'text','text':'Note: (1) Please evaluate the naturalness of both audio outputs based on the following criteria: Prosody and Intonation, Pacing and Rhythm, Articulation and Clarity, and Overall Naturalness. (2) After conducting a detailed analysis of each criterion, using the following output template to highlight your conclusion: Output A: X, Output B: X.'}]}]

def waveform(path):
 y,sr=sf.read(path,dtype='float32',always_2d=True);y=y.mean(1);g=gcd(sr,16000)
 return resample_poly(y,16000//g,sr//g).astype(np.float32)

def main():
 from transformers import Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniThinkerConfig,Qwen2_5OmniProcessor
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--provenance',required=True);ap.add_argument('--jobs',required=True);ap.add_argument('--output',required=True);ap.add_argument('--max-tokens',type=int,default=700);ap.add_argument('--official-processor',action='store_true');a=ap.parse_args()
 torch.set_num_threads(5);torch.set_num_interop_threads(1)
 d=pathlib.Path(a.model);manifest=json.loads(pathlib.Path(a.provenance).read_text())
 if manifest['revision']!='8d4ce4ddf54c5e7ac0b457fc7b561cec1bd784c1' or manifest['status']!='complete':raise ValueError('unselected critic dependency')
 for f in manifest['files']:
  if f['path']=='model.safetensors.index.json':continue
  if sha(d/f['path'])!=f['sha256']:raise ValueError('critic bytes changed '+f['path'])
 if sha(d/'model.safetensors.index.json')!=manifest['derived_index_sha256']:raise ValueError('critic index changed')
 cfg=Qwen2_5OmniThinkerConfig.from_pretrained(d)
 # The released text configuration and separate checkpoint tensors specify
 # untied input/output embeddings. Preserve that at the Thinker config level.
 cfg.tie_word_embeddings=cfg.text_config.tie_word_embeddings
 processor=Qwen2_5OmniProcessor.from_pretrained(d)
 model,loading=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(d,config=cfg,torch_dtype=torch.bfloat16,
  device_map='cpu',attn_implementation='sdpa',output_loading_info=True)
 if loading['missing_keys'] or loading['unexpected_keys'] or loading['mismatched_keys']:raise ValueError('incomplete critic load: '+str(loading))
 model.eval();print('critic loaded',sum(p.numel() for p in model.parameters()),flush=True)
 rows=[]
 for job in json.loads(pathlib.Path(a.jobs).read_text()):
  class Progress:
   def __init__(self):self.ids=[];self.first=True;self.started=time.monotonic()
   def put(self,value):
    if self.first:self.first=False;return
    self.ids.extend(value.reshape(-1).tolist())
    if len(self.ids)%20==0:
     pathlib.Path(a.output+'.progress.json').write_text(json.dumps({'id':job['id'],'tokens':len(self.ids),'elapsed_s':time.monotonic()-self.started,'partial_response':processor.tokenizer.decode(self.ids,skip_special_tokens=True)},indent=2))
   def end(self):pass
  started=time.monotonic();messages=conversation(job['text'],job['a'],job['b'])
  prompt=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
  if a.official_processor:
   from qwen_omni_utils import process_mm_info
   audios,images,videos=process_mm_info(messages,use_audio_in_video=False)
   inputs=processor(text=prompt,audio=audios,images=images,videos=videos,return_tensors='pt',padding=True,use_audio_in_video=False)
  else:inputs=processor(text=prompt,audio=[waveform(job['a']),waveform(job['b'])],return_tensors='pt',padding=True,use_audio_in_video=False)
  features={'tokens':int((inputs.input_ids==cfg.audio_token_id).sum()),'lengths':inputs.feature_attention_mask.sum(1).tolist(),
   'input_features_sha256':hashlib.sha256(inputs.input_features.numpy().tobytes()).hexdigest()}
  audio_forward=[]
  def observe_audio(module,args,output):
   tensor=output.last_hidden_state if hasattr(output,'last_hidden_state') else output[0]
   audio_forward.append({'shape':list(tensor.shape),'finite':bool(torch.isfinite(tensor).all()),'rms':float(tensor.float().square().mean().sqrt())})
  hook=model.audio_tower.register_forward_hook(observe_audio)
  for k,v in inputs.items():
   if torch.is_floating_point(v):inputs[k]=v.to(torch.bfloat16)
  with torch.inference_mode():out=model.generate(**inputs,use_audio_in_video=False,do_sample=False,max_new_tokens=a.max_tokens,eos_token_id=[151645],pad_token_id=151643,streamer=Progress())
  hook.remove()
  decoded=processor.batch_decode(out[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
  match=re.findall(r'Output A:\s*(\d+(?:\.\d+)?).*?Output B:\s*(\d+(?:\.\d+)?)',decoded.replace('**',''),re.S)
  score={'a':float(match[-1][0]),'b':float(match[-1][1])} if match else None
  row={'id':job['id'],'input':job,'audio_hashes':{'a':sha(job['a']),'b':sha(job['b'])},'scores':score,
   'model_response':decoded,'elapsed_s':time.monotonic()-started,'model_revision':manifest['revision'],
   'decoding':'greedy, CPU BF16 SDPA; source prompt retained','scope':'unqualified machine judgment; no character-completion inference',
   'generated_tokens':int(out.shape[1]-inputs['input_ids'].shape[1]),'official_processor':a.official_processor,
   'audio_input':features,'audio_encoder_forward':audio_forward}
  rows.append(row);pathlib.Path(a.output).write_text(json.dumps(rows,indent=2)+'\n');print(job['id'],score,decoded[-700:],flush=True)
if __name__=='__main__':main()
