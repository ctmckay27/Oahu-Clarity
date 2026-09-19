"""Local audio-input critic. Predictions remain separate from human judgments."""
import argparse,json,pathlib,time
def main():
 import torch,librosa
 from transformers import Qwen2_5OmniThinkerForConditionalGeneration,Qwen2_5OmniThinkerConfig,Qwen2_5OmniProcessor
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--jobs',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 raw=json.loads((pathlib.Path(a.model)/'config.json').read_text())
 config=Qwen2_5OmniThinkerConfig(**raw['thinker_config'])
 config.tie_word_embeddings=False
 model=Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(a.model,config=config,torch_dtype=torch.bfloat16,device_map='cpu',low_cpu_mem_usage=True,attn_implementation='eager').eval()
 processor=Qwen2_5OmniProcessor.from_pretrained(a.model)
 print('AUDIO_CRITIC_LOADED',flush=True)
 records=[]
 for job in json.loads(pathlib.Path(a.jobs).read_text()):
  y,sr=librosa.load(job['audio'],sr=16000)
  conversation=[{'role':'system','content':[{'type':'text','text':'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'}]},{'role':'user','content':[{'type':'audio','audio':job['audio']},{'type':'text','text':job['question']}]}]
  text=processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=False)
  inputs=processor(text=text,audio=[y],return_tensors='pt',padding=True)
  inputs={k:v.to(dtype=torch.bfloat16) if torch.is_floating_point(v) else v for k,v in inputs.items()}
  start=time.monotonic()
  with torch.inference_mode():ids=model.generate(**inputs,max_new_tokens=job.get('max_tokens',100),do_sample=False,eos_token_id=151645,pad_token_id=151643,repetition_penalty=1.05)
  answer=processor.batch_decode(ids[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
  row={'id':job['id'],'audio':job['audio'],'question':job['question'],'answer':answer,'elapsed_s':time.monotonic()-start,'evidence_type':'local audio-language model prediction, not human judgment'}
  records.append(row);pathlib.Path(a.output).write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(row),flush=True)
if __name__=='__main__':main()
