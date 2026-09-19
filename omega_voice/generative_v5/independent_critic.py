"""Independent AF3 audio evaluation. Requires qualification; never self-certifies."""
import argparse,json,pathlib,time
def main():
 import torch,librosa
 from transformers import AudioFlamingo3ForConditionalGeneration,AutoProcessor
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--jobs',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 torch.set_num_threads(5);torch.set_num_interop_threads(1)
 model=AudioFlamingo3ForConditionalGeneration.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cpu',low_cpu_mem_usage=True,attn_implementation='sdpa').eval()
 processor=AutoProcessor.from_pretrained(a.model)
 print('INDEPENDENT_AUDIO_CRITIC_LOADED',flush=True);records=[]
 for job in json.loads(pathlib.Path(a.jobs).read_text()):
  # Inputs contain the sound and an observation question, never its desired label.
  y,sr=librosa.load(job['audio'],sr=16000)
  conversation=[{'role':'user','content':[{'type':'audio','audio':job['audio']},{'type':'text','text':job['question']}]}]
  prompt=processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True)
  inputs=processor(text=prompt,audio=[y],sampling_rate=sr,return_tensors='pt').to('cpu',dtype=torch.bfloat16)
  start=time.monotonic()
  with torch.inference_mode():ids=model.generate(**inputs,max_new_tokens=job.get('max_tokens',80),do_sample=False)
  answer=processor.batch_decode(ids[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
  row={'id':job['id'],'audio':job['audio'],'question':job['question'],'answer':answer,'elapsed_s':time.monotonic()-start,'evidence_type':'independent audio-language model prediction, not human listening','qualified':False}
  records.append(row);pathlib.Path(a.output).write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(row),flush=True)
if __name__=='__main__':main()
