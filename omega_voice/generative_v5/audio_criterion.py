"""One audio, one criterion, with auditable label probabilities and controls.

No rating is admitted without qualification. Numeric expectations are machine
estimates, not listening claims or an aggregate voice-completion score.
"""
import argparse,json,pathlib,hashlib,time
import torch,librosa
from .download_pinned_audio_evaluator import REV,sha

def main():
 from transformers import AutoProcessor,Qwen2AudioForConditionalGeneration
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--provenance',required=True);ap.add_argument('--jobs',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 torch.set_num_threads(5);torch.set_num_interop_threads(1);d=pathlib.Path(a.model);manifest=json.loads(pathlib.Path(a.provenance).read_text())
 if manifest['revision']!=REV or not manifest['complete']:raise ValueError('unselected evaluator')
 for item in manifest['files']:
  if sha(d/item['path'])!=item['sha256']:raise ValueError('evaluator dependency changed')
 processor=AutoProcessor.from_pretrained(d)
 model,loading=Qwen2AudioForConditionalGeneration.from_pretrained(d,torch_dtype=torch.bfloat16,device_map='cpu',attn_implementation='sdpa',output_loading_info=True)
 if loading['missing_keys'] or loading['unexpected_keys'] or loading['mismatched_keys']:raise ValueError('evaluator load mismatch '+str(loading))
 model.eval();rows=[];print('loaded',sum(p.numel() for p in model.parameters()),flush=True)
 for job in json.loads(pathlib.Path(a.jobs).read_text()):
  started=time.monotonic();audio_path=job['audio'];wave,sr=librosa.load(audio_path,sr=processor.feature_extractor.sampling_rate)
  messages=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':[{'type':'audio','audio_url':audio_path},{'type':'text','text':job['question']}]}]
  prompt=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
  inputs=processor(text=prompt,audios=[wave],return_tensors='pt',padding=True)
  for k,v in inputs.items():
   if torch.is_floating_point(v):inputs[k]=v.to(torch.bfloat16)
  row={'id':job['id'],'job':job,'audio_sha256':sha(audio_path),'model_revision':REV,'scope':'unqualified machine estimate','audio_features_shape':list(inputs['input_features'].shape)}
  if job.get('labels'):
   labels=job['labels'];ids=[processor.tokenizer.encode(x,add_special_tokens=False) for x in labels]
   if any(len(x)!=1 for x in ids):raise ValueError('criterion labels must each be one token')
   with torch.inference_mode():result=model(**inputs);logits=result.logits[0,-1].float();probs=logits.softmax(-1);chosen=logits[torch.tensor([x[0] for x in ids])].softmax(-1)
   row.update(label_probabilities={label:float(prob) for label,prob in zip(labels,chosen)},allowed_probability_mass=float(probs[[x[0] for x in ids]].sum()),top_token=processor.tokenizer.decode([int(logits.argmax())]))
   if all(x.isdigit() for x in labels):row['expected_score']=sum(float(label)*float(prob) for label,prob in zip(labels,chosen))
   del result,logits,probs,chosen
  else:
   with torch.inference_mode():result=model.generate(**inputs,do_sample=False,max_new_tokens=job.get('max_tokens',48))
   row['response']=processor.batch_decode(result[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0];del result
  row['elapsed_s']=time.monotonic()-started;rows.append(row);pathlib.Path(a.output).write_text(json.dumps(rows,indent=2)+'\n');print(job['id'],{k:row[k] for k in ['response','label_probabilities','allowed_probability_mass','expected_score'] if k in row},flush=True)
if __name__=='__main__':main()
