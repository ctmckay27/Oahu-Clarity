"""Pinned official audio-understanding evaluator; no speech production route."""
import argparse,pathlib,json,hashlib,shutil,concurrent.futures
import requests
from .session import durable_json

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);e=r/'continuation/phi4_evaluator';d=pathlib.Path('/tmp/mari-phi4');d.mkdir(exist_ok=True);j=json.loads((e/'REPOSITORY.json').read_text());rev='93f923e1a7727d1c4f446756212d9d3e8fcc5d81'
 if j['sha']!=rev:raise ValueError('unselected model revision')
 names={'config.json','generation_config.json','model.safetensors.index.json','preprocessor_config.json','processor_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','merges.txt','vocab.json','README.md','LICENSE','configuration_phi4mm.py','modeling_phi4mm.py','processing_phi4mm.py','speech_conformer_encoder.py','vision_siglip_navit.py'}|{f'model-{i:05d}-of-00003.safetensors'for i in range(1,4)}
 rows=[x for x in j['siblings']if x['rfilename']in names]
 if {x['rfilename']for x in rows}!=names:raise ValueError('incomplete file manifest')
 if shutil.disk_usage(d).free<sum(x['size']for x in rows if not(d/x['rfilename']).exists())+700_000_000:raise ValueError('measured storage headroom insufficient')
 def fetch(row):
  name=row['rfilename'];p=d/name;url=f'https://huggingface.co/microsoft/Phi-4-multimodal-instruct/resolve/{rev}/{name}';h=hashlib.sha256()
  if not p.exists():
   response=requests.get(url,stream=True,timeout=(30,120));response.raise_for_status();part=p.with_suffix(p.suffix+'.part')
   with part.open('wb')as f:
    for b in response.iter_content(1048576):f.write(b);h.update(b)
   if part.stat().st_size!=row['size']:raise ValueError('truncated file: '+name)
   part.replace(p)
  else:
   with p.open('rb')as f:
    for b in iter(lambda:f.read(1048576),b''):h.update(b)
  if p.stat().st_size!=row['size']:raise ValueError('cached size mismatch')
  if 'lfs'in row:
   if h.hexdigest()!=row['lfs']['sha256']:raise ValueError('upstream SHA256 mismatch')
  else:
   b=p.read_bytes()
   if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=row['blobId']:raise ValueError('Git blob mismatch')
  out={'name':name,'path':str(p),'url':url,'bytes':p.stat().st_size,'sha256':h.hexdigest(),'upstream':row};print(name,out['bytes'],flush=True);return out
 with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:out=list(pool.map(fetch,rows))
 durable_json(e/'MODEL_PROVENANCE.json',{'repository':'microsoft/Phi-4-multimodal-instruct','revision':rev,'files':out,'complete':True,'separate_adapter_files_not_needed':'Official full checkpoint index includes vision and speech LoRA weights; no replacement or omitted tensor','purpose':'local audio evaluator qualification only','full_completion':False})
if __name__=='__main__':main()
