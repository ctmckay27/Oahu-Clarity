"""Pinned public local audio-QA evaluator route; never a speech renderer."""
import argparse,pathlib,json,hashlib,shutil,concurrent.futures
import requests
from .session import durable_json

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);e=r/'continuation/voxtral_inspection';d=pathlib.Path('/tmp/mari-voxtral');d.mkdir(exist_ok=True);rev='3060fe34b35ba5d44202ce9ff3c097642914f8f3';repo='mistralai/Voxtral-Mini-3B-2507';tree=json.loads((e/'MODEL_TREE.json').read_text());names={'config.json','generation_config.json','model-00001-of-00002.safetensors','model-00002-of-00002.safetensors','model.safetensors.index.json','preprocessor_config.json','tekken.json'};files=[x for x in tree if x['path']in names]
 if {x['path']for x in files}!=names:raise ValueError('incomplete pinned model index')
 remaining=sum(x['size']for x in files if not(d/x['path']).exists())
 if shutil.disk_usage(d).free<remaining+700_000_000:raise ValueError('insufficient measured storage for exact selected files plus headroom')
 def fetch(row):
  p=d/row['path'];h=hashlib.sha256();url=f'https://huggingface.co/{repo}/resolve/{rev}/{row["path"]}'
  if p.exists():
   with p.open('rb')as f:
    while b:=f.read(1048576):h.update(b)
  else:
   res=requests.get(url,stream=True,timeout=(30,120));res.raise_for_status();part=p.with_suffix(p.suffix+'.part')
   with part.open('wb')as f:
    for b in res.iter_content(1048576):f.write(b);h.update(b)
   if part.stat().st_size!=row['size']:raise ValueError('truncated model file')
   part.replace(p)
  if p.stat().st_size!=row['size']:raise ValueError('cached size mismatch')
  if 'lfs'in row:
   if h.hexdigest()!=row['lfs']['oid']:raise ValueError('LFS content hash mismatch')
  else:
   b=p.read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
   if blob!=row['oid']:raise ValueError('Git blob mismatch')
  result={'name':row['path'],'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'upstream_object':row};print(row['path'],result['bytes'],flush=True);return result
 with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:out=list(pool.map(fetch,files))
 durable_json(e/'MODEL_PROVENANCE.json',{'repo':repo,'revision':rev,'files':out,'complete':True,'purpose':'local audio-understanding evaluator qualification, no voice rendering or production selection','load_format':'HF safetensors shards; duplicate consolidated format intentionally not downloaded'})
if __name__=='__main__':main()
