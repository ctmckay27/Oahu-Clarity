"""Recover the exact public Qwen2-Audio evaluator; never a voice backend."""
import argparse,hashlib,json,pathlib,requests,time
REPO='Qwen/Qwen2-Audio-7B-Instruct'
REV='0a095220c30b7b31434169c3086508ef3ea5bf0a'
def sha(path):
 with pathlib.Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('source');ap.add_argument('destination');ap.add_argument('receipt');a=ap.parse_args()
 meta=json.loads(pathlib.Path(a.source).read_text());assert meta['sha']==REV and meta['id']==REPO
 d=pathlib.Path(a.destination);d.mkdir(exist_ok=True);records=[]
 for item in meta['siblings']:
  name=item['rfilename']
  if name.startswith('.') or not name.endswith(('.json','.txt','.md','.safetensors')):continue
  target=d/name;expected=item.get('lfs',{}).get('sha256')
  if target.exists() and target.stat().st_size==item['size'] and (not expected or sha(target)==expected):pass
  else:
   for attempt in range(3):
    try:
     url=f'https://huggingface.co/{REPO}/resolve/{REV}/{name}?download=true&cache={time.time_ns()}'
     with requests.get(url,stream=True,timeout=(30,60)) as res:
      res.raise_for_status()
      if res.status_code!=200:raise ValueError('unexpected partial response')
      tmp=target.with_suffix(target.suffix+'.part')
      with tmp.open('wb') as f:
       for chunk in res.iter_content(4*1024*1024):f.write(chunk)
     if tmp.stat().st_size!=item['size']:raise ValueError('size mismatch')
     if expected and sha(tmp)!=expected:raise ValueError('hash mismatch')
     tmp.replace(target);break
    except Exception as e:
     print(name,'attempt',attempt+1,type(e).__name__,str(e)[:160],flush=True)
     if attempt==2:raise
  records.append({'path':name,'bytes':target.stat().st_size,'sha256':sha(target),'source_git_blob':item.get('blobId')})
  pathlib.Path(a.receipt).write_text(json.dumps({'repository':REPO,'revision':REV,'files':records,'complete':False},indent=2)+'\n');print(name,target.stat().st_size,flush=True)
 pathlib.Path(a.receipt).write_text(json.dumps({'repository':REPO,'revision':REV,'files':records,'complete':True,'role':'local audio evaluator; no synthesis or speaker substitution'},indent=2)+'\n')
if __name__=='__main__':main()
