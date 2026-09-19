"""Pinned independent audio-understanding evaluator; never a voice backend."""
import argparse,concurrent.futures,hashlib,json,pathlib,time
import requests
MODEL='nvidia/audio-flamingo-3-hf'
REV='7d4bae64ee29878af6504ae6f6bb3e40492838ad'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('destination');a=ap.parse_args();dest=pathlib.Path(a.destination);dest.mkdir(parents=True,exist_ok=True)
 files=requests.get(f'https://huggingface.co/api/models/{MODEL}/tree/{REV}?recursive=true',timeout=60).json()
 selected=[p for p in files if p['type']=='file' and '/' not in p['path'] and (p['path'].endswith(('.json','.txt','.jinja')) or p['path'].startswith('model-'))]
 def fetch(p):
  target=dest/p['path'];partial=target.with_suffix(target.suffix+'.partial')
  for attempt in range(4):
   try:
    if not target.exists() or target.stat().st_size!=p['size']:
     with requests.get(f'https://huggingface.co/{MODEL}/resolve/{REV}/{p["path"]}',stream=True,timeout=(30,120)) as r:
      r.raise_for_status()
      with partial.open('wb') as f:
       for chunk in r.iter_content(4*1024*1024):f.write(chunk)
     if partial.stat().st_size!=p['size']:raise RuntimeError('download length mismatch')
     partial.replace(target)
    h=hashlib.sha256()
    with target.open('rb') as f:
     for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    result={'path':p['path'],'size':p['size'],'sha256':h.hexdigest()}
    expected=p.get('lfs',{}).get('oid')
    if expected and expected!=result['sha256']:raise RuntimeError('download hash mismatch')
    print(json.dumps(result),flush=True);return result
   except Exception:
    if attempt==3:raise
    time.sleep(1+attempt)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(fetch,selected))
 (dest/'MARI_EVALUATOR_PROVENANCE.json').write_text(json.dumps({'model':MODEL,'revision':REV,'role':'independent internal audio evaluator, not production speech backend','files':records},indent=2)+'\n')
if __name__=='__main__':main()
