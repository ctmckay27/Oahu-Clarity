"""Fetch exact public dependencies for isolated timbre-conversion qualification."""
import pathlib,json,hashlib,requests,shutil,argparse
SPECS={
 'Plachta/Seed-VC':('257283f9f41585055e8f858fba4fd044e5caed6e','cfm',['v2/cfm_small.pth']),
 'Plachta/ASTRAL-quantization':('4a2e9679f76eb03753adc8c503e3c23bb9c22f26','astral',['bsq2048/bsq2048_light.pth']),
 'facebook/hubert-large-ll60k':('ff022d095678a2995f3c49bab18a96a9e553f782','hubert',['config.json','preprocessor_config.json','pytorch_model.bin']),
 'nvidia/bigvgan_v2_22khz_80band_256x':('633ff708ed5b74903e86ff1298cf4a98e921c513','vocoder',['config.json','bigvgan_generator.pt']),
 'funasr/campplus':('e4b6ede7ce16997aff4ae69fbca1f0175e2afede','speaker',['campplus_cn_common.bin']),
 'openai/whisper-small':('973afd24965f72e36ca33b3055d56a652f456b4d','tokenizer',['added_tokens.json','config.json','merges.txt','normalizer.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json'])}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();e=pathlib.Path(a.root)/'continuation/seed_vc_inspection';dst=pathlib.Path('/tmp/mari-seed-vc-models');dst.mkdir(exist_ok=True)
 work=[]
 for repo,(rev,folder,names)in SPECS.items():
  meta=json.loads((e/(repo.replace('/','__')+'_API.json')).read_text());assert meta['sha']==rev
  for name in names:
   row=next(x for x in meta['siblings']if x['rfilename']==name);work.append((repo,rev,folder,name,row))
 required=sum(x[4]['size']for x in work if not(dst/x[2]/pathlib.Path(x[3]).name).exists());reserve=4*1024**3;free=shutil.disk_usage(dst).free
 if free<required+reserve:raise RuntimeError('insufficient space for pinned dependency set plus 4GiB reserve')
 plan={'required_bytes':required,'free_bytes':free,'reserve_bytes':reserve,'purpose':'isolated conversion diagnostic, fixed Mari target; no voice selection or production fallback'};(e/'DOWNLOAD_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
 result=[]
 for repo,rev,folder,name,meta in work:
  p=dst/folder/pathlib.Path(name).name;p.parent.mkdir(exist_ok=True)
  if not p.exists():
   response=requests.get(f'https://huggingface.co/{repo}/resolve/{rev}/{name}',stream=True,timeout=60);response.raise_for_status();temp=p.with_suffix(p.suffix+'.partial')
   try:
    with temp.open('wb')as f:
     for chunk in response.iter_content(4*1024**2):
      if shutil.disk_usage(dst).free<reserve+len(chunk):raise RuntimeError('disk reserve')
      f.write(chunk)
    data=temp.read_bytes();assert len(data)==meta['size']
    if meta.get('lfs'):assert hashlib.sha256(data).hexdigest()==meta['lfs']['sha256']
    else:assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==meta['blobId']
    temp.rename(p)
   except BaseException:
    temp.unlink(missing_ok=True);raise
  h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
  if meta.get('lfs'):assert h==meta['lfs']['sha256']
  result.append({'repository':repo,'revision':rev,'source_path':name,'path':str(p),'bytes':p.stat().st_size,'sha256':h})
  (e/'MODEL_PROVENANCE.json').write_text(json.dumps({'files':result,'complete':len(result)==len(work),'scope':plan['purpose']},indent=2)+'\n');print(folder,p.name,p.stat().st_size,flush=True)
if __name__=='__main__':main()
