"""Pinned SpeechJudge thinker-only evaluation checkpoint, without conversion.

Omit unused speech generation weights. This model never renders Mari's voice.
Large contiguous transfers avoid repeated small-range connection overhead.
"""
import argparse,hashlib,json,pathlib,struct
from concurrent.futures import ThreadPoolExecutor
import requests
REV='8d4ce4ddf54c5e7ac0b457fc7b561cec1bd784c1'
REPO='RMSnow/SpeechJudge-GRM'
def digest(p):
 with pathlib.Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('directory');ap.add_argument('receipt');a=ap.parse_args()
 d=pathlib.Path(a.directory);d.mkdir(parents=True,exist_ok=True);root=f'https://huggingface.co/{REPO}/resolve/{REV}/'
 files=['config.json','model.safetensors.index.json','added_tokens.json','chat_template.json','generation_config.json','merges.txt','preprocessor_config.json','special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json']
 records=[]
 for name in files:
  p=d/name
  if not p.exists():
   response=requests.get(root+name,timeout=90);response.raise_for_status();p.write_bytes(response.content)
  records.append({'path':name,'sha256':digest(p),'bytes':p.stat().st_size})
 original=json.loads((d/'model.safetensors.index.json').read_text());weight_map={k:v for k,v in original['weight_map'].items() if k.startswith('thinker.')}
 def download(name):
  retained=[k for k,v in weight_map.items() if v==name];all_keys=[k for k,v in original['weight_map'].items() if v==name]
  p=d/name;temp=d/(name+'.partial')
  if len(retained)==len(all_keys):
   with requests.get(root+name,stream=True,timeout=180) as response:
    response.raise_for_status()
    with temp.open('wb') as f:
     for chunk in response.iter_content(4*1024**2):f.write(chunk)
   with temp.open('rb') as f:
    n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
   if set(header)-{'__metadata__'}!=set(retained):raise ValueError('unexpected tensor names')
   total=max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
   if temp.stat().st_size!=8+n+total:raise ValueError('truncated checkpoint')
   kind='exact original shard'
  else:
   def ranged(start,end):
    resp=requests.get(root+name+f'?mari_range={start}-{end}',headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=180)
    if resp.status_code!=206 or not resp.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):
     resp.close();raise ValueError('server did not honor requested tensor range')
    return resp
   with ranged(0,7) as response:n=struct.unpack('<Q',response.content)[0]
   with ranged(8,7+n) as response:header=response.json()
   selected=sorted([(k,header[k]) for k in retained],key=lambda kv:kv[1]['data_offsets'][0]);derived={'__metadata__':{'format':'pt'}};total=0;groups=[]
   for k,v in selected:
    start,end=v['data_offsets'];derived[k]=dict(v,data_offsets=[total,total+end-start]);total+=end-start
    if groups and groups[-1][1]==start:groups[-1][1]=end
    else:groups.append([start,end])
   hb=json.dumps(derived,separators=(',',':')).encode();hb+=b' '*((-len(hb))%8)
   with temp.open('wb') as f:
    f.write(struct.pack('<Q',len(hb)));f.write(hb)
    for start,end in groups:
     with ranged(8+n+start,8+n+end-1) as response:
      got=0
      for chunk in response.iter_content(4*1024**2):f.write(chunk);got+=len(chunk)
     if got!=end-start:raise ValueError('truncated tensor range')
   kind='exact thinker tensors; unchanged names and numerical bytes; speech generator omitted'
  temp.replace(p);record={'path':name,'sha256':digest(p),'bytes':p.stat().st_size,'tensor_bytes':total,'derivation':kind}
  print(name,record['bytes'],flush=True);return record
 shards=list(ThreadPoolExecutor(2).map(download,sorted(set(weight_map.values()))));records.extend(shards)
 source_index=d/'source.model.safetensors.index.json'
 if not source_index.exists():source_index.write_text(json.dumps(original,indent=2))
 (d/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':sum(x['tensor_bytes'] for x in shards)},'weight_map':weight_map},indent=2))
 receipt={'model':REPO,'revision':REV,'source_commit':'1fbd1c883b03f3f7437506b421f7cf57bad78071','role':'internal evaluation only; not a speaker or renderer','license':'cc-by-nc-4.0','files':records,'derived_index_sha256':digest(d/'model.safetensors.index.json'),'status':'complete'}
 pathlib.Path(a.receipt).write_text(json.dumps(receipt,indent=2)+'\n');print('COMPLETE',flush=True)
if __name__=='__main__':main()
