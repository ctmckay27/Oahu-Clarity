"""Human annotated backchannels for evaluator qualification, never voice cloning."""
import argparse,pathlib,json,re,struct,requests,hashlib
import xml.etree.ElementTree as ET
import numpy as np,soundfile as sf

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();d=pathlib.Path(a.root);ann=d/'annotations';ns={'n':'http://nite.sourceforge.net/'};nid='{'+ns['n']+'}id'
 meeting=next(x for x in ET.parse(ann/'corpusResources/meetings.xml').getroot() if x.get('observation')=='ES2002a')
 channels={x.get('nxt_agent'):x.get('channel') for x in meeting}
 selection=(d/'SIGNAL_SELECTION.html').read_text();session=requests.Session();rows=[]
 for agent in ['A','B','D']:
  source_url=re.search(r'https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/[^\s\\\'<]*Headset-'+channels[agent]+r'\.wav',selection).group(0)
  response=session.get(source_url,headers={'Range':'bytes=0-4095'},timeout=50);response.raise_for_status()
  if response.status_code!=206:raise ValueError('audio range request ignored')
  header=response.content
  if header[:4]!=b'RIFF' or header[8:12]!=b'WAVE':raise ValueError('not RIFF/WAVE')
  pos=12;data_offset=None;fmt=None
  while pos+8<=len(header):
   kind=header[pos:pos+4];size=struct.unpack_from('<I',header,pos+4)[0]
   if kind==b'fmt ':fmt=struct.unpack_from('<HHIIHH',header,pos+8)
   if kind==b'data':data_offset=pos+8;break
   pos+=8+size+(size%2)
  if fmt is None or data_offset is None or fmt[0]!=1 or fmt[1]!=1 or fmt[5]!=16:raise ValueError('unqualified source PCM format')
  sr=fmt[2];acts=[]
  for act in ET.parse(ann/f'dialogueActs/ES2002a.{agent}.dialog-act.xml').getroot():
   ptr=act.find('n:pointer',ns);child=act.find('n:child',ns)
   if ptr is None or child is None or not ptr.get('href','').endswith('id(ami_da_1)'):continue
   ids=re.findall(r'id\(([^)]+)\)',child.get('href'));acts.append((int(ids[0].split('words')[-1]),int(ids[-1].split('words')[-1]),act.get(nid)))
  candidates=[]
  for word in ET.parse(ann/f'words/ES2002a.{agent}.words.xml').getroot():
   label=(word.text or '').lower().strip();id=word.get(nid,'')
   if label not in {'hmm','mm','mm-hmm','mhm'}:continue
   idx=int(id.split('words')[-1]);act=next((v for lo,hi,v in acts if lo<=idx<=hi),None)
   if not act:continue
   start=float(word.get('starttime'));end=float(word.get('endtime'))
   if .12<=end-start<=.6:candidates.append((id,label,start,end,act))
  for id,label,start,end,act in candidates[:2]:
   first=round(max(0,start-.05)*sr);last=round((end+.05)*sr);lo=data_offset+2*first;hi=data_offset+2*last-1
   res=session.get(source_url,headers={'Range':f'bytes={lo}-{hi}'},timeout=50);res.raise_for_status()
   if res.status_code!=206 or len(res.content)!=2*(last-first):raise ValueError('incomplete audio range')
   y=np.frombuffer(res.content,dtype='<i2');p=d/(id+'.wav');sf.write(p,y,sr,subtype='PCM_16')
   rows.append({'id':id,'annotation':label,'dialogue_act_id':act,'dialogue_act':'Backchannel (ami_da_1)',
    'source_url':source_url,'source_content_range':res.headers.get('Content-Range'),'source_etag':res.headers.get('ETag'),
    'word_start_s':start,'word_end_s':end,'extracted_start_sample':first,'extracted_end_sample':last,'sample_rate':sr,
    'channel_mapping_source_sha256':hashlib.sha256((ann/'corpusResources/meetings.xml').read_bytes()).hexdigest(),
    'source_pcm_sha256':hashlib.sha256(res.content).hexdigest(),'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
   (d/'HUMAN_BACKCHANNELS.json').write_text(json.dumps({'selection':'first two duration .12-.6s, annotated nasal backchannels per A/B/D in ES2002a; fixed before model scores',
    'license':'CC BY 4.0','attribution':'AMI Meeting Corpus, https://groups.inf.ed.ac.uk/ami/corpus/','role':'evaluator positives only; no speaker selection or synthesis reference','examples':rows},indent=2)+'\n')
   print(id,label,end-start,flush=True)
if __name__=='__main__':main()
