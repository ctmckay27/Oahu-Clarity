"""Published longer vocal-event controls; no new Mari speaker selection."""
import argparse,pathlib,json,re,struct,requests
import xml.etree.ElementTree as ET
import numpy as np,soundfile as sf
from .session import durable_json
from ..causal_v4.renderer import sha
import hashlib

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);source=r/'continuation/backchannel_reference';d=r/'continuation/laughter_reference';d.mkdir(exist_ok=True);ann=source/'annotations';nid='{http://nite.sourceforge.net/}id';meeting=next(x for x in ET.parse(ann/'corpusResources/meetings.xml').getroot()if x.get('observation')=='ES2002a');channels={x.get('nxt_agent'):x.get('channel')for x in meeting};selection=(source/'SIGNAL_SELECTION.html').read_text();cases=[]
 for agent in ['A','C','D']:
  eligible=[]
  for e in ET.parse(ann/f'words/ES2002a.{agent}.words.xml').getroot():
   if e.tag!='vocalsound'or e.get('type')!='laugh':continue
   start=float(e.get('starttime'));end=float(e.get('endtime'))
   if 1.<=end-start<=4.:eligible.append({'id':e.get(nid),'speaker':agent,'start':start,'end':end,'annotation':'laugh','annotation_sha256':sha(ann/f'words/ES2002a.{agent}.words.xml')})
  cases+=eligible[:2]
 durable_json(d/'PREDECLARED.json',{'cases':cases,'selection':'first two explicitly annotated1-4second laughs per A/C/D in ES2002a; B has zero-duration annotations and cannot supply bounded controls','purpose':'test whether longer situated vocal gestures support event-family and speaker-continuity evaluation where subsecond backchannels did not','license':'CC BY4.0','attribution':'AMI Meeting Corpus, https://groups.inf.ed.ac.uk/ami/corpus/','Mari_voice_selection':False,'full_completion':False})
 session=requests.Session();headers={};rows=[]
 for c in cases:
  agent=c['speaker'];url=re.search(r'https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/[^\s\\\'<]*Headset-'+channels[agent]+r'\.wav',selection).group(0)
  if agent not in headers:
   res=session.get(url,headers={'Range':'bytes=0-4095'},timeout=50);res.raise_for_status();b=res.content
   if res.status_code!=206 or b[:4]!=b'RIFF'or b[8:12]!=b'WAVE':raise ValueError('unverified WAVE range')
   pos=12;fmt=None;offset=None
   while pos+8<=len(b):
    kind=b[pos:pos+4];size=struct.unpack_from('<I',b,pos+4)[0]
    if kind==b'fmt ':fmt=struct.unpack_from('<HHIIHH',b,pos+8)
    if kind==b'data':offset=pos+8;break
    pos+=8+size+size%2
   if fmt is None or offset is None or fmt[0]!=1 or fmt[1]!=1 or fmt[5]!=16:raise ValueError('unqualified PCM source')
   headers[agent]={'sample_rate':fmt[2],'offset':offset,'etag':res.headers.get('ETag'),'header_sha256':hashlib.sha256(b).hexdigest()}
  h=headers[agent];sr=h['sample_rate'];first=round((c['start']-.05)*sr);last=round((c['end']+.05)*sr);lo=h['offset']+2*first;hi=h['offset']+2*last-1;res=session.get(url,headers={'Range':f'bytes={lo}-{hi}'},timeout=50);res.raise_for_status()
  if res.status_code!=206 or len(res.content)!=2*(last-first)or res.headers.get('ETag')!=h['etag']:raise ValueError('inconsistent source range')
  p=d/(c['id']+'.wav');sf.write(p,np.frombuffer(res.content,dtype='<i2'),sr,subtype='PCM_16');rows.append(dict(c,path=str(p),sha256=sha(p),source_url=url,source_content_range=res.headers.get('Content-Range'),source_etag=h['etag'],extracted_samples=[first,last],source_pcm_sha256=hashlib.sha256(res.content).hexdigest(),source_header=h));durable_json(d/'REFERENCES.json',{'examples':rows,'license':'CC BY4.0','attribution':'AMI Meeting Corpus','channel_mapping_sha256':sha(ann/'corpusResources/meetings.xml'),'no_Mari_identity_selection':True});print(c['id'],c['end']-c['start'],flush=True)
if __name__=='__main__':main()
