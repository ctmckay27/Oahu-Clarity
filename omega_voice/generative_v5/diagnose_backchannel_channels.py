"""Inspect cross-talk as a competing cause of short-event identity errors.

AMI headset labels are annotations, not proof of acoustic isolation. This does
not delete/relabel a failed case and does not admit a synthetic event.
"""
import argparse,pathlib,json,requests,hashlib,re,struct
import numpy as np,soundfile as sf
import xml.etree.ElementTree as ET
from scipy.signal import correlate,correlation_lags
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);src=r/'continuation/backchannel_reference';dst=r/'continuation/backchannel_crosstalk';dst.mkdir(exist_ok=True)
 cases=json.loads((src/'HUMAN_BACKCHANNELS.json').read_text())['examples'];s=requests.Session();channels={'A':0,'B':1,'C':2,'D':3};rows=[]
 for case in cases:
  signals={};overlap={};first,last=case['extracted_start_sample'],case['extracted_end_sample'];sr=case['sample_rate']
  for agent,ch in channels.items():
   url=re.sub(r'Headset-\d+\.wav',f'Headset-{ch}.wav',case['source_url']);p=dst/(case['id']+'.channel_'+agent+'.wav');prov=p.with_suffix('.json')
   if not p.exists():
    lo=44+2*first;hi=44+2*last-1;res=s.get(url,headers={'Range':f'bytes={lo}-{hi}'},timeout=50);res.raise_for_status()
    if res.status_code!=206 or len(res.content)!=2*(last-first):raise ValueError('incomplete range')
    pcm=np.frombuffer(res.content,dtype='<i2');sf.write(p,pcm,sr,subtype='PCM_16')
    prov.write_text(json.dumps({'source_url':url,'etag':res.headers.get('ETag'),'content_range':res.headers.get('Content-Range'),'pcm_sha256':sha(res.content),'wave_sha256':sha(p.read_bytes()),'extracted_samples':[first,last],'annotation_source':case['id'],'license':'CC BY4.0','attribution':'AMI Meeting Corpus, https://groups.inf.ed.ac.uk/ami/corpus/','role':'evaluator cross-talk diagnostic only'},indent=2)+'\n')
   assert sha(p.read_bytes())==json.loads(prov.read_text())['wave_sha256'];x,rate=sf.read(p);assert rate==sr;signals[agent]=x
   words=ET.parse(src/f'annotations/words/ES2002a.{agent}.words.xml').getroot()
   overlap[agent]=[{'text':w.text,'start':float(w.get('starttime')),'end':float(w.get('endtime'))}for w in words if w.get('starttime') and float(w.get('starttime'))<last/sr and float(w.get('endtime'))>first/sr]
  own=case['id'].split('.')[1];y=signals[own];diagnostic={}
  for agent,x in signals.items():
   cc=correlate(y-y.mean(),x-x.mean(),mode='full');lag=correlation_lags(len(y),len(x));keep=np.abs(lag)<=round(.02*sr);i=np.argmax(np.abs(cc[keep]));den=np.linalg.norm(y-y.mean())*np.linalg.norm(x-x.mean())
   diagnostic[agent]={'rms':float(np.sqrt(np.mean(x*x))),'max_abs_correlation_with_labeled_channel':float(abs(cc[keep][i])/max(den,1e-12)),'lag_samples':int(lag[keep][i])}
  row={'id':case['id'],'labeled_speaker':own,'overlapping_annotations':overlap,'signals':diagnostic,'scope':'correlation is cross-talk evidence only; no relabeling or identity admission'};rows.append(row);(dst/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n');print(case['id'],diagnostic,flush=True)
if __name__=='__main__':main()
