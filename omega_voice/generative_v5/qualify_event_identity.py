"""Test ECAPA short backchannel identification against annotated AMI speakers.

The full-sentence Mari similarity gate must not silently stand in for this
different duration/phonetic scope. These references are evaluator controls.
"""
import argparse,pathlib,json,requests,hashlib
import numpy as np,soundfile as sf
from .session import durable_json
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);source=r/'continuation/backchannel_reference';d=r/'continuation/event_identity';d.mkdir(exist_ok=True)
 examples=json.loads((source/'HUMAN_BACKCHANNELS.json').read_text())['examples']
 intervals={'A':[132.57,141.98],'B':[55.98,67.55],'D':[82.08,86.5]}
 durable_json(d/'PREDECLARED_SCOPE.json',{'references':intervals,'reference_selection':'continuous annotated speaking spans before first selected backchannel; no model score selection',
  'gate':'all six short human events choose their annotated speaker among three enrollment speakers and Mari, with positive margin; zero controls must not pass any established positive margin',
  'scope':'small narrow qualification only; not absolute Mari similarity threshold or listener recognition','full_completion':False})
 records=[]
 for agent,(start,end) in intervals.items():
  case=next(x for x in examples if x['id'].split('.')[1]==agent);url=case['source_url'];p=d/(agent+'.enrollment.wav');record=d/(agent+'.source.json')
  if not p.exists():
   lo=44+round(start*16000)*2;hi=44+round(end*16000)*2-1
   response=requests.get(url,headers={'Range':f'bytes={lo}-{hi}'},timeout=60);response.raise_for_status()
   if response.status_code!=206 or len(response.content)!=hi-lo+1:raise ValueError('incomplete annotated reference')
   if response.headers.get('ETag')!=case['source_etag']:raise ValueError('AMI source changed since event retrieval')
   sf.write(p,np.frombuffer(response.content,dtype='<i2'),16000,subtype='PCM_16')
   durable_json(record,{'source_url':url,'interval_s':[start,end],'source_etag':response.headers.get('ETag'),'content_range':response.headers.get('Content-Range'),
    'raw_pcm_sha256':hashlib.sha256(response.content).hexdigest(),'audio_sha256':sha(p),'license':'CC BY4.0','attribution':'AMI Meeting Corpus, https://groups.inf.ed.ac.uk/ami/corpus/',
    'annotation_sha256':sha(source/f'annotations/words/ES2002a.{agent}.words.xml'),'channel_mapping_sha256':case['channel_mapping_source_sha256'],'role':'speaker-evaluator control only'})
  if sha(p)!=json.loads(record.read_text())['audio_sha256']:raise ValueError('enrollment bytes changed')
  records.append({'id':agent,'path':str(p),'sha256':sha(p)})
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 enroll={x['id']:ev.embedding(x['path']) for x in records};enroll['Mari']=ev.anchor
 rows=[]
 for case in examples:
  if sha(case['path'])!=case['sha256']:raise ValueError('event source changed')
  embedding=ev.embedding(case['path']);scores={k:float(embedding@v) for k,v in enroll.items()};expected=case['id'].split('.')[1];choice=max(scores,key=scores.get)
  margin=scores[expected]-max(v for k,v in scores.items() if k!=expected)
  rows.append({'id':case['id'],'expected':expected,'choice':choice,'scores':scores,'margin':margin,'correct':choice==expected and margin>0})
  durable_json(d/'RESULTS.json',rows);print(case['id'],choice,margin,flush=True)
 passed=all(x['correct'] for x in rows)
 durable_json(d/'ASSESSMENT.json',{'all_human_positive_identifications_correct':passed,'count':sum(x['correct'] for x in rows),
  'qualified':False,'reason':'negative controls and absolute threshold still required' if passed else 'fails short-event human positives; do not use full-sentence threshold for event admission',
  'full_completion':False})
if __name__=='__main__':main()
