"""Longer annotated vocal events: identity and family remain separate gates."""
import argparse,pathlib,json,requests,hashlib
import numpy as np,soundfile as sf
from .session import durable_json
from .vocal_event_evaluator import VocalEventEvaluator
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/laughter_reference';examples=json.loads((d/'REFERENCES.json').read_text())['examples'];c=next(x for x in examples if x['speaker']=='C');start,end=268.,273.99;sr=c['source_header']['sample_rate'];offset=c['source_header']['offset'];p=d/'C.enrollment.wav'
 durable_json(d/'QUALIFICATION_PREDECLARED.json',{'C_enrollment_interval':[start,end],'enrollment_source':'first continuous C speaking span of at least4s in the recovered annotation; A/B/D fixed prior enrollment controls reused with their hashes','positive_gate':'all six longer laughs choose their annotated speaker among A/B/C/D/Mari, with margin>=.03 and same-speaker similarity>=.60','event_family_gate':'all annotated laughs have adult laughter-family probability>=.5; selected carrier and silence have family probability<.2','scope':'small longer-event evaluator qualification only, no Mari voice selection, perception or acting admission','full_completion':False})
 if not p.exists():
  lo=offset+round(start*sr)*2;hi=offset+round(end*sr)*2-1;res=requests.get(c['source_url'],headers={'Range':f'bytes={lo}-{hi}'},timeout=60);res.raise_for_status()
  if res.status_code!=206 or len(res.content)!=hi-lo+1 or res.headers.get('ETag')!=c['source_etag']:raise ValueError('enrollment source/range mismatch')
  sf.write(p,np.frombuffer(res.content,dtype='<i2'),sr,subtype='PCM_16');durable_json(d/'C.enrollment.source.json',{'url':c['source_url'],'source_etag':c['source_etag'],'content_range':res.headers.get('Content-Range'),'raw_pcm_sha256':hashlib.sha256(res.content).hexdigest(),'audio_sha256':sha(p),'interval':[start,end],'license':'CC BY4.0','attribution':'AMI Meeting Corpus','role':'evaluator reference only'})
 if sha(p)!=json.loads((d/'C.enrollment.source.json').read_text())['audio_sha256']:raise ValueError('enrollment changed')
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');enroll={}
 for who in ['A','B','D']:
  path=r/'continuation/event_identity'/(who+'.enrollment.wav');prov=json.loads(path.with_name(who+'.source.json').read_text());assert sha(path)==prov['audio_sha256'];enroll[who]=ev.embedding(path)
 enroll['C']=ev.embedding(p);enroll['Mari']=ev.anchor;rows=[]
 for case in examples:
  assert sha(case['path'])==case['sha256'];e=ev.embedding(case['path']);scores={k:float(e@v)for k,v in enroll.items()};expected=case['speaker'];margin=scores[expected]-max(v for k,v in scores.items()if k!=expected);row={'id':case['id'],'expected':expected,'choice':max(scores,key=scores.get),'scores':scores,'margin':margin,'identity_gate':margin>=.03 and scores[expected]>=.60};rows.append(row);durable_json(d/'IDENTITY_RESULTS.json',rows);print(case['id'],row['choice'],margin,scores[expected],flush=True)
 family={'Laughter','Chuckle, chortle','Giggle','Snicker','Belly laugh'};event=VocalEventEvaluator('/tmp/mari-ast',r/'continuation/vocal_event_evaluator/MODEL_PROVENANCE.json');events=[]
 controls=examples+[{'id':'carrier_negative','path':str(r/'continuation/calibration/unchanged.wav')},{'id':'silence_negative','path':str(r/'continuation/backchannel_reference/silence.wav')}]
 for c in controls:
  q=event.evaluate(c['path']);scores=dict(q['vocal_events']);scores.update({x['label']:x['probability']for x in q['top']});value=max([scores.get(k,0.)for k in family]);positive=not c['id'].endswith('_negative');row={'id':c['id'],'family_probability':value,'expected_laughter':positive,'pass':value>=.5 if positive else value<.2,'result':q};events.append(row);durable_json(d/'FAMILY_RESULTS.json',events);print(c['id'],value,row['pass'],flush=True)
 durable_json(d/'ASSESSMENT.json',{'identity_qualified':all(x['identity_gate']for x in rows),'identity_positive_count':sum(x['identity_gate']for x in rows),'family_qualified':all(x['pass']for x in events),'family_results_correct':sum(x['pass']for x in events),'no_character_admission':True,'full_completion':False})
if __name__=='__main__':main()
