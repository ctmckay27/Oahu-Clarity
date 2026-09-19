"""Independent WavLM short-event identity qualification, no waveform padding."""
import argparse,pathlib,json,torch,librosa,numpy as np
from transformers import Wav2Vec2FeatureExtractor,WavLMForXVector
from .session import durable_json
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/wavlm_event_identity';modeldir=pathlib.Path('/tmp/mari-wavlm-sv')
 meta=json.loads((d/'MODEL_PROVENANCE.json').read_text())
 if not meta['complete'] or meta['revision']!='feb593a6c23c1cc3d9510425c29b0a14d2b07b1e' or not all(sha(modeldir/x['path'])==x['sha256'] for x in meta['files']):raise ValueError('speaker evaluator provenance mismatch')
 if tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])<(2,6):raise ValueError('patched torch loader required for official pickle weights')
 torch.set_num_threads(4);processor=Wav2Vec2FeatureExtractor.from_pretrained(modeldir);model=WavLMForXVector.from_pretrained(modeldir).eval()
 def embed(p):
  y,sr=librosa.load(p,sr=16000);inputs=processor(y,sampling_rate=sr,return_tensors='pt')
  with torch.inference_mode():v=model(**inputs).embeddings[0].numpy()
  if not np.isfinite(v).all():raise ValueError('nonfinite event embedding')
  return v/np.linalg.norm(v)
 durable_json(d/'PREDECLARED_SCOPE.json',{'gate':'all six annotated events correctly identified with positive margin using unmodified waveforms and same enrollment as ECAPA; no padding/repetition to rescue short events','full_completion':False})
 enroll={x:embed(r/f'continuation/event_identity/{x}.enrollment.wav') for x in 'ABD'};enroll['Mari']=embed(r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');rows=[]
 for case in json.loads((r/'continuation/backchannel_reference/HUMAN_BACKCHANNELS.json').read_text())['examples']:
  if sha(case['path'])!=case['sha256']:raise ValueError('human control changed')
  try:
   v=embed(case['path']);scores={k:float(v@w) for k,w in enroll.items()};expected=case['id'].split('.')[1];choice=max(scores,key=scores.get);margin=scores[expected]-max(v for k,v in scores.items() if k!=expected)
   row={'id':case['id'],'expected':expected,'choice':choice,'scores':scores,'margin':margin,'correct':choice==expected and margin>0}
  except Exception as exc:row={'id':case['id'],'correct':False,'error':str(exc)}
  rows.append(row);durable_json(d/'RESULTS.json',rows);print(row,flush=True)
 durable_json(d/'ASSESSMENT.json',{'positive_count':sum(x['correct'] for x in rows),'qualified':False,'full_completion':False,
  'reason':'negative controls and threshold qualification pending' if all(x['correct'] for x in rows) else 'failed short-event positives, cannot admit identity'})
if __name__=='__main__':main()
