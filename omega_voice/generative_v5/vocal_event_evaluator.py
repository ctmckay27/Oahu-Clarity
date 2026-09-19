"""Independent acoustic-event classifier; never a character/personhood judge."""
import argparse,json,pathlib
import numpy as np,librosa,torch
from ..causal_v4.renderer import sha
from .session import durable_json

class VocalEventEvaluator:
 def __init__(self,model_dir,provenance):
  from transformers import AutoFeatureExtractor,AutoModelForAudioClassification
  self.directory=pathlib.Path(model_dir);self.provenance=json.loads(pathlib.Path(provenance).read_text())
  if self.provenance['revision']!='f826b80d28226b62986cc218e5cec390b1096902':raise ValueError('unselected event evaluator')
  for row in self.provenance['files']:
   if sha(self.directory/row['path'])!=row['sha256']:raise ValueError('event evaluator bytes changed')
  torch.set_num_threads(3);self.extractor=AutoFeatureExtractor.from_pretrained(self.directory)
  self.model=AutoModelForAudioClassification.from_pretrained(self.directory).eval()

 def evaluate(self,path):
  y,sr=librosa.load(path,sr=16000);inputs=self.extractor(y,sampling_rate=sr,return_tensors='pt')
  with torch.inference_mode():logits=self.model(**inputs).logits[0];prob=logits.sigmoid().numpy()
  order=np.argsort(prob)[::-1]
  wanted={'Speech','Female speech, woman speaking','Speech synthesizer','Laughter','Chuckle, chortle','Sigh','Humming','Breathing','Cough','Silence','Clock','Tick','Tick-tock'}
  return {'audio_sha256':sha(path),'model_revision':self.provenance['revision'],'top':[{ 'label':self.model.config.id2label[int(i)],'probability':float(prob[i])} for i in order[:8]],
   'vocal_events':{self.model.config.id2label[int(i)]:float(prob[i]) for i in order if self.model.config.id2label[int(i)] in wanted},
   'scope':'audio-event machine estimates; no character, cause, naturalness or identity judgment'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/vocal_event_evaluator'
 ev=VocalEventEvaluator('/tmp/mari-ast',d/'MODEL_PROVENANCE.json');rows=[]
 for case in json.loads((d/'DATA_PROVENANCE.json').read_text())['examples']:
  q=ev.evaluate(case['path']);rows.append({'expected':case['category'],'path':case['path'],'result':q});durable_json(d/'QUALIFICATION_RESULTS.json',rows);print(case['category'],q['top'][:2],flush=True)
 for name,path in [('carrier',r/'continuation/calibration/unchanged.wav'),('silence',r/'continuation/speechjudge_qualification/silence.wav')]:
  q=ev.evaluate(path);rows.append({'expected':name,'path':str(path),'result':q});durable_json(d/'QUALIFICATION_RESULTS.json',rows);print(name,q['top'][:2],flush=True)
if __name__=='__main__':main()
