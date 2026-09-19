"""Qualify a published expressiveness regressor before any Mari assessment.

The physiology-heavy spontaneity label is not character specificity. No
maximization objective or permission to insert random vocal events follows.
"""
import argparse,pathlib,json,importlib.util
import numpy as np,soundfile as sf,torch,torchaudio
from .session import durable_json
from ..causal_v4.renderer import sha
class DeEARProbe:
 def __init__(self,root):
  self.root=pathlib.Path(root);self.evidence=self.root/'continuation/deear_inspection';d=pathlib.Path('/tmp/mari-deear')
  manifest=json.loads((self.evidence/'MODEL_PROVENANCE.json').read_text())
  if manifest['revision']!='8a381030001e5778e17f4d8da04c70010bdac377' or not manifest['complete']:raise ValueError('unselected weights')
  for row in manifest['files']:
   if sha(row['path'])!=row['sha256']:raise ValueError('changed weights')
  tree=json.loads((self.evidence/'TREE.json').read_text());up=self.evidence/'inference.py'
  import hashlib
  b=up.read_bytes();expected=next(x['sha']for x in tree['tree']if x['path']=='inference.py')
  if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=expected:raise ValueError('changed evaluator implementation')
  spec=importlib.util.spec_from_file_location('upstream_deear',up);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  self.model,info=mod.ParallelExpressivenessModel.from_pretrained(d,local_files_only=True,output_loading_info=True)
  if any(info[k]for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError('unloaded evaluator weights '+repr(info))
  self.model.eval();torch.set_num_threads(3)
  self.extractor=mod.Wav2Vec2FeatureExtractor.from_pretrained(d,local_files_only=True)
  import xgboost as xgb
  self.xgb=xgb;self.fusion=xgb.Booster();self.fusion.load_model(d/'xgboost_model.json')
  if sha(d/'xgboost_model.json')!=json.loads((self.evidence/'XGBOOST_SOURCE.json').read_text())['sha256']:raise ValueError('changed fusion weights')
  self.provenance={'revision':manifest['revision'],'load_audit':info,'upstream_source_sha256':sha(up),'implementation_sha256':sha(__file__),'preprocessing':'official inference.py raw waveform->resample16k->feature_extractor.pad; batch1, no duration truncation','scope':'unqualified score probe; no character, causal appropriateness or listening judgment'}
 def evaluate(self,path):
  y,sr=sf.read(path,dtype='float32',always_2d=True)
  if y.shape[1]!=1 or not np.all(np.isfinite(y)):raise ValueError('invalid mono input')
  if not .15<=len(y)/sr<=15:raise ValueError('outside evaluator duration scope; never silently truncate')
  x=torchaudio.functional.resample(torch.from_numpy(y[:,0].copy()),sr,16000)
  inputs=self.extractor.pad([{'input_values':x}],return_tensors='pt',padding=True)
  with torch.inference_mode():pred=self.model(**inputs)
  names=['arousal','prosody','nature'];scores={f'score_{n}':float(pred[f's_{n}'].item())for n in names}
  matrix=self.xgb.DMatrix(np.array([[scores['score_'+n]for n in names]],dtype=np.float32),feature_names=['score_'+n for n in names])
  scores['score_expressive']=float(self.fusion.predict(matrix)[0])
  if not all(np.isfinite(v)for v in scores.values()):raise ValueError('nonfinite evaluator output')
  return {'audio_sha256':sha(path),'seconds':len(y)/sr,**scores}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/deear_qualification';d.mkdir(exist_ok=True)
 corpus=[]
 for p in sorted((r/'continuation/deear_inspection/reference/docs/audio_exp_samples').rglob('*.wav')):
  corpus.append({'id':'human_labeled_'+str(len(corpus)),'path':str(p),'published_bucket':p.parent.name,'origin':'published demonstration; not held-out evidence from model training'})
 old=json.loads((r/'continuation/minicpm_qualification/PREDECLARED.json').read_text())['jobs']
 for id in ['carrier','silence','reversed','mechanical_gaps','retained_rejected','human_annotated']:
  job=next(x for x in old if x['id']==id);corpus.append({'id':id,'path':job['source'],'sha256':job['source_sha256']})
 durable_json(d/'PREDECLARED.json',{'corpus':corpus,'qualification_gates':{'published_bins':'mean overall score increases across each of five published bins; illustrative, not held out','causeless_gaps':'spontaneity/nature must be lower than unmodified carrier','retained_overacting':'nature must not prefer Carl-rejected acting over clean carrier','nonspeech':'scores reported as OOD diagnostics; no acoustic faithfulness claim'},'prohibited_inferences':['high score proves Mari','higher score is universally better','more disfluency required','trained physiology score verifies appropriate cognitive cause'],'full_completion':False})
 probe=DeEARProbe(r);durable_json(d/'PROVENANCE.json',probe.provenance);rows=[]
 for case in corpus:
  if 'sha256'in case and sha(case['path'])!=case['sha256']:raise ValueError('control changed')
  try:result=probe.evaluate(case['path']);row={'case':case,'result':result}
  except Exception as exc:row={'case':case,'error':repr(exc)}
  rows.append(row);durable_json(d/'RESULTS.json',rows);print(case['id'],row.get('result',row.get('error')),flush=True)
 valid={x['case']['id']:x['result']for x in rows if 'result'in x};buckets={}
 for x in rows:
  if 'published_bucket'in x['case'] and 'result'in x:buckets.setdefault(x['case']['published_bucket'],[]).append(x['result']['score_expressive'])
 means={k:float(np.mean(v))for k,v in buckets.items()};ordered=[means.get(k,float('nan'))for k in ['0-2','2-4','4-6','6-8','8-10']]
 checks={'all_valid':len(valid)==len(rows),'published_bins_monotonic':bool(np.all(np.diff(ordered)>0)),'causeless_gaps_not_rewarded':valid['mechanical_gaps']['score_nature']<valid['carrier']['score_nature'],'rejected_acting_not_rewarded':valid['retained_rejected']['score_nature']<valid['carrier']['score_nature']}
 durable_json(d/'ASSESSMENT.json',{'checks':checks,'published_bin_means':means,'qualified_general_character_judge':False,'narrow_controls_pass':all(checks.values()),'scope':'published small demonstration and retained negative controls; full independent validation still required','full_completion':False})
if __name__=='__main__':main()
