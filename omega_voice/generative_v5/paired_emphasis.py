"""Qualified signed acoustic-change observer; no intent or personhood judgment."""
import pathlib,json
import numpy as np,joblib
from .prosodic_features import extract,provenance
from ..causal_v4.renderer import sha

class PairedEmphasisObserver:
 def __init__(self,directory):
  d=pathlib.Path(directory);self.directory=d;self.qualification=json.loads((d/'QUALIFIED_ASSESSMENT.json').read_text());m=json.loads((d/'MODEL_PROVENANCE.json').read_text())
  if not self.qualification['admitted_for_paired_acoustic_prominence']or sha(d/'MODEL.joblib')!=m['model_sha256']or m['model_sha256']!=self.qualification['model_sha256']or provenance()!=m['feature_provenance']:raise ValueError('observer not qualified or changed')
  files={'fresh_original':'FRESH_FEATURES.json','fresh_recovered':'RECOVERED_FRESH_FEATURES.json','recovery_policy':'RECOVERY_PREDECLARED.json','model_provenance':'MODEL_PROVENANCE.json'}
  for key,name in files.items():
   if sha(d/name)!=self.qualification['source_hashes'][key]:raise ValueError('qualification evidence changed')
  self.model=joblib.load(d/'MODEL.joblib');self.indices=m.get('feature_indices',list(range(len(m['feature_provenance']['features']))));self.scope=m['scope']

 def compare(self,before,before_alignment,after,after_alignment):
  for path,alignment in [(before,before_alignment),(after,after_alignment)]:
   if not alignment['admitted']or alignment['minimum_word_probability']<.35 or alignment['source_sha256']!=sha(path):raise ValueError('unverified acoustic clock')
  a,wa=extract(before,before_alignment);b,wb=extract(after,after_alignment)
  if [x['word']for x in wa]!=[x['word']for x in wb]:raise ValueError('paired observation requires identical spoken words')
  difference=(b-a)[:,self.indices];pred=self.model.predict(difference);prob=self.model.predict_proba(difference)
  return {'before_sha256':sha(before),'after_sha256':sha(after),'model_sha256':self.qualification['model_sha256'],'qualification_sha256':sha(self.directory/'QUALIFIED_ASSESSMENT.json'),'classes':self.model.classes_.tolist(),'words':[{'index':i,'word':w['word'],'change':int(pred[i]),'probabilities':prob[i].tolist()}for i,w in enumerate(wb)],'scope':'paired acoustic prominence change only','character_or_dramatic_appropriateness_admitted':False,'full_completion':False}
