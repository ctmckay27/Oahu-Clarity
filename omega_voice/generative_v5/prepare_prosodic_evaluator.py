"""Source-only development split, with held-out text and held-out speaker."""
import argparse,pathlib,json,collections,hashlib
import librosa
from .alignment import ForcedAligner,AlignmentRejected
from .prosodic_features import extract,provenance
from .session import durable_json
from ..causal_v4.renderer import sha
from ..causal_v4.evaluate import normalized

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/prosodic_evaluator';d.mkdir(exist_ok=True);data=pathlib.Path('/tmp/mari-emphassess-data');meta=data/'gold_df.json';raw=[json.loads(x)for x in meta.read_text().splitlines()];groups=collections.defaultdict(list)
 for row in raw:groups[tuple(row['src_sentence'])].append(row)
 buckets={'multiple':[],'single':[]}
 for key,rows in groups.items():buckets['multiple'if any(len(x['gold_emphasis'])>1 for x in rows)else 'single'].append(key)
 selected={}
 for category,train_count,test_count in [('multiple',8,4),('single',12,6)]:
  keys=sorted(buckets[category],key=lambda k:hashlib.sha256(json.dumps(k).encode()).hexdigest())
  for i,k in enumerate(keys[:train_count+test_count]):selected[k]='development_text'if i<train_count else 'heldout_text'
 cases=[]
 for key,split in selected.items():
  for row in sorted(groups[key],key=lambda x:x['id']):
   fold='train'if split=='development_text'and row['voice']!='expresso_ex04'else 'heldout_voice'if split=='development_text'else 'heldout_text_and_voice'if row['voice']=='expresso_ex04'else 'heldout_text';cases.append(dict(row,fold=fold,text_group_sha256=hashlib.sha256(json.dumps(key).encode()).hexdigest()))
 prov=json.loads((r/'continuation/emphasis_inspection/DATASET_PROVENANCE.json').read_text());paths={pathlib.Path(x['name']).stem:x for x in prov['files']if x['name'].endswith('.wav')}
 pre={'cases':cases,'metadata_sha256':sha(meta),'selection':'hash-ordered20development/10heldout text groups stratified single/multiple; all versions and voices; voice ex04 never in training','feature_provenance':provenance(),'gates':{'each_holdout_precision_min':.8,'each_holdout_recall_min':.7,'multiple_target_recall_min':.7},'fixed_model':{'type':'HistGradientBoostingClassifier','learning_rate':.05,'max_iter':150,'max_leaf_nodes':7,'max_depth':3,'l2_regularization':1.,'class_weight':'balanced','random_state':97900,'decision_probability':.5},'Mari_data_in_training_or_selection':False,'scope':'independent acoustic emphasis observer qualification, not character completion','full_completion':False};durable_json(d/'PREDECLARED.json',pre)
 al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');done=json.loads((d/'FEATURES.json').read_text())if(d/'FEATURES.json').exists()else [];seen={x['case']['id']for x in done}
 for case in cases:
  if case['id']in seen:continue
  p=pathlib.Path(paths[case['id']]['path']);assert sha(p)==paths[case['id']]['sha256'];wave,_=librosa.load(p,sr=16000);gold=[];offset=0
  for i,t in enumerate(case['src_sentence']):
   n=len(normalized(t))
   if i in case['gold_emphasis']:gold.extend(range(offset,offset+n))
   offset+=n
  try:
   alignment=al.align(wave,' '.join(case['src_sentence']),sha(p),True);x,words=extract(p,alignment);row={'case':case,'path':str(p),'audio_sha256':sha(p),'gold':gold,'alignment':alignment,'features':x.tolist(),'word_features':words,'admitted':True}
  except AlignmentRejected as error:row={'case':case,'path':str(p),'audio_sha256':sha(p),'gold':gold,'admitted':False,'rejected_alignment':error.report}
  done.append(row);durable_json(d/'FEATURES.json',done);print(len(done),'of',len(cases),case['id'],row['admitted'],flush=True)
 durable_json(d/'PREPARATION_COMPLETE.json',{'cases':len(done),'rejected':sum(not x['admitted']for x in done),'full_completion':False})
if __name__=='__main__':main()
