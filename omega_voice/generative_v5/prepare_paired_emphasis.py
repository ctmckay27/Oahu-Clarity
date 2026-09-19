"""Matched-word acoustic changes cancel phonetic and ordinary cadence effects."""
import argparse,pathlib,json,collections,hashlib
import librosa
from .alignment import ForcedAligner,AlignmentRejected
from .prosodic_features import extract,provenance
from .session import durable_json
from ..causal_v4.renderer import sha
from ..causal_v4.evaluate import normalized

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--intrinsic',action='store_true');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation'/('intrinsic_emphasis_evaluator'if a.intrinsic else'paired_emphasis_evaluator');d.mkdir(exist_ok=True);old=r/'continuation/prosodic_evaluator';prior=json.loads((old/'FEATURES.json').read_text());old_groups={x['case']['text_group_sha256']for x in prior};meta=pathlib.Path('/tmp/mari-emphassess-data/gold_df.json');raw=[json.loads(x)for x in meta.read_text().splitlines()];groups=collections.defaultdict(list)
 for row in raw:groups[tuple(row['src_sentence'])].append(row)
 def keyhash(k):return hashlib.sha256(json.dumps(k).encode()).hexdigest()
 if a.intrinsic:
  for extra in [r/'continuation/paired_emphasis_evaluator/FRESH_FEATURES.json',r/'continuation/coupled_prominence_calibration/REFERENCES.json',r/'continuation/matched_emphasis/RESULTS.json']:
   old_groups.update(keyhash(tuple(x['case']['src_sentence']))for x in json.loads(extra.read_text()))
 eligible=[k for k,v in groups.items()if keyhash(k)not in old_groups and len({tuple(x['gold_emphasis'])for x in v})>=2];selected=sorted(eligible,key=keyhash)[:8];cases=[]
 for key in selected:
  for row in sorted(groups[key],key=lambda x:x['id']):cases.append(dict(row,fold='new_text_and_voice'if row['voice']=='expresso_ex04'else 'new_text',text_group_sha256=keyhash(key)))
 pre={'fresh_cases':cases,'prior_feature_file_sha256':sha(old/'FEATURES.json'),'feature_provenance':provenance(),'source_metadata_sha256':sha(meta),'selection':'first eight hash-ordered new text groups with alternate emphasis labels, excluding all thirty previously inspected groups; all voices/versions','representation':'per-word acoustic feature differences between same-text/same-voice recordings; labels added, removed or unchanged emphasis. No word-id, speaker-id or Mari state input.','training':'only previously declared training text/voices with independently admitted clocks; rejected source clocks remain excluded and recorded, never relabeled','model':{'type':'HistGradientBoostingClassifier','learning_rate':.05,'max_iter':150,'max_leaf_nodes':7,'max_depth':3,'l2_regularization':1.,'class_weight':'balanced','random_state':98000,'decision':'argmax among -1,0,1; no tuned threshold'},'gates':{'fresh_clocks_all_admitted':True,'each_new_fold_added_precision_min':.8,'each_new_fold_added_recall_min':.7,'each_new_fold_removed_precision_min':.8,'each_new_fold_removed_recall_min':.7,'zero_difference_predicts_unchanged':True},'scope':'causal acoustic prominence change, not absolute character or dramatic-appropriateness judgment','full_completion':False};durable_json(d/'PREDECLARED.json',pre)
 if a.intrinsic:
  pre.update(feature_indices=[0,1,2,3,4,5,6,7,8,9,21],scope='within-word acoustic prominence change; explicit interword gap features excluded, no claim about pause-only emphasis',selection='first eight hash-ordered texts excluding all earlier observer-development/qualification and actuator-calibration texts; all voices/versions',original_paired_model_preserved=True,preparation_source_sha256=sha(__file__));durable_json(d/'PREDECLARED.json',pre)
 prov=json.loads((r/'continuation/emphasis_inspection/DATASET_PROVENANCE.json').read_text());paths={pathlib.Path(x['name']).stem:x for x in prov['files']if x['name'].endswith('.wav')};al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');done=json.loads((d/'FRESH_FEATURES.json').read_text())if(d/'FRESH_FEATURES.json').exists()else [];seen={x['case']['id']for x in done}
 for case in cases:
  if case['id']in seen:continue
  p=pathlib.Path(paths[case['id']]['path']);assert sha(p)==paths[case['id']]['sha256'];wave,_=librosa.load(p,sr=16000);gold=[];offset=0
  for i,t in enumerate(case['src_sentence']):
   n=len(normalized(t))
   if i in case['gold_emphasis']:gold.extend(range(offset,offset+n))
   offset+=n
  try:alignment=al.align(wave,' '.join(case['src_sentence']),sha(p),True);x,words=extract(p,alignment);record={'case':case,'path':str(p),'audio_sha256':sha(p),'gold':gold,'alignment':alignment,'features':x.tolist(),'word_features':words,'admitted':True}
  except AlignmentRejected as error:record={'case':case,'path':str(p),'audio_sha256':sha(p),'gold':gold,'admitted':False,'rejected_alignment':error.report}
  done.append(record);durable_json(d/'FRESH_FEATURES.json',done);print(len(done),'of',len(cases),case['id'],record['admitted'],flush=True)
 durable_json(d/'PREPARATION_COMPLETE.json',{'cases':len(done),'rejected':sum(not x['admitted']for x in done),'full_completion':False})
if __name__=='__main__':main()
