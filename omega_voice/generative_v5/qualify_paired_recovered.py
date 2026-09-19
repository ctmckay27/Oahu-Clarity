"""Finish frozen-model qualification after resolving reference clocks independently."""
import argparse,pathlib,json,copy,collections,itertools
import joblib,numpy as np
from .prosodic_features import extract,provenance,NAMES
from .train_prosodic_evaluator import metrics
from .session import durable_json
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--intrinsic',action='store_true');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation'/('intrinsic_emphasis_evaluator'if a.intrinsic else'paired_emphasis_evaluator');ad=r/'continuation'/('intrinsic_alignment'if a.intrinsic else'alignment_padding');original=json.loads((d/'FRESH_FEATURES.json').read_text());manifest=json.loads((d/'MODEL_PROVENANCE.json').read_text());pre=json.loads((d/'PREDECLARED.json').read_text());indices=manifest.get('feature_indices',list(range(len(NAMES))))
 if sha(d/'MODEL.joblib')!=manifest['model_sha256']or sha(d/'FRESH_FEATURES.json')!=manifest['source_hashes']['fresh_features']or provenance()!=manifest['feature_provenance']:raise ValueError('frozen model/source changed')
 padded={x['id']:x for x in json.loads((ad/'RESULTS.json').read_text())};mms={x['id']:x for x in json.loads((ad/'MMS_RESULTS.json').read_text())};rows=[];recovered=[]
 durable_json(d/'RECOVERY_PREDECLARED.json',{'model_sha256':manifest['model_sha256'],'model_frozen_no_refit':True,'clock_resolution_order':['original admitted large CTC','fixed400ms padded large CTC','alignment-trained MMS'],'unchanged_word_confidence_min':.35,'original_rejections_preserved':True,'all_original_fresh_cases_included':True,'prediction_gates':pre['gates'],'full_completion':False})
 for old in original:
  x=copy.deepcopy(old)
  if not x['admitted']:
   case=x['case']['id'];candidate=next((q for q in [padded.get(case),mms.get(case)]if q and q['admitted']),None)
   if candidate is None:raise ValueError('reference clock remains unresolved: '+case)
   alignment=candidate['alignment']
   if alignment['minimum_word_probability']<.35 or alignment['source_sha256']!=sha(x['path']):raise ValueError('reference confidence/source gate')
   matrix,words=extract(x['path'],alignment);x.update(alignment=alignment,features=matrix.tolist(),word_features=words,admitted=True,original_rejected_clock_retained=True);recovered.append({'id':case,'method':alignment['method'],'padding_s':alignment.get('evaluator_input_padding_s',0),'minimum_word_probability':alignment['minimum_word_probability']})
  rows.append(x)
 durable_json(d/'RECOVERED_FRESH_FEATURES.json',rows);model=joblib.load(d/'MODEL.joblib');groups=collections.defaultdict(list)
 for x in rows:groups[(x['case']['text_group_sha256'],x['case']['voice'],x['case']['fold'])].append(x)
 results=[]
 for (group,voice,fold),group_rows in groups.items():
  for before,after in itertools.permutations(group_rows,2):
   if set(before['gold'])==set(after['gold']):continue
   words=[x['word']for x in after['word_features']]
   if words!=[x['word']for x in before['word_features']]:raise ValueError('paired words differ')
   matrix=(np.asarray(after['features'])-np.asarray(before['features']))[:,indices];truth=[int(i in after['gold'])-int(i in before['gold'])for i in range(len(matrix))];pred=model.predict(matrix).tolist();results.append({'before':before['case']['id'],'after':after['case']['id'],'voice':voice,'fold':fold,'words':words,'truth':truth,'predicted':pred,'probabilities':model.predict_proba(matrix).tolist()})
 durable_json(d/'RECOVERED_FRESH_PREDICTIONS.json',results);folds={}
 for fold in ['new_text','new_text_and_voice']:
  chosen=[x for x in results if x['fold']==fold];truth=np.concatenate([x['truth']for x in chosen]);pred=np.concatenate([x['predicted']for x in chosen]);folds[fold]={'added':metrics(truth==1,pred==1),'removed':metrics(truth==-1,pred==-1),'unchanged':metrics(truth==0,pred==0),'accuracy':float(np.mean(truth==pred)),'pairs':len(chosen)}
 zero=int(model.predict(np.zeros((1,len(indices))))[0]);checks={'all_fresh_cases_included':len(rows)==len(pre['fresh_cases']),'all_reference_clocks_admitted':all(x['admitted']for x in rows),'added_precision':all(folds[f]['added']['precision']>=.8 for f in folds),'added_recall':all(folds[f]['added']['recall']>=.7 for f in folds),'removed_precision':all(folds[f]['removed']['precision']>=.8 for f in folds),'removed_recall':all(folds[f]['removed']['recall']>=.7 for f in folds),'zero_difference_unchanged':zero==0}
 report={'folds':folds,'checks':checks,'recovered':recovered,'model_frozen_no_refit':True,'model_sha256':manifest['model_sha256'],'source_hashes':{'fresh_original':sha(d/'FRESH_FEATURES.json'),'fresh_recovered':sha(d/'RECOVERED_FRESH_FEATURES.json'),'recovery_policy':sha(d/'RECOVERY_PREDECLARED.json'),'model_provenance':sha(d/'MODEL_PROVENANCE.json')},'admitted_for_paired_acoustic_prominence':all(checks.values()),'absolute_prominence_or_character_admitted':False,'full_completion':False};durable_json(d/'QUALIFIED_ASSESSMENT.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
