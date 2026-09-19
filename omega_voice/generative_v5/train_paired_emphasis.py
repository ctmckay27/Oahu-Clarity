"""Train one fixed signed-change observer on matched reference realizations."""
import argparse,pathlib,json,collections,itertools,importlib.metadata
import numpy as np,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from .prosodic_features import provenance,NAMES
from .train_prosodic_evaluator import metrics
from .session import durable_json
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/paired_emphasis_evaluator';old=r/'continuation/prosodic_evaluator';pre=json.loads((d/'PREDECLARED.json').read_text());fresh=json.loads((d/'FRESH_FEATURES.json').read_text());complete=json.loads((d/'PREPARATION_COMPLETE.json').read_text());prior=json.loads((old/'FEATURES.json').read_text())
 if len(fresh)!=len(pre['fresh_cases'])or complete['cases']!=len(fresh):raise ValueError('fresh source preparation incomplete')
 if sha(old/'FEATURES.json')!=pre['prior_feature_file_sha256']or provenance()!=pre['feature_provenance']:raise ValueError('source/feature provenance mismatch')
 groups=collections.defaultdict(list)
 for row in prior+fresh:
  if row['admitted']:groups[(row['case']['text_group_sha256'],row['case']['voice'],row['case']['fold'])].append(row)
 pairs=[]
 for (group,voice,fold),rows in groups.items():
  for before,after in itertools.permutations(rows,2):
   if set(before['gold'])==set(after['gold']):continue
   if [x['word']for x in before['word_features']]!=[x['word']for x in after['word_features']]:raise ValueError('matched text differs')
   x=np.asarray(after['features'])-np.asarray(before['features']);y=np.array([int(i in after['gold'])-int(i in before['gold'])for i in range(len(x))]);pairs.append({'before':before['case']['id'],'after':after['case']['id'],'text_group':group,'voice':voice,'fold':fold,'features':x,'labels':y,'words':[w['word']for w in after['word_features']]})
 training=[p for p in pairs if p['fold']=='train'];x=np.concatenate([p['features']for p in training]);y=np.concatenate([p['labels']for p in training]);model=HistGradientBoostingClassifier(learning_rate=.05,max_iter=150,max_leaf_nodes=7,max_depth=3,l2_regularization=1.,class_weight='balanced',random_state=98000,early_stopping=False);model.fit(x,y);path=d/'MODEL.joblib';joblib.dump(model,path,compress=3);results=[]
 for p in pairs:
  pred=model.predict(p['features']);prob=model.predict_proba(p['features']);results.append({k:v for k,v in p.items()if k not in ['features','labels']}|{'labels':p['labels'].tolist(),'predicted':pred.tolist(),'probabilities':prob.tolist()})
 durable_json(d/'PREDICTIONS.json',results);folds={}
 for fold in sorted({p['fold']for p in results}):
  chosen=[p for p in results if p['fold']==fold];truth=np.concatenate([p['labels']for p in chosen]);pred=np.concatenate([p['predicted']for p in chosen]);folds[fold]={'added':metrics(truth==1,pred==1),'removed':metrics(truth==-1,pred==-1),'unchanged':metrics(truth==0,pred==0),'accuracy':float(np.mean(truth==pred)),'pairs':len(chosen)}
 zero=int(model.predict(np.zeros((1,len(NAMES))))[0]);checks={'fresh_clocks_all_admitted':all(x['admitted']for x in fresh),'added_precision':all(folds[f]['added']['precision']>=.8 for f in ['new_text','new_text_and_voice']),'added_recall':all(folds[f]['added']['recall']>=.7 for f in ['new_text','new_text_and_voice']),'removed_precision':all(folds[f]['removed']['precision']>=.8 for f in ['new_text','new_text_and_voice']),'removed_recall':all(folds[f]['removed']['recall']>=.7 for f in ['new_text','new_text_and_voice']),'zero_difference_unchanged':zero==0}
 manifest={'model_sha256':sha(path),'feature_provenance':provenance(),'training_source_sha256':sha(__file__),'source_hashes':{'previous_features':sha(old/'FEATURES.json'),'fresh_features':sha(d/'FRESH_FEATURES.json'),'predeclared':sha(d/'PREDECLARED.json')},'classes':model.classes_.tolist(),'training_words':len(y),'training_pairs':len(training),'dependencies':{x:importlib.metadata.version(x)for x in ['scikit-learn','numpy','scipy','joblib','cmudict','praat-parselmouth']},'model_parameters':model.get_params(),'scope':'paired acoustic changes; no Mari data, state or word identity in fitting'};durable_json(d/'MODEL_PROVENANCE.json',manifest)
 report={'folds':folds,'checks':checks,'excluded_unadmitted_prior_sources':[x['case']['id']for x in prior if not x['admitted']],'fresh_rejected_sources':[x['case']['id']for x in fresh if not x['admitted']],'admitted_for_paired_acoustic_prominence':all(checks.values()),'absolute_prominence_or_character_admitted':False,'zero_difference_prediction':zero,'full_completion':False};durable_json(d/'ASSESSMENT.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
