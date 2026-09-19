"""One predeclared model; separate text/voice and multiple-focus qualification."""
import argparse,pathlib,json,importlib.metadata
import numpy as np,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from .session import durable_json
from .prosodic_features import provenance,NAMES
from ..causal_v4.renderer import sha

def metrics(y,p):
 y=np.asarray(y,dtype=bool);p=np.asarray(p,dtype=bool);tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
 return {'TP':tp,'FP':fp,'FN':fn,'precision':precision,'recall':recall,'F1':2*precision*recall/max(1e-9,precision+recall),'words':len(y)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/prosodic_evaluator';pre=json.loads((d/'PREDECLARED.json').read_text());complete=json.loads((d/'PREPARATION_COMPLETE.json').read_text());rows=json.loads((d/'FEATURES.json').read_text())
 if len(rows)!=len(pre['cases'])or complete['cases']!=len(rows):raise ValueError('incomplete source corpus')
 if pre['feature_provenance']!=provenance():raise ValueError('feature implementation or lexicon changed during preparation')
 valid=[x for x in rows if x['admitted']];train=[x for x in valid if x['case']['fold']=='train'];x=np.concatenate([np.asarray(a['features'])for a in train]);y=np.concatenate([np.array([i in a['gold']for i in range(len(a['features']))])for a in train]);model=HistGradientBoostingClassifier(learning_rate=.05,max_iter=150,max_leaf_nodes=7,max_depth=3,l2_regularization=1.,class_weight='balanced',random_state=97900,early_stopping=False);model.fit(x,y);path=d/'MODEL.joblib';joblib.dump(model,path,compress=3);results=[]
 for a in valid:
  probability=model.predict_proba(np.asarray(a['features']))[:,1];truth=[i in a['gold']for i in range(len(a['features']))];results.append({'id':a['case']['id'],'fold':a['case']['fold'],'text_group':a['case']['text_group_sha256'],'multiple':len(a['gold'])>1,'gold':a['gold'],'probabilities':probability.tolist(),'predicted':np.flatnonzero(probability>=.5).tolist(),'metrics':metrics(truth,probability>=.5)})
 durable_json(d/'PREDICTIONS.json',results);folds={}
 for fold in ['train','heldout_voice','heldout_text','heldout_text_and_voice']:
  selected=[a for a in results if a['fold']==fold];yy=[];pp=[]
  for a in selected:yy += [i in a['gold']for i in range(len(a['probabilities']))];pp += [v>=.5 for v in a['probabilities']]
  folds[fold]=metrics(yy,pp)
 yy=[];pp=[]
 for a in results:
  if a['fold']!='train'and a['multiple']:yy += [i in a['gold']for i in range(len(a['probabilities']))];pp += [v>=.5 for v in a['probabilities']]
 multi=metrics(yy,pp);rejected=[a['case']['id']for a in rows if not a['admitted']];checks={'all_selected_clocks_admitted':not rejected,'heldout_precision':all(folds[f]['precision']>=.8 for f in folds if f!='train'),'heldout_recall':all(folds[f]['recall']>=.7 for f in folds if f!='train'),'multiple_target_recall':multi['recall']>=.7}
 manifest={'model_sha256':sha(path),'feature_provenance':provenance(),'training_source_sha256':sha(__file__),'features_file_sha256':sha(d/'FEATURES.json'),'selection_sha256':sha(d/'PREDECLARED.json'),'dependency_versions':{x:importlib.metadata.version(x)for x in ['scikit-learn','numpy','scipy','joblib','cmudict','praat-parselmouth']},'model_parameters':model.get_params(),'training_words':int(len(y)),'training_positive_words':int(y.sum()),'scope':'diagnostic acoustic emphasis observer; no Mari data in training or model selection'};durable_json(d/'MODEL_PROVENANCE.json',manifest)
 report={'folds':folds,'multiple_focus_holdout':multi,'rejected_clocks':rejected,'checks':checks,'admitted_for_acoustic_emphasis':all(checks.values()),'Mari_data_in_training':False,'decision_threshold':.5,'full_completion':False};durable_json(d/'ASSESSMENT.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
