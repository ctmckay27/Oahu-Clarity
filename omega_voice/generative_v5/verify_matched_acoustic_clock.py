"""Independent same-text/speaker acoustic time correspondence with held-out error bounds."""
import argparse,pathlib,json,collections,itertools
import numpy as np,librosa
from .session import durable_json
from ..causal_v4.renderer import sha

def warp(source,target,source_words):
 def features(path):
  y,_=librosa.load(path,sr=16000);m=librosa.feature.mfcc(y=y,sr=16000,n_mfcc=14,n_fft=400,hop_length=160,n_mels=40)[1:];m=(m-m.mean(axis=1,keepdims=True))/(m.std(axis=1,keepdims=True)+1e-5);return m
 x=features(source);y=features(target);cost,path=librosa.sequence.dtw(X=x,Y=y,metric='cosine',step_sizes_sigma=np.array([[1,1],[1,2],[2,1]]),weights_add=np.array([0.,.05,.05]),global_constraints=True,band_rad=.2);path=path[::-1];indexes=np.unique(path[:,0]);mapping=np.array([np.mean(path[path[:,0]==i,1])for i in indexes]);words=[{'word':w['word'],'start':float(np.interp(w['start']*100,indexes,mapping)/100),'end':float(np.interp(w['end']*100,indexes,mapping)/100)}for w in source_words];return words,float(cost[-1,-1]/len(path))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);source=r/'continuation/intrinsic_emphasis_evaluator';d=r/'continuation/matched_clock_transfer';d.mkdir(exist_ok=True);rows=json.loads((source/'FRESH_FEATURES.json').read_text());recover={}
 for name in ['RESULTS.json','MMS_RESULTS.json','MMS_PAD_RESULTS.json']:
  for x in json.loads((r/'continuation/intrinsic_alignment'/name).read_text()):
   if x['admitted']and x['id']not in recover:recover[x['id']]=x['alignment']
 groups=collections.defaultdict(list)
 for x in rows:
  if not x['admitted']and x['case']['id']in recover:x=dict(x,admitted=True,alignment=recover[x['case']['id']])
  groups[(tuple(x['case']['src_sentence']),x['case']['voice'])].append(x)
 pairs=[]
 for key,rs in sorted(groups.items()):
  good=[x for x in rs if x['admitted']]
  if len(good)>1:pairs.append((good[0],good[1]))
 pairs=pairs[:24];failed=[x for rs in groups.values()for x in rs if not x['admitted']];durable_json(d/'PREDECLARED.json',{'qualified_pairs':[(a['case']['id'],b['case']['id'])for a,b in pairs],'unresolved_targets':[x['case']['id']for x in failed],'method':'CMVN low-order MFCC (C1-C13),10ms frames, cosine DTW with monotonic bounded steps and20percent diagonal band; no transcript posterior, emphasis label or Mari state enters matching','gates':{'qualification_boundary_p95_s':.08,'qualification_boundary_median_s':.04,'target_two_reference_agreement_max_s':.08,'target_path_cost_no_worse_than_max_qualification':True},'full_completion':False});tests=[];errors=[]
 for a,b in pairs:
  aligned,cost=warp(a['path'],b['path'],a['alignment']['words']);err=[abs(x[k]-y[k])for x,y in zip(aligned,b['alignment']['words'])for k in ['start','end']];errors+=err;tests.append({'source':a['case']['id'],'target':b['case']['id'],'mapped':aligned,'observed':b['alignment']['words'],'path_cost':cost,'max_error_s':max(err),'errors_s':err});durable_json(d/'QUALIFICATION_RESULTS.json',tests);print(a['case']['id'],b['case']['id'],'error',max(err),flush=True)
 median=float(np.median(errors));p95=float(np.percentile(errors,95));qualified=median<=.04 and p95<=.08;assessment={'qualified':qualified,'median_error_s':median,'p95_error_s':p95,'max_error_s':max(errors),'pairs':len(tests),'full_completion':False};durable_json(d/'ASSESSMENT.json',assessment)
 if not qualified:print('QUALIFICATION FAILED',assessment,flush=True);return
 out=[];maxcost=max(x['path_cost']for x in tests)
 for target in failed:
  key=(tuple(target['case']['src_sentence']),target['case']['voice']);references=[x for x in groups[key]if x['admitted']][:2]
  if len(references)<2:out.append({'id':target['case']['id'],'admitted':False,'error':'fewer than two admitted same-speaker/text references'});continue
  predicted=[warp(x['path'],target['path'],x['alignment']['words'])for x in references];delta=max(abs(x[k]-y[k])for x,y in zip(predicted[0][0],predicted[1][0])for k in ['start','end']);cost=max(x[1]for x in predicted);words=[{'word':x['word'],'start':(x['start']+y['start'])/2,'end':(x['end']+y['end'])/2}for x,y in zip(predicted[0][0],predicted[1][0])];admitted=delta<=.08 and cost<=maxcost;out.append({'id':target['case']['id'],'admitted':admitted,'alignment':{'source_sha256':sha(target['path']),'words':words,'exact_words':True,'admitted':admitted,'method':'qualified same-text/speaker acoustic clock transfer','confidence_kind':'empirical matched-acoustic boundary agreement; no invented CTC probability','reference_ids':[x['case']['id']for x in references],'reference_clock_hashes':[sha(source/'FRESH_FEATURES.json')],'reference_agreement_max_s':delta,'path_cost':cost,'qualification_max_path_cost':maxcost,'qualification_sha256':sha(d/'QUALIFICATION_RESULTS.json')}});durable_json(d/'RECOVERY_RESULTS.json',out);print('target',target['case']['id'],admitted,delta,cost,flush=True)
if __name__=='__main__':main()
