"""Independent observers on delivered held-out session outputs."""
import argparse,pathlib,json
from .emphasis_evaluator import EmphasisEvaluator,word_observations
from .naturalness_observer import NaturalnessObserver
from .session import durable_json
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/contrast_session';cases=json.loads((d/'RESULTS.json').read_text())
 if len(cases)!=3 or not all(x['passed']for x in cases):raise ValueError('held-out session incomplete or failed')
 if not json.loads((r/'continuation/emphasis_qualification/ASSESSMENT.json').read_text())['admitted_for_acoustic_emphasis']:raise ValueError('unqualified observer')
 ev=EmphasisEvaluator(r);mos=NaturalnessObserver();out=[]
 for i,row in enumerate(cases):
  p=d/'session'/row['id'];receipt=json.loads((p/'receipt.json').read_text());trial=receipt['trials'][-1];audio=pathlib.Path(trial['audio']);result=ev.infer(audio);words=word_observations(result,trial['observed_alignment']);focus=[w for w in words if w['word'].lower()in [x.lower()for x in row['focus_words']]];baseline=mos.score(p/'carrier.wav');score=mos.score(audio);windows=receipt['prominence']['windows'];checks={'verified_audio':sha(audio)==trial['evaluation']['audio']['sha256'],'UTMOS_loss':baseline['predicted_MOS']-score['predicted_MOS']<=.15,'single_focus':len(focus)==1}
  if i<2:checks['correction_target_emphasized']=len(focus)==1 and focus[0]['emphasized']
  else:checks['shared_amusement_duration_bounded_below_correction']=windows[0]['duration_ratio']<cases[0]['prominence']['windows'][0]['duration_ratio']
  out.append({'id':row['id'],'checks':checks,'passed':all(checks.values()),'focus':focus,'emphasis_observation':result,'word_observations':words,'UTMOS_baseline':baseline,'UTMOS_delivered':score,'duration_windows':windows,'scope':'acoustic prominence and quality; no claim that a listener perceives amusement or Mari character','full_completion':False});durable_json(d/'OBSERVER_RESULTS.json',out);print(row['id'],checks,focus,score['predicted_MOS'],flush=True)
 durable_json(d/'OBSERVER_ASSESSMENT.json',{'all_pass':all(x['passed']for x in out),'scope':'held-out lexical prominence, weaker shared-contrast duration and predicted quality; no broad character admission','full_completion':False})
if __name__=='__main__':main()
