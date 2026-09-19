"""Measure the fixed duration ablation; failed content stays failed."""
import argparse,pathlib,json
from .emphasis_evaluator import EmphasisEvaluator,word_observations
from .naturalness_observer import NaturalnessObserver
from .session import durable_json

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/contrast_duration';q=json.loads((r/'continuation/emphasis_qualification/ASSESSMENT.json').read_text())
 if not q['admitted_for_acoustic_emphasis']:raise ValueError('unqualified observer')
 ev=EmphasisEvaluator(r);mos=NaturalnessObserver();results=[];base={i:mos.score(r/f'continuation/contrast_focus/{i}_baseline.wav')for i in range(2)}
 for row in json.loads((d/'RESULTS.json').read_text()):
  p=pathlib.Path(row['audio']);result=ev.infer(p);quality=mos.score(p);obs=word_observations(result,row['alignment']) if row['alignment']else None;focus=obs[row['focus']] if obs else None
  checks={'intelligibility':row['quality']['wer']==0,'identity':row['quality']['identity_pass'],'actual_clock':row['clock_error_s']is not None and row['clock_error_s']<=.08,'predicted_naturalness':base[row['source']]['predicted_MOS']-quality['predicted_MOS']<=.15,'target_emphasized':focus is not None and focus['emphasized']}
  out={'source':row['source'],'variant':row['variant'],'checks':checks,'all_pass':all(checks.values()),'emphasis':result,'focus':focus,'word_observations':obs,'UTMOS':quality,'baseline_UTMOS':base[row['source']],'full_completion':False};results.append(out);durable_json(d/'OBSERVER_RESULTS.json',results);print(row['source'],row['variant'],checks,focus,quality['predicted_MOS'],flush=True)
 durable_json(d/'OBSERVER_ASSESSMENT.json',{'variants':{v:all(x['all_pass']for x in results if x['variant']==v)for v in ['duration_only','duration_plus_contour_pressure']},'scope':'bounded acoustic prominence, identity, content, word clocks and predicted quality; not character presence','full_completion':False})
if __name__=='__main__':main()
