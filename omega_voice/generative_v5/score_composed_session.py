"""Independent emphasis and predicted-quality gates after deterministic repair."""
import argparse,pathlib,json
from .emphasis_evaluator import EmphasisEvaluator,word_observations
from .naturalness_observer import NaturalnessObserver
from .session import durable_json

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/composed_session_replay';rows=json.loads((d/'RESULTS.json').read_text());ev=EmphasisEvaluator(r);mos=NaturalnessObserver();base=mos.score(r/'continuation/text_release/carrier_parity.wav');results=[]
 for row in rows:
  if not row['admitted']:raise ValueError('failed waveform cannot enter observer admission')
  rec=row['receipt'];trial=rec['trials'][-1];audio=pathlib.Path(trial['audio']);emphasis=ev.infer(audio);obs=word_observations(emphasis,trial['observed_alignment']);quality=mos.score(audio);focus=[w['word']for w in rec['prominence']['windows']];selected=[obs[i]for i in focus];checks={'every_target_emphasized':all(x['emphasized']for x in selected),'predicted_naturalness':base['predicted_MOS']-quality['predicted_MOS']<=.15,'identity':trial['evaluation']['identity_pass'],'actual_clock':trial['alignment_error_s']<=.08}
  result={'id':row['id'],'checks':checks,'all_pass':all(checks.values()),'targets':selected,'emphasis':emphasis,'word_observations':obs,'UTMOS':quality,'baseline_UTMOS':base,'raw_whole_WER':trial['evaluation']['wer'],'strict_clause_ASR':trial['evaluation'].get('clause_ASR_verification'),'full_completion':False};results.append(result);durable_json(d/'OBSERVER_RESULTS.json',results);print(row['id'],checks,selected,quality['predicted_MOS'],flush=True)
 durable_json(d/'OBSERVER_ASSESSMENT.json',{'all_pass':all(x['all_pass']for x in results),'scope':'composed acoustic emphasis and predicted quality, not perceived dramatic intent or Mari character presence','full_completion':False})
if __name__=='__main__':main()
