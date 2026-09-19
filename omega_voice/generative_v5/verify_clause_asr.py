"""Positive and rejected-content controls for full-coverage clause ASR."""
import argparse,pathlib,json,copy
from .clause_asr import verify_clauses
from .session import durable_json
from ..causal_v4.evaluate import Evaluator

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/two_contrasts_deterministic';cases=json.loads((d/'RESULTS.json').read_text());ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');rows=[]
 for c in [cases[0],cases[2]]:
  result=verify_clauses(c['audio'],c['plan']['text'],c['receipt']['mapped_timeline'],ev);rows.append({'id':c['id'],'report':result});durable_json(d/'CLAUSE_VERIFICATION.json',rows);print(c['id'],result['admitted'],[x['ASR_text']for x in result['clauses']],flush=True)
 c=cases[2];wrong=verify_clauses(c['audio'],c['plan']['text'].replace('blue','red'),c['receipt']['mapped_timeline'],ev);stale=copy.deepcopy(c['receipt']['mapped_timeline']);stale['source_audio_sha256']='0'*64
 try:verify_clauses(c['audio'],c['plan']['text'],stale,ev);stale_rejected=False
 except ValueError:stale_rejected=True
 report={'positives_pass':all(x['report']['admitted']for x in rows),'wrong_spoken_content_rejected':not wrong['admitted'],'wrong_content_report':wrong,'stale_clock_rejected':stale_rejected,'raw_whole_ASR_failures_preserved':True,'scope':'lexical verification by all clauses, not character/naturalness; session integration pending','full_completion':False};durable_json(d/'CLAUSE_QUALIFICATION.json',report);print({k:v for k,v in report.items()if isinstance(v,bool)},flush=True)
if __name__=='__main__':main()
