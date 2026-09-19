"""Actual performed event clock and state continuity on retained matched recordings."""
import argparse,pathlib,json
from .acknowledgment_clock import observe_episode,reconstruct_episode
from .mms_alignment import MMSAligner
from .session import durable_json
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.runtime import digest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);source=r/'continuation/ack_counterfactual';d=r/'continuation/acknowledgment_clock';d.mkdir(exist_ok=True);cases=json.loads((source/'PREDECLARED.json').read_text())['cases'];durable_json(d/'PREDECLARED.json',{'cases':[x['id']for x in cases],'source':'retained matched native phonetic counterfactuals; no rerender claim','no_trim_or_added_pause':True,'gates':['independent exact lexical continuation','whole episode identity','MMS suffix confidence>=.35','observed distinct prelexical phonation','actual duration state conservation','replay equality'],'full_completion':False});ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');al=MMSAligner(r/'continuation/alignment_padding/MMS_DEPENDENCIES.json');out=[]
 for c in cases:
  try:
   observation=observe_episode(source/(c['id']+'.wav'),c['plan']['event'],c['continuation'],ev,al);plan=reconstruct_episode(c['plan']['initial_state'],c['plan']['event'],c['continuation'],observation);repeat=reconstruct_episode(c['plan']['initial_state'],c['plan']['event'],c['continuation'],observation);row={'id':c['id'],'observation':observation,'plan':plan,'replay_identical':digest(plan)==digest(repeat),'pass':digest(plan)==digest(repeat)}
  except Exception as e:row={'id':c['id'],'pass':False,'error':type(e).__name__+': '+str(e)}
  out.append(row);durable_json(d/'RESULTS.json',out);print(c['id'],row['pass'],row.get('error'),flush=True)
 durable_json(d/'ASSESSMENT.json',{'actual_clock_gates':all(x['pass']for x in out),'meaning_and_isolated_identity_unverified':True,'full_completion':False})
if __name__=='__main__':main()
