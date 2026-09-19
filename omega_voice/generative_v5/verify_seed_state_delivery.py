"""Test actual delivery of a verified knowledge contrast through local CFM."""
import argparse,pathlib,json
from .seed_vc_diagnostic import ConversionProbe,sha
from .session import durable_json

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/seed_state_delivery';d.mkdir(exist_ok=True)
 rows=json.loads((r/'continuation/factive_scene/WAVE_RESULTS.json').read_text());cases=[x for x in rows if x['source']==0 and x['case']in {'no_result','related'}];trace=json.loads((r/'continuation/seed_state_information/RESULTS.json').read_text());t=next(x for x in trace if x['id']=='factive_knowledge')
 if t['identical_generation_information']:raise ValueError('encoder erased contrast; do not waste two renders')
 durable_json(d/'PREDECLARED.json',{'cases':cases,'encoder_observation':t,'purpose':'test whether distinct structured knowledge/physical contour retains its intended acoustic consequence through CFM, including causal timing','gates':{'WER':0,'speaker_similarity_min':.6013473320007324,'resolved_final_contour_steeper_than_no_result':True,'pre_event_PCM_exact_between_counterfactuals':True,'UTMOS_loss_max_vs_source':.15},'not_claimed':'higher similarity or changed audio alone demonstrates cognition/personality','full_completion':False})
 p=ConversionProbe(r);out=[]
 for case in cases:
  source=r/f"continuation/factive_scene/0_{case['case']}.wav"
  if sha(source)!=case['sha256']:raise ValueError('causal input changed')
  receipt=p.render(source,d/(case['case']+'.wav'));out.append({'case':case['case'],'source_evidence':case,'receipt':receipt});durable_json(d/'RESULTS.json',out)
if __name__=='__main__':main()
