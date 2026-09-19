"""Fixed diagnostic event conversion after a reproducible Mari reconstruction."""
import argparse,pathlib,json
from .seed_vc_diagnostic import ConversionProbe,sha
from .session import durable_json

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);q=r/'continuation/seed_vc_qualification'
 quality=json.loads((q/'SPEECH_QUALITY.json').read_text());repeat=json.loads((q/'REPRODUCIBILITY.json').read_text());mos=json.loads((q/'UTMOS_24k.json').read_text())['score']
 checks={'speech_quality':quality['quality_screen_pass']and quality['wer']==0,'identity':quality['speaker_similarity']>=.6013473320007324,'utmos_drop':4.511293888092041-mos<=.15,'duration':abs(quality['audio']['duration_s']/4.32-1)<=.04,'repeat':repeat['repeat_PCM_exact']and repeat['native_PCM_exact']}
 durable_json(q/'RECONSTRUCTION_ASSESSMENT.json',{'checks':checks,'all_pass':all(checks.values()),'scope':'single-text same-Mari reconstruction capability only','event_identity_or_semantics_admitted':False,'full_completion':False})
 if not all(checks.values()):raise ValueError('reconstruction failed; no event conversion authorized by probe')
 source=json.loads((r/'continuation/backchannel_reference/HUMAN_BACKCHANNELS.json').read_text());cases=source['examples'][::2];d=r/'continuation/seed_vc_events';d.mkdir(exist_ok=True)
 durable_json(d/'PREDECLARED.json',{'cases':cases,'selection':'first previously selected annotated AMI event per channel; no model-score selection','purpose':'test whether timbre-conditioned content conversion preserves event class','target':'unchanged G1-1 anchor','identity_gate':'not qualified for short events; no admission from sentence ECAPA','attribution':source['attribution'],'license':source['license'],'full_completion':False})
 probe=ConversionProbe(r);rows=[]
 for case in cases:
  if sha(case['path'])!=case['sha256']:raise ValueError('source event changed')
  receipt=probe.render(case['path'],d/(case['id']+'.wav'))
  rows.append({'id':case['id'],'source':case,'receipt':receipt,'identity_admitted':False,'meaning_admitted':False,'full_completion':False});durable_json(d/'RESULTS.json',rows)
if __name__=='__main__':main()
