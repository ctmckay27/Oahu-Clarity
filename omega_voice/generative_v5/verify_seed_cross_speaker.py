"""Discriminate timbre conversion from same-speaker reconstruction only."""
import argparse,pathlib,json
from .seed_vc_diagnostic import ConversionProbe,sha
from .session import durable_json

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/seed_vc_cross_speaker';d.mkdir(exist_ok=True)
 # Existing fixed enrollment segments: no target/source audition or score search.
 cases=[{'id':k,'source':str(r/f'continuation/event_identity/{k}.enrollment.wav'),'provenance':json.loads((r/f'continuation/event_identity/{k}.source.json').read_text())}for k in ['A','B','D']]
 durable_json(d/'PREDECLARED.json',{'cases':cases,'target':'unchanged G1-1 Mari anchor','purpose':'test cross-speaker timbre transfer on long-enough speech before interpreting short events','gates':{'target_speaker_similarity_min':.6013473320007324,'target_must_be_highest_among_Mari_and_three_source_enrollments':True,'source_ASR_to_converted_ASR_WER_max':.12,'duration_relative_error_max':.04,'no_clipping_or_corruption':True},'not_claimed':['source ASR is human ground truth','source performance instantiates Mari cognition','short-event identity automatically passes if long-form conversion passes'],'full_completion':False})
 p=ConversionProbe(r);rows=[]
 for case in cases:
  if sha(case['source'])!=case['provenance']['audio_sha256']:raise ValueError('annotated enrollment changed')
  receipt=p.render(case['source'],d/(case['id']+'.wav'));rows.append({'case':case,'receipt':receipt});durable_json(d/'RESULTS.json',rows)
if __name__=='__main__':main()
