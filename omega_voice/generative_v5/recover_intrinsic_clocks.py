"""Reuse the already discriminated clock-recovery policy on a new holdout."""
import argparse,pathlib,json,copy,gc
import numpy as np,librosa
from .alignment import ForcedAligner,AlignmentRejected
from .mms_alignment import MMSAligner
from .session import durable_json
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);source=r/'continuation/intrinsic_emphasis_evaluator';d=r/'continuation/intrinsic_alignment';d.mkdir(exist_ok=True);cases=[x for x in json.loads((source/'FRESH_FEATURES.json').read_text())if not x['admitted']];durable_json(d/'PREDECLARED.json',{'cases':[x['case']['id']for x in cases],'policy':'unchanged400ms evaluator padding, then pinned MMS alignment-trained model on original audio','word_probability_min':.35,'model_frozen_no_refit':True,'full_completion':False});al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');rows=[]
 for c in cases:
  wave,_=librosa.load(c['path'],sr=16000)
  try:report=al.align(np.pad(wave,(6400,6400)),' '.join(c['case']['src_sentence']),sha(c['path']),True)
  except AlignmentRejected as e:report=e.report
  mapped=copy.deepcopy(report)
  for w in mapped['words']:
   for key in ['start','end']:w[key]-=.4
   for ch in w['characters']:
    for key in ['start','end']:ch[key]-=.4
  inside=all(0<=w['start']<w['end']<=len(wave)/16000 for w in mapped['words']);mapped.update(evaluator_input_padding_s=.4,original_waveform_unmodified=True,admitted=bool(report['admitted']and inside));rows.append({'id':c['case']['id'],'admitted':mapped['admitted'],'alignment':mapped});durable_json(d/'RESULTS.json',rows);print('pad',c['case']['id'],mapped['admitted'],mapped['minimum_word_probability'],flush=True)
 del al;gc.collect();al=MMSAligner(r/'continuation/alignment_padding/MMS_DEPENDENCIES.json');left={x['id']for x in rows if not x['admitted']};out=[]
 for c in cases:
  if c['case']['id']not in left:continue
  wave,_=librosa.load(c['path'],sr=16000)
  try:report=al.align(wave,' '.join(c['case']['src_sentence']),sha(c['path']),True)
  except AlignmentRejected as e:report=e.report
  out.append({'id':c['case']['id'],'admitted':report['admitted'],'alignment':report});durable_json(d/'MMS_RESULTS.json',out);print('mms',c['case']['id'],report['admitted'],report['minimum_word_probability'],flush=True)
 if not out:durable_json(d/'MMS_RESULTS.json',[])
 durable_json(d/'ASSESSMENT.json',{'all_recovered':all(x['admitted']or any(q['id']==x['id']and q['admitted']for q in out)for x in rows),'full_completion':False})
if __name__=='__main__':main()
