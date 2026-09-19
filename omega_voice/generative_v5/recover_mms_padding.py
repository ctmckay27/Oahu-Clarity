"""Apply the same fixed evaluator edge context to remaining MMS clock failures."""
import argparse,pathlib,json,copy
import librosa,numpy as np
from .mms_alignment import MMSAligner
from .alignment import AlignmentRejected
from .session import durable_json
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/intrinsic_alignment';source=r/'continuation/intrinsic_emphasis_evaluator';failed={x['id']for x in json.loads((d/'MMS_RESULTS.json').read_text())if not x['admitted']};cases=[x for x in json.loads((source/'FRESH_FEATURES.json').read_text())if x['case']['id']in failed];durable_json(d/'MMS_PADDING_PREDECLARED.json',{'cases':sorted(failed),'padding_s':.4,'confidence_min':.35,'original_audio_unmodified':True,'full_completion':False});al=MMSAligner(r/'continuation/alignment_padding/MMS_DEPENDENCIES.json');out=[]
 for c in cases:
  y,_=librosa.load(c['path'],sr=16000)
  try:report=al.align(np.pad(y,(6400,6400)),' '.join(c['case']['src_sentence']),sha(c['path']),True)
  except AlignmentRejected as e:report=e.report
  mapped=copy.deepcopy(report)
  for w in mapped['words']:
   for k in ['start','end']:w[k]-=.4
   for ch in w['characters']:
    for k in ['start','end']:ch[k]-=.4
  inside=all(0<=w['start']<w['end']<=len(y)/16000 for w in mapped['words']);mapped.update(evaluator_input_padding_s=.4,original_waveform_unmodified=True,admitted=bool(mapped['admitted']and inside));out.append({'id':c['case']['id'],'admitted':mapped['admitted'],'alignment':mapped});durable_json(d/'MMS_PAD_RESULTS.json',out);print(c['case']['id'],mapped['admitted'],mapped['minimum_word_probability'],flush=True)
if __name__=='__main__':main()
