"""Test missing acoustic context at evaluator edges; output speech is untouched."""
import argparse,pathlib,json,copy
import numpy as np,librosa
from .alignment import ForcedAligner,AlignmentRejected
from .session import durable_json
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/alignment_padding';d.mkdir(exist_ok=True);sources=[r/'continuation/prosodic_evaluator/FEATURES.json',r/'continuation/paired_emphasis_evaluator/FRESH_FEATURES.json'];cases=[x for p in sources for x in json.loads(p.read_text())if not x['admitted']];padding=.4
 durable_json(d/'PREDECLARED.json',{'cases':[x['case']['id']for x in cases],'padding_s':padding,'mechanism':'zero context supplied only to the CTC evaluator at both waveform edges; measured boundaries shifted back to original audio clock','gates':{'same_original_threshold':.35,'all_tokens_inside_original_audio':True,'actual_waveform_unchanged':True},'purpose':'distinguish short/edge context failure from phonetic/transcript failure; no weaker alignment confidence threshold','full_completion':False})
 al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');rows=[]
 for c in cases:
  p=pathlib.Path(c['path']);wave,_=librosa.load(p,sr=16000);pad=round(padding*16000);observed=np.pad(wave,(pad,pad));error=None
  try:report=al.align(observed,' '.join(c['case']['src_sentence']),sha(p),True)
  except AlignmentRejected as e:report=e.report;error='unchanged confidence gate failed'
  mapped=copy.deepcopy(report)
  for w in mapped['words']:
   for k in ['start','end']:w[k]-=padding
   for ch in w['characters']:
    for k in ['start','end']:ch[k]-=padding
  inside=all(0<=w['start']<w['end']<=len(wave)/16000 for w in mapped['words']);mapped['evaluator_input_padding_s']=padding;mapped['original_waveform_unmodified']=True;mapped['original_audio_sha256']=sha(p);mapped['admitted']=bool(report['admitted']and inside);row={'id':c['case']['id'],'source_feature_record':c['case'],'original_clock_min_probability':c['rejected_alignment']['minimum_word_probability'],'padded_min_probability':report['minimum_word_probability'],'admitted':mapped['admitted'],'inside_original_audio':inside,'error':error,'alignment':mapped};rows.append(row);durable_json(d/'RESULTS.json',rows);print(c['case']['id'],row['admitted'],row['original_clock_min_probability'],row['padded_min_probability'],flush=True)
 durable_json(d/'ASSESSMENT.json',{'all_recovered':all(x['admitted']for x in rows),'recovered':sum(x['admitted']for x in rows),'total':len(rows),'full_completion':False})
if __name__=='__main__':main()
