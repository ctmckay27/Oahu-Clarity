"""Change the alignment representation for known graphemic failures, retain all results."""
import argparse,pathlib,json
import librosa
from .phoneme_clock import PhonemeAligner
from .alignment import AlignmentRejected
from .session import durable_json
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/phoneme_alignment';fresh=json.loads((r/'continuation/intrinsic_emphasis_evaluator/FRESH_FEATURES.json').read_text());failed={x['id']for x in json.loads((r/'continuation/intrinsic_alignment/MMS_PAD_RESULTS.json').read_text())if not x['admitted']};cases=[x for x in fresh if x['case']['id']in failed]+[x for x in fresh if x['admitted']][:8];durable_json(d/'PREDECLARED.json',{'cases':[x['case']['id']for x in cases],'word_probability_min':.35,'primary_sources':['https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft','https://huggingface.co/docs/transformers/model_doc/wav2vec2_phoneme'],'reason':'grapheme-to-phone ambiguity and reduced pronunciation in via/email/you; phonetic units instead of forced spelling','no_Mari_voice_generation':True,'full_completion':False});al=PhonemeAligner('/tmp/mari-phoneme-alignment',d/'MODEL_PROVENANCE.json');durable_json(d/'RUNTIME_PROVENANCE.json',al.provenance);out=[]
 for x in cases:
  y,_=librosa.load(x['path'],sr=16000)
  try:report=al.align(y,' '.join(x['case']['src_sentence']),sha(x['path']),True)
  except AlignmentRejected as e:report=e.report
  row={'id':x['case']['id'],'admitted':report['admitted'],'alignment':report,'was_admitted':x['admitted']};out.append(row);durable_json(d/'RESULTS.json',out);print(x['case']['id'],report['minimum_word_probability'],report['admitted'],report['unprompted_greedy_phones'],flush=True)
 durable_json(d/'ASSESSMENT.json',{'all_target_clocks_admitted':all(x['admitted']for x in out if x['id']in failed),'control_clocks_admitted':all(x['admitted']for x in out if x['id']not in failed),'full_completion':False})
if __name__=='__main__':main()
