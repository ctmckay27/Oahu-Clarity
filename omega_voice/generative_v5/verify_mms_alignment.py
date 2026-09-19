"""Alignment representation discriminator; original rejected evidence is retained."""
import argparse,pathlib,json
import librosa
from .mms_alignment import MMSAligner
from .alignment import AlignmentRejected
from .session import durable_json
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/alignment_padding';sources=[r/'continuation/prosodic_evaluator/FEATURES.json',r/'continuation/paired_emphasis_evaluator/FRESH_FEATURES.json'];allrows=[x for p in sources for x in json.loads(p.read_text())];rejected=[x for x in allrows if not x['admitted']];controls=[x for x in allrows if x['admitted']][:8];cases=rejected+controls
 durable_json(d/'MMS_PREDECLARED.json',{'rejected':[x['case']['id']for x in rejected],'controls':[x['case']['id']for x in controls],'mechanism':'alignment-trained MMS romanized targets; original unmodified reference waveform and transcript','minimum_word_probability':.35,'boundary_comparison_controls':8,'frozen_paired_model_sha256':sha(r/'continuation/paired_emphasis_evaluator/MODEL.joblib'),'full_completion':False})
 al=MMSAligner(d/'MMS_DEPENDENCIES.json');out=[]
 for x in cases:
  p=pathlib.Path(x['path']);assert sha(p)==x['audio_sha256'];wave,_=librosa.load(p,sr=16000)
  try:report=al.align(wave,' '.join(x['case']['src_sentence']),sha(p),True)
  except AlignmentRejected as e:report=e.report
  old=x['alignment']if x['admitted']else x['rejected_alignment'];err=max(abs(a[k]-b[k])for a,b in zip(old['words'],report['words'])for k in ['start','end']);row={'id':x['case']['id'],'was_admitted':x['admitted'],'alignment':report,'admitted':report['admitted'],'previous_boundary_max_difference_s':err};out.append(row);durable_json(d/'MMS_RESULTS.json',out);print(row['id'],report['minimum_word_probability'],row['admitted'],err,flush=True)
 durable_json(d/'MMS_ASSESSMENT.json',{'recovered':sum(x['admitted']and not x['was_admitted']for x in out),'rejected_count':len(rejected),'controls_admitted':all(x['admitted']for x in out if x['was_admitted']),'control_max_boundary_difference_s':max(x['previous_boundary_max_difference_s']for x in out if x['was_admitted']),'full_completion':False})
if __name__=='__main__':main()
