"""Blind retained waveform/acting controls plus matched emphasis positions."""
import argparse,pathlib,json,shutil
from .voxtral_probe import VoxtralProbe
from .session import durable_json
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/voxtral_qualification';d.mkdir(exist_ok=True);old=json.loads((r/'continuation/minicpm_qualification/PREDECLARED.json').read_text());jobs=[x for x in old['jobs']if x['id']in {'carrier','silence','reversed','carrier_rhythm','mechanical_gaps','retained_rejected','contour_0','contour_1'}]
 refs=json.loads((r/'continuation/matched_emphasis/RESULTS.json').read_text())
 for row in refs:
  if row['case']['id']in {'ex01_emph_eval_00004','ex01_emph_eval_00005'}:
   jobs.append({'id':row['case']['id'],'source':row['path'],'source_sha256':sha(row['path']),'prompt':'Which single word in this recording receives the clearest vocal emphasis? Identify the word and describe the audible timing or pitch evidence. Do not select a word merely because it seems important in the sentence.','gold_emphasis_indices':row['gold']})
 durable_json(d/'PREDECLARED.json',{'jobs':jobs,'blinding':'numeric local filenames, no source/state labels supplied to the model','qualification_field':{'input_validity':'no hallucinated transcript on silence/reversal, carrier words preserved','rhythm':'mechanical insertion and retained rejected acting must not be praised as spontaneous living thought','prosody':'opposite measured contour signs and alternate labeled emphasis positions distinguished'},'scope':'qualification before any Mari character verdict; narrow passes remain separate','full_completion':False})
 probe=VoxtralProbe(r);durable_json(d/'EXECUTION_PROVENANCE.json',probe.provenance);results=[]
 for i,job in enumerate(jobs):
  source=pathlib.Path(job['source']);assert sha(source)==job['source_sha256'];audio=d/f'{i:03d}.wav';shutil.copyfile(source,audio);result=probe.infer(audio,job['prompt']);results.append({'id':job['id'],'result':result});durable_json(d/'RESULTS.json',results);print(job['id'],result['response'],flush=True)
 durable_json(d/'EXECUTION_COMPLETE.json',{'cases':len(results),'qualification_assessment_pending':True,'full_completion':False})
if __name__=='__main__':main()
