"""Blinded retained controls before any character or acting judgment."""
import argparse,pathlib,json,shutil
from .phi4_probe import Phi4Probe
from .session import durable_json
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/phi4_qualification';d.mkdir(exist_ok=True)
 original=json.loads((r/'continuation/voxtral_qualification/PREDECLARED.json').read_text());jobs=original['jobs'];durable_json(d/'PREDECLARED.json',dict(original,evaluator='official Phi-4-multimodal-instruct pinned weights with eager CPU attention',jobs=jobs,prior_answers_hidden=True,maximum_new_tokens=100,qualification_pending=True))
 probe=Phi4Probe(r);durable_json(d/'EXECUTION_PROVENANCE.json',probe.provenance);results=[]
 for i,job in enumerate(jobs):
  p=pathlib.Path(job['source'])
  if sha(p)!=job['source_sha256']:raise ValueError('control source changed')
  blinded=d/f'{i:03d}.wav';shutil.copyfile(p,blinded);result=probe.infer(blinded,job['prompt']);results.append({'id':job['id'],'result':result});durable_json(d/'RESULTS.json',results);print(job['id'],result['response'],flush=True)
 durable_json(d/'EXECUTION_COMPLETE.json',{'cases':len(results),'assessment_pending':True,'full_completion':False})
if __name__=='__main__':main()
