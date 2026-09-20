"""Discriminate compound-instruction failure from actual acoustic task failure.

Official ASR task plus single-question choices; all original failures remain.
This is evaluator-interface qualification, never renderer direction.
"""
import argparse,pathlib,json,shutil
from .phi4_probe import Phi4Probe
from .session import durable_json
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/phi4_task_qualification';d.mkdir(exist_ok=True);old=json.loads((r/'continuation/minicpm_qualification/PREDECLARED.json').read_text());byid={x['id']:x for x in old['jobs']};jobs=[]
 for name in ['carrier','silence','reversed']:
  c=byid[name];jobs.append(dict(c,id=name+'_asr',prompt='Transcribe the audio clip into text.'))
 for name in ['carrier_rhythm','mechanical_gaps','retained_rejected']:
  c=byid[name];jobs.append(dict(c,id=name+'_rhythm',prompt='Which best describes the vocal timing? A: Smoothly flowing speech. B: Noticeable mechanical gaps interrupting speech. C: Conspicuously acted hesitations. D: No speech. Answer with one letter.'))
 for name in ['contour_0','contour_1','contour_2','contour_3']:
  c=byid[name];jobs.append(dict(c,id=name+'_pitch',prompt='At the end of the last spoken word, does the pitch A: rise, B: fall, or C: stay level? Answer with one letter.'))
 refs=json.loads((r/'continuation/voxtral_qualification/PREDECLARED.json').read_text())['jobs']
 for c in refs:
  if'gold_emphasis_indices'in c:jobs.append(dict(c,id=c['id']+'_focus',prompt='Which word is spoken with the strongest emphasis? Answer with only that word.'))
 durable_json(d/'PREDECLARED.json',{'jobs':jobs,'source_interface':'https://huggingface.co/microsoft/Phi-4-multimodal-instruct, official ASR task and Appendix A single-choice benchmark interface','original_compound_instruction_failures_retained':True,'no_renderer_change':True,'no_labels_or_prior_answers_given':True,'single_fixed_successor_interface_no_prompt_search':True,'gates':'correct carrier transcription; no invented intelligible speech on silence/reversal; distinguish three timing controls; four measured contour signs; both publisher emphasis positions','full_completion':False});probe=Phi4Probe(r);out=[]
 for i,c in enumerate(jobs):
  p=pathlib.Path(c['source'])
  if sha(p)!=c['source_sha256']:raise ValueError('source control changed')
  q=d/f'{i:03d}.wav';shutil.copyfile(p,q);res=probe.infer(q,c['prompt'],max_new_tokens=60);out.append({'id':c['id'],'result':res});durable_json(d/'RESULTS.json',out);print(c['id'],res['response'],flush=True)
 durable_json(d/'EXECUTION_COMPLETE.json',{'cases':len(out),'assessment_pending':True,'full_completion':False})
if __name__=='__main__':main()
