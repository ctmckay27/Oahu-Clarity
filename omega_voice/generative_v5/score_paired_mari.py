"""Frozen observer on Mari matched tests; no fitting on Mari waveforms."""
import argparse,pathlib,json
from .paired_emphasis import PairedEmphasisObserver
from .session import durable_json
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/paired_mari_evaluation';d.mkdir(exist_ok=True);base=r/'continuation/composed_session_replay';observer=PairedEmphasisObserver(r/'continuation/paired_emphasis_evaluator');records={}
 for name in ['first_only','both']:
  folder=base/name/'turn';receipt=json.loads((folder/'receipt.json').read_text());records[name]=(folder/'performance.wav',receipt['trials'][-1]['observed_alignment'])
 cases=[('add_second','first_only','both',{13:1}),('remove_second','both','first_only',{13:-1}),('identical','both','both',{})]
 durable_json(d/'PREDECLARED.json',{'cases':cases,'all_unlisted_words_expected_unchanged':True,'earlier_upstairs_focus_expected_unchanged':True,'model_frozen':True,'model_sha256':observer.qualification['model_sha256'],'full_completion':False});out=[]
 for name,before,after,expected in cases:
  p,pa=records[before];q,qa=records[after];result=observer.compare(p,pa,q,qa);errors=[w for w in result['words']if w['change']!=expected.get(w['index'],0)];out.append({'id':name,'result':result,'expected':expected,'errors':errors,'pass':not errors});durable_json(d/'RESULTS.json',out);print(name,'changes',[(w['word'],w['change'])for w in result['words']if w['change']],'pass',not errors,flush=True)
 # Different text under equivalent contrast: score the already retained,
 # quality-qualified source and duration-only target without rerendering.
 earlier=json.loads((r/'continuation/contrast_duration/RESULTS.json').read_text());extra=[]
 for row in earlier:
  if row['variant']!='duration_only':continue
  print('retained single case available',row['source'],flush=True)
 durable_json(d/'ASSESSMENT.json',{'composed_paired_acoustic_gate':all(x['pass']for x in out),'full_completion':False,'character_presence_admitted':False})
if __name__=='__main__':main()
