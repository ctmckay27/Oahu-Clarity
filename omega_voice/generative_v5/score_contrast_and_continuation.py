"""Independent acoustic/naturalness observations for causal mechanism tests."""
import argparse,pathlib,json
from .session import durable_json
from .naturalness_observer import NaturalnessObserver
from .emphasis_evaluator import EmphasisEvaluator,word_observations

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('stage',choices=['continuation_mos','contrast_mos','emphasis']);a=ap.parse_args();r=pathlib.Path(a.root)
 if a.stage=='emphasis':
  q=json.loads((r/'continuation/emphasis_qualification/ASSESSMENT.json').read_text())
  if not q['admitted_for_acoustic_emphasis']:raise ValueError('emphasis observer not qualified')
  ev=EmphasisEvaluator(r);d=r/'continuation/contrast_focus';rows=[]
  for case in json.loads((d/'WAVE_RESULTS.json').read_text()):
   if case['case'] not in {'baseline','correct'}:continue
   p=d/f"{case['source']}_{case['case']}.wav";result=ev.infer(p);obs=word_observations(result,case['alignment']);row={'source':case['source'],'case':case['case'],'focus':case['focus'],'result':result,'words':obs};rows.append(row);durable_json(d/'EMPHASIS_RESULTS.json',rows);print(case['source'],case['case'],[(w['word'],round(w['mean_probability'] or 0,3),w['emphasized'])for w in obs],flush=True)
  checks=[]
  for i in range(2):
   base=next(x for x in rows if x['source']==i and x['case']=='baseline');changed=next(x for x in rows if x['source']==i and x['case']=='correct');f=base['focus'];b=base['words'][f];c=changed['words'][f]
   checks.append({'source':i,'focus':c['word'],'base_probability':b['mean_probability'],'changed_probability':c['mean_probability'],'probability_increases':c['mean_probability']>b['mean_probability'],'base_emphasized':b['emphasized'],'changed_emphasized':c['emphasized']})
  durable_json(d/'EMPHASIS_ASSESSMENT.json',{'checks':checks,'all_increase':all(x['probability_increases']for x in checks),'scope':'acoustic contrast only; not evidence of dramatic appropriateness or Mari character','full_completion':False});return
 ev=NaturalnessObserver();rows=[]
 if a.stage=='continuation_mos':
  d=r/'continuation/seed_causal_continuation'
  for k in ['no_result','related']:
   source=ev.score(r/f'continuation/factive_scene/0_{k}.wav');changed=ev.score(d/(k+'.joined.wav'));rows.append({'case':k,'source':source,'changed':changed,'loss':source['predicted_MOS']-changed['predicted_MOS']});durable_json(d/'UTMOS_RESULTS.json',{'provenance':ev.provenance,'rows':rows});print(rows[-1],flush=True)
 else:
  d=r/'continuation/contrast_focus'
  for i in range(2):
   source=ev.score(d/f'{i}_baseline.wav');changed=ev.score(d/f'{i}_correct.wav');rows.append({'source':i,'baseline':source,'changed':changed,'loss':source['predicted_MOS']-changed['predicted_MOS']});durable_json(d/'UTMOS_RESULTS.json',{'provenance':ev.provenance,'rows':rows});print(rows[-1],flush=True)
 durable_json(d/'UTMOS_ASSESSMENT.json',{'all_loss_within_gate':all(x['loss']<=.15 for x in rows),'max_loss':.15,'scope':'predicted naturalness; no direct listening or character assessment','full_completion':False})
if __name__=='__main__':main()
