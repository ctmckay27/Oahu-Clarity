"""Discriminate missing pitch excursion from mere duration at fixed semantic strength."""
import argparse,pathlib,json
import numpy as np,soundfile as sf
from .prominence_duration import realize
from .alignment import ForcedAligner
from .paired_emphasis import PairedEmphasisObserver
from .clause_asr import verify_clauses
from .session import durable_json
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.runtime import compile_scene
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/coupled_prominence';d.mkdir(exist_ok=True);source=r/'continuation/text_release/carrier_parity.wav';cal=r/'continuation/matched_emphasis/SUMMARY.json';gesture=r/'continuation/coupled_prominence_calibration/CALIBRATION.json';old=json.loads((r/'continuation/two_contrasts/RESULTS.json').read_text());cases=[{'id':x['id'],'plan':x['plan']}for x in old];ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');observer=PairedEmphasisObserver(r/'continuation/paired_emphasis_evaluator');text=cases[0]['plan']['text'];durable_json(d/'PREDECLARED.json',{'cases':[x['id']for x in cases],'gesture_sha256':sha(gesture),'existing_duration_mapping_unchanged':True,'control_strength_unchanged':.55,'calibration_disjoint_from_observer_training_text':True,'gates':{'original_full_quality':True,'exact_whole_or_each_clause_ASR':True,'word_clock_error_max_s':.08,'no_pre_event_difference':True,'paired_blue_change':True,'UTMOS_loss_max':.15},'full_completion':False});rows=[]
 for c in cases:
  for joint in [False,True]:
   name=c['id']+('_joint'if joint else'_duration');p=d/(name+'.wav')
   try:
    receipt=realize(source,p,c['plan'],cal,gesture=gesture if joint else None,gesture_sha256=sha(gesture)if joint else None);quality=ev.evaluate(p,text);lexical=quality['wer']==0;clause=None
    if not lexical:clause=verify_clauses(p,text,receipt['mapped_timeline'],ev);lexical=clause['admitted']
    if not quality['quality_screen_pass']or not lexical:raise ValueError('independent lexical/identity quality gate failed')
    alignment=al.align(ev.load(p),text,sha(p),True);error=max(abs(x[k]-y[k])for x,y in zip(receipt['mapped_timeline']['words'],alignment['words'])for k in ['start','end']);row={'id':name,'audio':str(p),'receipt':receipt,'quality':quality,'clause_ASR':clause,'alignment':alignment,'clock_error_s':error,'pass':error<=.08}
   except Exception as e:row={'id':name,'pass':False,'error':repr(e)}
   rows.append(row);durable_json(d/'RESULTS.json',rows);print(name,row['pass'],row.get('error'),flush=True)
 byid={x['id']:x for x in rows};pairs=[]
 for suffix in ['duration','joint']:
  aa=byid['first_only_'+suffix];bb=byid['both_'+suffix]
  if not(aa['pass']and bb['pass']):continue
  result=observer.compare(aa['audio'],aa['alignment'],bb['audio'],bb['alignment']);expected={13:1};ok=all(w['change']==expected.get(w['index'],0)for w in result['words']);a,sr=sf.read(aa['audio'],dtype='int16');b,_=sf.read(bb['audio'],dtype='int16');cut=bb['receipt']['windows'][1]['delivered_start_sample'];prefix=np.array_equal(a[:cut],b[:cut]);pairs.append({'mechanism':suffix,'result':result,'expected':expected,'paired_acoustic_pass':ok,'past_prefix_exact':bool(prefix)});print('paired',suffix,ok,prefix,[(w['word'],w['change'])for w in result['words']if w['change']],flush=True)
 durable_json(d/'PAIRED_RESULTS.json',pairs);durable_json(d/'ASSESSMENT.json',{'quality_and_clock':all(x['pass']for x in rows),'paired_joint_pass':any(x['mechanism']=='joint'and x['paired_acoustic_pass']and x['past_prefix_exact']for x in pairs),'naturalness_pending':True,'full_completion':False})
if __name__=='__main__':main()
