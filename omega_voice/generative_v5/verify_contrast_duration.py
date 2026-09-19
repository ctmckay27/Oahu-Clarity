"""Fixed duration-only versus duration+contour/pressure mechanism ablation."""
import argparse,json,pathlib
import numpy as np,soundfile as sf
from .session import durable_json,timeline_from_evaluation
from .alignment import ForcedAligner
from .prominence_duration import realize
from .prominence import realize as contour
from ..causal_v4.runtime import compile_scene
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/contrast_duration';d.mkdir(exist_ok=True);cases=[x for x in json.loads((r/'continuation/contrast_focus/WAVE_RESULTS.json').read_text())if x['case']=='correct'];cal=r/'continuation/matched_emphasis/SUMMARY.json'
 durable_json(d/'PREDECLARED.json',{'cases':[{'source':x['source'],'text':x['text'],'focus':x['focus']}for x in cases],'calibration_sha256':sha(cal),'variants':['duration_only','duration_plus_contour_pressure'],'gates':{'WER':0,'identity_min':.6013473320007324,'clock_error_max_s':.08,'UTMOS_loss_max':.15,'no_earlier_PCM_changes':True,'emphasized_target':True},'reason':'matched references show consistent word duration increase and inconsistent mean pitch/level increase; distinguish missing timing mechanism from arbitrary stronger gain','full_completion':False})
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');rows=json.loads((d/'RESULTS.json').read_text()) if (d/'RESULTS.json').exists() else []
 for case in cases:
  i=case['source'];p=case['plan'];source=pathlib.Path(case['source_path']);out=d/f'{i}_duration.wav'
  cached=next((x for x in rows if x['source']==i and x['variant']=='duration_only'),None)
  if cached:
   receipt=cached['receipt'];q=cached['quality'];alignment=cached['alignment'];assert sha(out)==receipt['output_sha256'] and sha(source)==receipt['input_sha256']
  else:
   receipt=realize(source,out,p,cal);q=ev.evaluate(out,case['text']);alignment=al.align(ev.load(out),case['text'],sha(out),q['wer']==0)
  clock_error=max(abs(x[k]-y[k])for x,y in zip(receipt['mapped_timeline']['words'],alignment['words'])for k in ['start','end'])
  x,_=sf.read(source,dtype='int16');y,_=sf.read(out,dtype='int16');a=receipt['windows'][0]['start_sample'];b=receipt['windows'][-1]['end_sample'];extra=receipt['added_samples'];untouched=np.array_equal(x[:a],y[:a])and np.array_equal(x[b:],y[b+extra:])
  if cached:rows.remove(cached)
  rows.append({'source':i,'variant':'duration_only','text':case['text'],'focus':case['focus'],'audio':str(out),'receipt':receipt,'quality':q,'alignment':alignment,'clock_error_s':clock_error,'unchanged_context_exact':untouched});durable_json(d/'RESULTS.json',rows);print(i,'duration',q['wer'],q['speaker_similarity'],clock_error,untouched,flush=True)
  combined=d/f'{i}_combined.wav';new_plan=compile_scene(case['text'],p['scene'],p['initial_state'],timeline=timeline_from_evaluation(dict(q,alignment=alignment)),policy='contrast_focus_v7');cr=contour(out,combined,new_plan) if not combined.exists() else json.loads(combined.with_suffix('.prominence.json').read_text());assert sha(combined)==cr['output_sha256'];cq=ev.evaluate(combined,case['text']);ca=al.align(ev.load(combined),case['text'],sha(combined),True) if cq['wer']==0 else None
  rows.append({'source':i,'variant':'duration_plus_contour_pressure','text':case['text'],'focus':case['focus'],'audio':str(combined),'receipt':cr,'quality':cq,'alignment':ca,'clock_error_s':max(abs(x[k]-y[k])for x,y in zip(alignment['words'],ca['words'])for k in ['start','end']) if ca else None,'failure':'intelligibility' if cq['wer']!=0 else None});durable_json(d/'RESULTS.json',rows);print(i,'combined',cq['wer'],cq['speaker_similarity'],flush=True)
 durable_json(d/'ASSESSMENT.json',{'quality_and_clock_pass':all(x['quality']['wer']==0 and x['quality']['identity_pass'] and x['clock_error_s'] is not None and x['clock_error_s']<=.08 for x in rows),'emphasis_and_UTMOS_pending':True,'full_completion':False})
if __name__=='__main__':main()
