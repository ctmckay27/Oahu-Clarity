"""Matched-text checks for measured-word contour realization."""
import argparse,json,pathlib
from .physical_finality import realize
from .acoustics import measure
from .session import timeline_from_evaluation
from .alignment import ForcedAligner
from ..causal_v4.runtime import compile_scene
from ..causal_v4.evaluate import Evaluator

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--output',default='physical_finality_ctc');a=ap.parse_args();r=pathlib.Path(a.root)
 d=r/'continuation'/a.output;d.mkdir(exist_ok=True)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 aligner=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json')
 rows=[];alignments={}
 for i,row in enumerate(json.loads((r/'continuation/isolated_heldout/RESULTS.json').read_text())):
  text=row['text'];source=r/f'continuation/isolated_heldout/{i//2}_carrier.wav';base=row['baseline']
  if text not in alignments:alignments[text]=aligner.align(ev.load(source),text,base['audio']['sha256'],base['wer']==0)
  timeline=timeline_from_evaluation(dict(base,alignment=alignments[text]));plan=compile_scene(text,row['plan']['scene'],timeline=timeline);out=d/f'{i//2}_{row["condition"]}.wav'
  receipt=realize(source,out,plan) if not out.exists() else json.loads(out.with_suffix('.finality.json').read_text())
  result=ev.evaluate(out,text);result['acoustics']=measure(out)
  observed=aligner.align(ev.load(out),text,result['audio']['sha256'],result['wer']==0)
  error=max(abs(x[k]-y[k]) for x,y in zip(alignments[text]['words'],observed['words']) for k in ['start','end'])
  rows.append({'text':text,'condition':row['condition'],'audio':str(out),'result':result,'realization':receipt,'plan':plan,'alignment_error_s':error,'observed_alignment':observed})
  (d/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n');print(row['condition'],result['speaker_similarity'],result['wer'],error,result['acoustics'],flush=True)
if __name__=='__main__':main()
