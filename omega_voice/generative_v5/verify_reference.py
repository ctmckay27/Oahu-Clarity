"""Test the native acoustic-prefix route with identical identity and transcript.

The reference condition changes only a diagnosed acoustic feature. These are
mechanism tests, never emotion presets or selected Mari performance assets.
"""
import argparse,json,pathlib
from .native import render
from .attack_calibration import attack_measure
from .acoustics import measure
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.runtime import ANCHOR
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root)
 d=r/'continuation/reference_heldout';d.mkdir(exist_ok=True)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 refs={'base':r/'continuation/calibration/pair_0_statement.wav',
       'firm':r/'continuation/attack_calibration/0_firm.reference.wav',
       'soft':r/'continuation/attack_calibration/0_soft.reference.wav'}
 source=refs['base'];source_receipt=source.with_suffix('.receipt.json')
 lineage={'source':str(source),'source_sha256':sha(source),'source_receipt_sha256':sha(source_receipt),
          'calibration_results_sha256':sha(r/'continuation/attack_calibration/RESULTS.json'),
          'selected_anchor_sha256':ANCHOR,'changes':'same reference text, duration, speaker; isolated onset amplitude envelope'}
 (d/'REFERENCE_LINEAGE.json').write_text(json.dumps(lineage,indent=2)+'\n')
 rows=[]
 for i,text in enumerate(['I know you have had a difficult day. We can finish this tomorrow.',
                          'Before we move the cabinet, check that the cable is clear of the door.']):
  for condition,ref in refs.items():
   out=d/f'{i}_{condition}.wav'
   reference={'audio':str(ref),'sha256':sha(ref),'text':'The package is ready.',
              'source_carrier_sha256':ANCHOR,'role':'same-speaker mechanism diagnostic; no performance selection'}
   if not out.exists():render(r,text,out,seed=93300+i,reference=reference)
   receipt=json.loads(out.with_suffix('.receipt.json').read_text())
   if receipt['reference']['sha256']!=sha(ref):raise ValueError('existing diagnostic does not match requested reference')
   result=ev.evaluate(out,text);result.update(attack=attack_measure(out),acoustics=measure(out))
   rows.append({'text':text,'condition':condition,'audio':str(out),'result':result})
   (d/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n')
   print(i,condition,'speaker',result['speaker_similarity'],'WER',result['wer'],'attack',result['attack']['initial_to_body_db'],flush=True)
if __name__=='__main__':main()
