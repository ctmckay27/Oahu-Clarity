"""Selected lineage's rich conditioning without the rejected prose directions."""
import argparse,pathlib,json
from .native import render,verify_graft_log
from .session import durable_json
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/graft_structure';d.mkdir(exist_ok=True)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');rows=[]
 cases=[('short','I thought the package was upstairs,',95000),('question','Did you leave the gate open?',95202),('heldout','The latch is loose. Hold the frame while I check the other side.',95402)]
 for name,text,seed in cases:
  p=d/(name+'.wav')
  try:
   if p.exists():
    log=p.with_suffix('.log');checks=verify_graft_log(log.read_text())
    if f'Text: "{text}"' not in log.read_text() or f'seed: {seed}\n' not in log.read_text():raise ValueError('existing diagnostic input differs')
    rp=p.with_suffix('.receipt.json')
    receipt=json.loads(rp.read_text()) if rp.exists() else {'role':'recovered diagnostic after incorrect frame-count audit; not original full build receipt',
      'log_sha256':sha(log),'audio_sha256':sha(p),'text':text,'seed':seed,'route_checks':checks}
    durable_json(p.with_suffix('.recovery.json'),receipt)
   else:receipt=render(r,text,p,seed,profile_mode='expressive_graft')
   q=ev.evaluate(p,text)
   row={'id':name,'text':text,'audio':str(p),'renderer':receipt,'quality':q,'full_completion':False};print(name,q['wer'],q['speaker_similarity'],flush=True)
  except Exception as ex:row={'id':name,'error':str(ex),'full_completion':False};print(name,str(ex),flush=True)
  rows.append(row);durable_json(d/'RESULTS.json',rows)
  if 'error' in row:break
 durable_json(d/'ASSESSMENT.json',{'all_quality':len(rows)==len(cases) and all('quality' in x and x['quality']['quality_screen_pass'] and x['quality']['wer']==0 for x in rows),'full_completion':False,'scope':'selected corrected graft without acting prose; controls and presence require separate testing'})
if __name__=='__main__':main()
