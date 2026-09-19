"""Discriminate altered phonetics from ASR context sensitivity; no gate rewrite."""
import argparse,pathlib,json
import numpy as np,soundfile as sf
from ..causal_v4.evaluate import Evaluator,normalized,distance
from ..causal_v4.renderer import sha
from .session import durable_json

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/two_contrasts_deterministic';cases=json.loads((d/'RESULTS.json').read_text());source=r/'continuation/text_release/carrier_parity.wav';ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');x,sr=sf.read(source,dtype='int16');rows=[]
 for c in [cases[0],cases[2]]:
  y,_=sf.read(c['audio'],dtype='int16');clock=c['plan']['realization_timeline'];cut=round(clock['words'][7]['start']*sr);shift=sum(w['extra_samples']for w in c['receipt']['windows']if w['word']<7);before=x[cut:];after=y[cut+shift:];samples_exact=bool(np.array_equal(before,after));suffix_text='I can see it beside the blue bowl.';observations=[]
  for label,wave in [('source_suffix',before),('output_suffix',after)]:
   p=d/(c['id']+'_'+label+'.wav');sf.write(p,wave,sr,subtype='PCM_16');segments,_=ev.asr.transcribe(ev.load(p),language='en',beam_size=5,temperature=0,condition_on_previous_text=False,vad_filter=False,word_timestamps=True);hyp=' '.join(s.text.strip()for s in segments);observations.append({'role':label,'audio':str(p),'sha256':sha(p),'text':hyp,'WER':distance(normalized(suffix_text),normalized(hyp))/len(normalized(suffix_text))})
  row={'id':c['id'],'suffix_samples_exact':samples_exact,'suffix_observations':observations,'interpretation':'If exact samples yield matching suffix ASR but differing whole ASR, the unmodified suffix discrepancy is decoder-context sensitivity. Modified segments still require independent content gates.','full_utterance_WER_gate_changed':False,'full_completion':False};rows.append(row);durable_json(d/'ASR_CONTEXT_DIAGNOSIS.json',rows);print(c['id'],samples_exact,[o['text']for o in observations],flush=True)
if __name__=='__main__':main()
