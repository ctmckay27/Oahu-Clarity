"""Independent unconstrained CTC decoding localizes source/ASR/decoder failures.

No forced transcript and no relaxation of production intelligibility gates.
"""
import argparse,pathlib,json,itertools
import numpy as np,soundfile as sf,torch,torchaudio
from .alignment import ForcedAligner
from .session import durable_json
from ..causal_v4.evaluate import distance,intelligibility_tokens
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/seed_vc_cross_speaker';human={x['id']:x['human_transcript']for x in json.loads((d/'HUMAN_TRANSCRIPTS.json').read_text())};al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');torch.set_num_threads(3);rows=[]
 for k in ['A','B','D']:
  for kind,path in [('source',r/f'continuation/event_identity/{k}.enrollment.wav'),('converted',d/(k+'.wav'))]:
   if not path.exists():continue
   y,sr=sf.read(path,dtype='float32');x=torchaudio.functional.resample(torch.from_numpy(y),sr,16000)
   with torch.inference_mode():logits,_=al.model(x[None]);ids=logits[0].argmax(-1).tolist()
   decoded=''.join(al.labels[i]for i,_ in itertools.groupby(ids)if i).replace('|',' ').strip().lower();wer=distance(intelligibility_tokens(human[k]),intelligibility_tokens(decoded))/len(intelligibility_tokens(human[k]));row={'id':k,'kind':kind,'audio_sha256':sha(path),'decoded':decoded,'human_WER':wer,'method':'unconstrained greedy CTC argmax with duplicate/blank collapse','transcript_used_for_decoding':False,'no_completion_inference':True};rows.append(row);durable_json(d/'CTC_DIAGNOSTIC.json',rows);print(row,flush=True)
if __name__=='__main__':main()
