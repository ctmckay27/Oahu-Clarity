"""Lexical/event separation without laundering nonlexical meaning into ASR."""
import argparse,pathlib,json
import soundfile as sf
from .session import durable_json
from ..causal_v4.evaluate import Evaluator,normalized
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--case-directory',default='native_ack_episode',choices=['native_ack_episode','ack_counterfactual']);a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation'/a.case_directory;cases=json.loads((d/'PREDECLARED.json').read_text())['cases'];ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');rows=[]
 for c in cases:
  p=d/(c['id']+'.wav');q=ev.evaluate(p,c['text']);hyp=normalized(q['asr_text']);expected=normalized(c['continuation']);suffix=bool(len(hyp)>=len(expected)and hyp[-len(expected):]==expected);prefix=hyp[:-len(expected)]if suffix else None
  row={'id':c['id'],'audio_sha256':sha(p),'requested_event_form':c['event_form'],'quality':q,'lexical_continuation_exact':suffix,'preceding_ASR_tokens':prefix,'event_meaning_admitted':False,'short_event_identity_admitted':False,'full_completion':False}
  # A diagnostic crop is only an ASR-estimated interval. It cannot supply an
  # actuation clock or identity/meaning verdict for the short event itself.
  if suffix and prefix:
   words=q['alignment']['words'];idx=next((i for i in range(len(words))if [x['word']for x in words[i:]]==expected),None)
   if idx is not None and idx>0:
    end=words[idx]['start'];wave,sr=sf.read(p,dtype='int16');crop=d/(c['id']+'_event_estimate.wav');sf.write(crop,wave[:round(end*sr)],sr,subtype='PCM_16');row['estimated_event_interval']={'start':0.,'end':end,'method':'ASR boundary estimate only; not independently verified CTC','audio':str(crop),'sha256':sha(crop)}
  rows.append(row);durable_json(d/'QUALITY_RESULTS.json',rows);print(c['id'],q['asr_text'],q['speaker_similarity'],suffix,prefix,flush=True)
 durable_json(d/'QUALITY_ASSESSMENT.json',{'whole_episode_identity':all(x['quality']['identity_pass']for x in rows),'lexical_continuations_retained':all(x['lexical_continuation_exact']for x in rows),'event_ASR_reported':True,'meaning_and_short_identity_unverified':True,'full_completion':False})
if __name__=='__main__':main()
