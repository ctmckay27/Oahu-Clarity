"""Independent lexical and alternate pinned CTC checks for rejected reference clocks."""
import argparse,pathlib,json
import librosa
from .alignment import ForcedAligner,AlignmentRejected
from .session import durable_json
from ..causal_v4.evaluate import Evaluator,normalized,distance
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/alignment_padding';fresh=json.loads((r/'continuation/paired_emphasis_evaluator/FRESH_FEATURES.json').read_text());cases=[x for x in fresh if not x['admitted']]
 durable_json(d/'ALTERNATE_PREDECLARED.json',{'cases':[x['case']['id']for x in cases],'mechanism':'independent Whisper transcript and previously pinned base CTC model, no modified input or confidence threshold','gates':{'CTC_word_probability_min':.35,'Whisper_WER':0},'frozen_paired_model':sha(r/'continuation/paired_emphasis_evaluator/MODEL.joblib'),'full_completion':False})
 al=ForcedAligner('/tmp/mari-alignment',r/'continuation/FORCED_ALIGNMENT_DEPENDENCIES.json');ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');out=[]
 for c in cases:
  p=pathlib.Path(c['path']);text=' '.join(c['case']['src_sentence']);wave,_=librosa.load(p,sr=16000)
  seg,_=ev.asr.transcribe(wave,language='en',beam_size=5,temperature=0,condition_on_previous_text=False,vad_filter=False,word_timestamps=True);seg=list(seg);hyp=' '.join(s.text.strip()for s in seg);wer=distance(normalized(text),normalized(hyp))/len(normalized(text))
  try:alignment=al.align(wave,text,sha(p),True)
  except AlignmentRejected as e:alignment=e.report
  row={'id':c['case']['id'],'expected':text,'ASR':hyp,'WER':wer,'ASR_word_times':[{'word':w.word,'start':w.start,'end':w.end,'probability':w.probability}for s in seg for w in s.words or []],'alignment':alignment,'admitted':alignment['admitted']and wer==0};out.append(row);durable_json(d/'ALTERNATE_RESULTS.json',out);print(row['id'],hyp,wer,alignment['minimum_word_probability'],row['admitted'],flush=True)
if __name__=='__main__':main()
