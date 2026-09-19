"""Independent source transcription for retained rejected calibration clocks."""
import argparse,pathlib,json
from ..causal_v4.evaluate import Evaluator,normalized,distance
from .session import durable_json

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/prosodic_evaluator';rows=json.loads((d/'FEATURES.json').read_text());ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');out=[]
 for x in rows:
  if x['admitted']:continue
  segments,_=ev.asr.transcribe(ev.load(x['path']),language='en',beam_size=5,temperature=0,condition_on_previous_text=False,vad_filter=False,word_timestamps=True);segments=list(segments);hyp=' '.join(s.text.strip()for s in segments);text=' '.join(x['case']['src_sentence']);wer=distance(normalized(text),normalized(hyp))/len(normalized(text));bad=[{'word':w['word'],'probability':w['probability']}for w in x['rejected_alignment']['words']if w['probability']<.35];rec={'id':x['case']['id'],'expected':text,'ASR':hyp,'WER':wer,'ASR_word_times':[{'word':w.word,'start':w.start,'end':w.end,'probability':w.probability}for s in segments for w in s.words or []],'failed_CTC_words':bad,'source_labels_rewritten':False,'alignment_admitted':False};out.append(rec);durable_json(d/'REJECTED_CLOCK_ASR.json',out);print(rec['id'],hyp,wer,bad,flush=True)
if __name__=='__main__':main()
