"""Independent clause ASR with complete sample coverage and a measured clock.

Whole-utterance ASR remains separately reported. This route diagnoses decoder
context errors without changing words, dropping difficult intervals, prompting
the recognizer with the answer, or treating a forced alignment as transcription.
"""
import pathlib,re
import numpy as np,soundfile as sf
from ..causal_v4.evaluate import normalized,distance
from ..causal_v4.renderer import sha

def verify_clauses(audio,text,timeline,evaluator):
 audio=pathlib.Path(audio);y,sr=sf.read(audio,dtype='float32')
 if y.ndim!=1 or sr!=24000 or not np.isfinite(y).all():raise ValueError('24k mono finite waveform required')
 if timeline.get('source_audio_sha256')!=sha(audio):raise ValueError('unbound clause clock')
 tokens=normalized(text);matches=list(re.finditer(r"[a-z0-9]+(?:'[a-z0-9]+)?",text.lower().replace('’',"'")))
 words=timeline['words']
 if len(words)!=len(tokens)or len(matches)!=len(tokens):raise ValueError('transcript/clock coverage mismatch')
 if abs(timeline.get('duration_s',-1)-len(y)/sr)>1/sr:raise ValueError('clause clock duration mismatch')
 for i,w in enumerate(words):
  if not all(isinstance(w.get(k),(int,float))and np.isfinite(w[k])for k in ['start','end']):raise ValueError('invalid measured word time')
  if not 0<=w['start']<w['end']<=len(y)/sr+.001 or(i and w['start']<words[i-1]['start']):raise ValueError('invalid word ordering or bounds')
 ends=[i+1 for i in range(len(matches)-1)if re.search(r'[.!?]',text[matches[i].end():matches[i+1].start()])]+[len(tokens)]
 if len(ends)<2:raise ValueError('independent clause verification needs explicit sentence boundaries')
 cuts=[0];boundaries=[]
 for i in ends[:-1]:
  left=round(words[i-1]['end']*sr);right=round(words[i]['start']*sr)
  if not 0<=left<right<=len(y)or right-left<round(.08*sr):raise ValueError('no measured quiet clause boundary')
  half=round(.02*sr);positions=range(left+half,right-half+1,max(1,round(.01*sr)));threshold=max(.001,float(np.max(np.abs(y)))*.01)
  candidates=[(abs(p-(left+right)/2),p,float(np.sqrt(np.mean(y[p-half:p+half]**2))))for p in positions]
  candidates=[v for v in candidates if v[2]<=threshold]
  if not candidates:raise ValueError('sentence boundary has no verified quiet cut')
  _,cut,rms=min(candidates);cuts.append(cut);boundaries.append({'before_word':i,'cut_sample':cut,'measured_gap':[left,right],'local_RMS':rms,'RMS_limit':threshold})
 cuts.append(len(y));start_word=0;rows=[]
 from scipy.signal import resample_poly
 for j,end_word in enumerate(ends):
  samples=y[cuts[j]:cuts[j+1]];wave16=resample_poly(samples,2,3).astype(np.float32)
  segments,_=evaluator.asr.transcribe(wave16,language='en',beam_size=5,temperature=0,condition_on_previous_text=False,vad_filter=False,word_timestamps=True)
  hyp=' '.join(s.text.strip()for s in segments);expected=tokens[start_word:end_word];wer=distance(expected,normalized(hyp))/max(1,len(expected));rows.append({'start_sample':cuts[j],'end_sample':cuts[j+1],'start_word':start_word,'end_word':end_word,'expected_tokens':expected,'ASR_text':hyp,'WER':wer,'exact':wer==0});start_word=end_word
 return {'schema':'mari-clause-asr/1.0','audio_sha256':sha(audio),'sample_rate':sr,'frames':len(y),'boundaries':boundaries,'clauses':rows,'all_samples_covered_once':cuts[0]==0 and cuts[-1]==len(y)and all(a<b for a,b in zip(cuts,cuts[1:])),'all_words_covered_once':start_word==len(tokens),'all_clauses_WER0':all(x['exact']for x in rows),'admitted':all(x['exact']for x in rows),'ASR_received_transcript_hint':False,'whole_utterance_result_replaced':False,'full_completion':False}
