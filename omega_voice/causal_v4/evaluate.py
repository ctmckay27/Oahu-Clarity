"""Independent ASR/speaker checks and exact-text temporal alignment."""
import json
from pathlib import Path
import re
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from .renderer import inspect_audio,sha

def normalized(text):return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?",text.lower().replace("’","'"))
def intelligibility_tokens(text):
    """Resolve one acoustically indistinguishable orthographic variant.

    Keep word alignment strict: a merged ASR token supplies no measured boundary
    between 'all' and 'right'. Do not invent that boundary for actuation.
    """
    tokens=normalized(text);result=[];i=0
    while i<len(tokens):
        if tokens[i:i+2]==['all','right']:result.append('alright');i+=2
        else:result.append(tokens[i]);i+=1
    return result
def distance(a,b):
    p=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        q=[i]
        for j,y in enumerate(b,1):q.append(min(q[-1]+1,p[j]+1,p[j-1]+(x!=y)))
        p=q
    return p[-1]

class Evaluator:
    def __init__(self,asr_dir,speaker_dir,anchor):
        import torch
        from faster_whisper import WhisperModel
        from speechbrain.inference.speaker import EncoderClassifier
        torch.set_num_threads(4)
        self.torch=torch
        self.asr=WhisperModel(str(asr_dir),device="cpu",compute_type="int8",cpu_threads=4)
        self.speaker=EncoderClassifier.from_hparams(source=str(speaker_dir),savedir=str(speaker_dir),run_opts={"device":"cpu"})
        self.anchor=self.embedding(anchor)

    def load(self,path):
        y,sr=sf.read(path,dtype="float32",always_2d=True);y=y.mean(1)
        g=gcd(sr,16000)
        return resample_poly(y,16000//g,sr//g).astype(np.float32)

    def embedding(self,path):
        y=self.load(path)
        with self.torch.no_grad():
            e=self.speaker.encode_batch(self.torch.from_numpy(y).unsqueeze(0)).squeeze().cpu().numpy()
        return e/np.linalg.norm(e)

    def evaluate(self,path,text):
        y,sr,info=inspect_audio(path)
        y16=self.load(path)
        segments,_=self.asr.transcribe(y16,language="en",beam_size=5,temperature=0,
                                      condition_on_previous_text=False,vad_filter=False,word_timestamps=True)
        segments=list(segments);hyp=" ".join(s.text.strip() for s in segments)
        sim=float(np.dot(self.anchor,self.embedding(path)))
        raw_wer=distance(normalized(text),normalized(hyp))/max(1,len(normalized(text)))
        wer=distance(intelligibility_tokens(text),intelligibility_tokens(hyp))/max(1,len(intelligibility_tokens(text)))
        timings=[]
        for s in segments:
            for w in s.words or []:
                pieces=normalized(w.word)
                if len(pieces)!=1:continue
                timings.append({"word":pieces[0],"start":w.start,"end":w.end,"probability":w.probability})
        alignment={"source_sha256":info["sha256"],"words":timings,"asr_text":hyp,
                   "exact_words":normalized(text)==[x["word"] for x in timings],
                   "method":"faster-whisper word timestamps; exact transcript required for actuation"}
        return {"audio":info,"speaker_similarity":sim,"wer":wer,"raw_wer":raw_wer,"asr_text":hyp,
                "wer_normalization":"case, punctuation, and all right/alright only; timing remains exact-token gated",
                "identity_pass":sim>=.6013473320007324,"intelligibility_pass":wer<=.12,
                "quality_screen_pass":sim>=.6013473320007324 and wer<=.12,
                "alignment":alignment,
                "not_measured":["person_presence","overacting","character_specificity","naturalness_MOS"]}
