"""Phonetic CTC time alignment; lexical verification remains a separate input."""
import json,pathlib,hashlib,importlib.metadata
import numpy as np,torch,torchaudio
from .alignment import AlignmentRejected
from ..causal_v4.evaluate import normalized
from ..causal_v4.renderer import sha

class PhonemeAligner:
 def __init__(self,directory,manifest):
  from phonemizer.backend.espeak.wrapper import EspeakWrapper
  from phonemizer.backend import EspeakBackend
  import espeakng_loader
  from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2ForCTC
  self.directory=pathlib.Path(directory);self.manifest=json.loads(pathlib.Path(manifest).read_text())
  if self.manifest['revision']!='ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4':raise ValueError('unselected phoneme model')
  for row in self.manifest['files']:
   if sha(self.directory/row['path'])!=row['sha256']:raise ValueError('phoneme model bytes changed')
  EspeakWrapper.set_library(espeakng_loader.get_library_path());EspeakWrapper.set_data_path(espeakng_loader.get_data_path());self.backend=EspeakBackend('en-us',with_stress=False);self.vocab=json.loads((self.directory/'vocab.json').read_text());self.labels={v:k for k,v in self.vocab.items()};self.extractor=Wav2Vec2FeatureExtractor.from_pretrained(self.directory);self.model=Wav2Vec2ForCTC.from_pretrained(self.directory).eval()
  self.provenance={'model':self.manifest,'implementation_sha256':sha(__file__),'espeak_library_sha256':sha(espeakng_loader.get_library_path()),'dependencies':{x:importlib.metadata.version(x)for x in ['torch','torchaudio','transformers','phonemizer-fork','espeakng-loader']},'lexical_verification_separate':True}

 def align(self,wave16,text,source_hash,transcript_verified):
  from phonemizer.separator import Separator
  if not transcript_verified:raise ValueError('phonetic alignment cannot verify lexical content')
  words=normalized(text);pronunciation=self.backend.phonemize([' '.join(words)],separator=Separator(phone=' ',word=' | ',syllable=''),strip=True)[0];groups=[x.strip().split()for x in pronunciation.split('|')]
  if len(groups)!=len(words):raise ValueError('phonemizer changed word segmentation')
  phones=[p for g in groups for p in g]
  if any(x not in self.vocab for x in phones):raise ValueError('unrepresented phonetic inventory')
  tokens=[self.vocab[x]for x in phones];inputs=self.extractor(wave16,sampling_rate=16000,return_tensors='pt')
  with torch.inference_mode():
   emission=self.model(**inputs).logits.log_softmax(-1);alignment,scores=torchaudio.functional.forced_align(emission,torch.tensor([tokens],dtype=torch.int32),blank=self.model.config.pad_token_id);spans=torchaudio.functional.merge_tokens(alignment[0],scores[0].exp(),blank=self.model.config.pad_token_id);greedy=torch.unique_consecutive(emission[0].argmax(-1)).tolist()
  if [x.token for x in spans]!=tokens:raise ValueError('phonetic alignment lost target units')
  step=len(wave16)/16000/emission.shape[1];result=[];cursor=0
  for word,group in zip(words,groups):
   ss=spans[cursor:cursor+len(group)];cursor+=len(group);duration=sum(x.end-x.start for x in ss);prob=sum(x.score*(x.end-x.start)for x in ss)/duration;result.append({'word':word,'start':float(ss[0].start*step),'end':float(ss[-1].end*step),'probability':float(prob),'phonemes':[{'phoneme':phone,'start':float(x.start*step),'end':float(x.end*step),'probability':float(x.score)}for phone,x in zip(group,ss)]})
  minimum=min(w['probability']for w in result);report={'source_sha256':source_hash,'words':result,'exact_words':True,'method':'dictionary phoneme CTC alignment of separately supplied transcript','model_revision':self.manifest['revision'],'minimum_word_probability':minimum,'threshold':.35,'admitted':minimum>=.35,'pronunciation':pronunciation,'unprompted_greedy_phones':[self.labels[i]for i in greedy if i!=self.model.config.pad_token_id],'transcript_verification_separate':True,'frame_period_s':step}
  if minimum<.35:raise AlignmentRejected(report)
  return report
