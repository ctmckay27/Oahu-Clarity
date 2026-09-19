"""Known-transcript CTC alignment, separate from intelligibility verification."""
import json,pathlib
import torch,torchaudio
from ..causal_v4.evaluate import normalized
from ..causal_v4.renderer import sha

class AlignmentRejected(ValueError):
 def __init__(self,report):
  self.report=report
  super().__init__('insufficient forced-alignment confidence: '+str(report['minimum_word_probability']))

class ForcedAligner:
 def __init__(self,model_directory,receipt):
  self.directory=pathlib.Path(model_directory);self.receipt=json.loads(pathlib.Path(receipt).read_text())
  bundles={'torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H':torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H,
           'torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H':torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H}
  if self.receipt['model'] not in bundles:raise ValueError('unselected alignment model')
  self.bundle=bundles[self.receipt['model']]
  if sha(self.directory/self.bundle._path)!=self.receipt['sha256']:raise ValueError('alignment model provenance mismatch')
  self.model=self.bundle.get_model(dl_kwargs={'model_dir':str(self.directory),'progress':False}).eval()
  self.labels=self.bundle.get_labels();self.dictionary={c:i for i,c in enumerate(self.labels)}

 def align(self,wave16,text,source_hash,transcript_verified):
  if not transcript_verified:raise ValueError('forced alignment cannot replace independent transcript verification')
  ws=normalized(text);sentence='|'.join(w.upper() for w in ws)
  if not ws or any(c not in self.dictionary for c in sentence):raise ValueError('spoken lexical normalization required before alignment')
  tokens=[self.dictionary[c] for c in sentence]
  with torch.inference_mode():
   emission,_=self.model(torch.as_tensor(wave16,dtype=torch.float32)[None]);emission=emission.log_softmax(-1)
   alignment,logp=torchaudio.functional.forced_align(emission,torch.tensor([tokens],dtype=torch.int32),blank=0)
   spans=torchaudio.functional.merge_tokens(alignment[0],logp[0].exp(),blank=0)
  if [s.token for s in spans]!=tokens:raise RuntimeError('alignment lost transcript tokens')
  step=len(wave16)/(16000*emission.shape[1]);results=[];cursor=0
  for w in ws:
   selected=spans[cursor:cursor+len(w)];cursor+=len(w)+1
   duration=sum(s.end-s.start for s in selected);score=sum(s.score*(s.end-s.start) for s in selected)/duration
   results.append({'word':w,'start':float(selected[0].start*step),'end':float(selected[-1].end*step),'probability':float(score),
    'characters':[{'letter':self.labels[s.token],'start':float(s.start*step),'end':float(s.end*step),'probability':float(s.score)} for s in selected]})
  minimum=min(x['probability'] for x in results)
  report={'source_sha256':source_hash,'words':results,'exact_words':True,'method':'CTC forced alignment of independently verified transcript',
   'transcript_verification_separate':True,'model_sha256':self.receipt['sha256'],'frame_period_s':step,
   'minimum_word_probability':minimum,'admitted':minimum>=.35,'threshold':.35}
  if minimum<.35:raise AlignmentRejected(report)
  return report
