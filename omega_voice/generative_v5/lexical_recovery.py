"""A bounded, independently triggered repair for an observed omitted final stop.

The spoken contract is immutable. Correct initial speech is never respelled.
One established phonetic route is available; failure does not launch seed search.
"""
import re,pathlib
import torch
from .native import render
from .session import durable_json
from ..causal_v4.renderer import sha,UnresolvedRealization

POLICY='checked_final_stop_recovery_v1'
def words(text):return re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())
def proposal(required,observed):
 a=words(required);b=words(observed)
 if len(a)!=len(b):return None
 changed=[i for i,(x,y)in enumerate(zip(a,b))if x!=y]
 if len(changed)!=1 or (a[changed[0]],b[changed[0]])!=('checked','check'):return None
 matches=list(re.finditer(r"[a-z]+(?:'[a-z]+)?",required,re.I));m=matches[changed[0]]
 return {'word_index':changed[0],'required_word':'checked','observed_word':'check','CMU_target':['CH','EH1','K','T'],'renderer_spelling':'chekt','renderer_text':required[:m.start()]+'chekt'+required[m.end():]}

def greedy_text(aligner,wave16):
 with torch.inference_mode():emission,_=aligner.model(torch.as_tensor(wave16,dtype=torch.float32)[None])
 ids=torch.unique_consecutive(emission.argmax(-1)[0]).tolist()
 return ''.join(aligner.labels[i]for i in ids if i).replace('|',' ').strip()

def attempt(root,text,initial,quality,seed,reference,evaluator,aligner,target):
 record={'policy':POLICY,'required_spoken_text':text,'initial_sha256':sha(initial),'initial_quality':quality,'maximum_repairs':1,'same_seed':seed,'no_seed_search':True,'full_completion':False}
 candidate=proposal(text,quality['asr_text'])
 if candidate is None or not quality['identity_pass']:
  record.update(triggered=False,reason='No independently confirmed eligible lexical defect; original gates remain operative.');durable_json(pathlib.Path(target)/'lexical_recovery.json',record);return initial,quality
 heard=greedy_text(aligner,evaluator.load(initial));record['independent_initial_CTC']=heard
 if proposal(text,heard)!=candidate:
  record.update(triggered=False,reason='Independent recognizers do not agree on the eligible omission.');durable_json(pathlib.Path(target)/'lexical_recovery.json',record);return initial,quality
 record.update(triggered=True,proposal=candidate);durable_json(pathlib.Path(target)/'lexical_recovery.json',record)
 audio=pathlib.Path(target)/'carrier_recovered.wav';native=render(root,candidate['renderer_text'],audio,seed=seed,reference=reference);q=evaluator.evaluate(audio,text);ctc=greedy_text(aligner,evaluator.load(audio));ok=q['quality_screen_pass']and q['wer']==0 and words(ctc)==words(text)
 record.update(native_receipt=native,recovered_quality=q,independent_recovered_CTC=ctc,repair_admitted=bool(ok),original_preserved=True);durable_json(pathlib.Path(target)/'lexical_recovery.json',record)
 if not ok:raise UnresolvedRealization('single bounded lexical recovery failed; state not committed')
 return audio,q
