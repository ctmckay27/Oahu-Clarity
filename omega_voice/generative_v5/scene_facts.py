"""Scene compiler through independently queried semantic facts and causal time.

Free event construction failed semantic qualification. Here the language model
answers finite entailment questions; deterministic rules build typed events.
Future scene observations are withheld until their explicit temporal boundary.
Numeric strengths are declared conventions, not inferred physical measurements.
"""
import argparse,json,re,pathlib
import torch
from ..causal_v4.runtime import compile_scene,digest,THOUGHTS

FACTS=[
 ('fear','Mari is internally frightened'),('anger','Mari is angry'),
 ('amusement','Mari is amused'),('tenderness','Mari feels tenderness toward the listener'),
 ('curiosity','Mari is curious'),('shame','Mari feels embarrassed or ashamed'),
 ('relief','Mari feels relief'),
 ('conceal_fear','Mari is trying to keep the listener from noticing her fear'),
 ('conceal_anger','Mari is trying to keep the listener from noticing her anger'),
 ('conceal_amusement','Mari is trying to keep the listener from noticing her amusement'),
 ('trust','Mari trusts the listener'),('familiarity','Mari and the listener know one another well'),
 ('suspicion','Mari distrusts the listener'),('concern','Mari is concerned for the listener'),
 ('irritation','Mari is irritated with the listener'),
 ('fatigue','Mari is physically tired'),('exertion','Mari has just physically exerted herself'),
 ('low_breath','Mari is short of breath'),
 ('searching','Mari is searching her memory or searching for a word'),
 ('deciding','Mari is making a decision now'),('reconsidering','Mari is reconsidering a previous judgment'),
 ('realizing','Mari obtains new knowledge from an observation now'),
 ('correcting','Mari is correcting something she previously said'),
 ('withholding','Mari deliberately chooses not to reveal information'),
 ('reassure','Mari is trying to reassure the listener'),('clarify','Mari is trying to clarify a misunderstanding'),
 ('protect','Mari is trying to protect the listener from danger'),('challenge','Mari is challenging the listener\'s claim'),
 ('set_boundary','Mari is setting a boundary with the listener'),('tease','Mari is teasing the listener'),
 ('admit','Mari is admitting something to the listener'),
 ('urgency','Mari needs to act or communicate urgently'),
]

def time_strata(scene):
 marks=list(re.finditer(r'\b(?:at\s+)?(?:spoken\s+)?word(?:\s+index)?\s+(\d+)\s*[,;:]?',scene,re.I))
 out=[];start=0;at=0
 for m in marks:
  if scene[start:m.start()].strip():out.append((at,scene[start:m.start()].strip()))
  at=int(m.group(1));start=m.end()
 if scene[start:].strip():out.append((at,scene[start:].strip()))
 if any(a>b for (a,_),(b,_) in zip(out,out[1:])):raise ValueError('nonmonotonic scene chronology needs explicit resolution')
 return out

class FactCompiler:
 def __init__(self,model,tokenizer):
  self.model=model;self.tokenizer=tokenizer;tokenizer.padding_side='left'
  if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.eos_token_id

 def query(self,scene,batch_size=4):
  tok=self.tokenizer;rows=[]
  for start in range(0,len(FACTS),batch_size):
   subset=FACTS[start:start+batch_size];prompts=[]
   for key,claim in subset:
    prompt='Does the scene establish the following claim about Mari NOW? Answer yes only if supported by the scene. If the claim is absent, contradicted, or concerns another person, answer no. Answer only yes or no.\nSCENE: '+scene+'\nCLAIM: '+claim+'.'
    prompts.append(tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True))
   inputs=tok(prompts,return_tensors='pt',padding=True)
   with torch.inference_mode():out=self.model.generate(**inputs,max_new_tokens=4,do_sample=False)
   answers=tok.batch_decode(out[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
   for (key,claim),answer in zip(subset,answers):
    raw=answer.strip();match=re.match(r'^(yes|no)(?:[.!,\s]|$)',raw,re.I)
    if not match:raise ValueError('unresolved semantic fact '+key+': '+raw)
    rows.append({'key':key,'claim':claim,'entailed':match[1].lower()=='yes','raw_answer':raw})
  return rows

 def compile(self,scene,text,prior=None):
  events=[];evidence=[]
  for at,source in time_strata(scene):
   facts=self.query(source);found={f['key'] for f in facts if f['entailed']};evidence.append({'at_word':at,'source':source,'judgments':facts})
   def emit(kind,**fields):
    event={'id':'fact-'+digest([source,at,kind,fields])[:16],'at_word':at,'kind':kind,
           'source':{'kind':'scene fact entailment','text':source,'scene_sha256':digest(scene)},**fields}
    events.append(event)
   values={}
   for emotion in ['fear','anger','amusement','tenderness','curiosity','shame','relief']:
    if emotion in found:
     if 'conceal_'+emotion in found:emit('mask',dimension=emotion,internal=.75,display=.1,effort=.8)
     else:values['emotional_state.'+emotion]=.75
   for key in ['trust','familiarity','suspicion','concern','irritation']:
    if key in found:values['relationship.'+key]=.8
   if 'fatigue' in found:values['body_state.fatigue']=.8
   if 'exertion' in found:values['body_state.exertion']=.75
   if 'low_breath' in found:values['body_state.breath_reserve']=.3
   if 'urgency' in found:values['interaction_state.urgency']=.75
   if values:emit('set',values=values)
   for thought in ['searching','deciding','reconsidering','correcting','withholding','realizing']:
    if thought in found:emit('thought',mode=thought)
   if 'realizing' in found:
    emit('knowledge',status='known',proposition=source,confidence=.95)
   for tactic in ['reassure','clarify','protect','challenge','set_boundary','tease','admit']:
    if tactic in found:emit('action',tactic=tactic,target='current listener')
  resolved={'events':events};plan=compile_scene(text,resolved,prior)
  return {'scene':resolved,'plan':plan,'fact_evidence':evidence,
          'semantic_status':'bounded fact compiler; qualification required; unrecognized dimensions are not claimed supported'}

class EntailmentCompiler(FactCompiler):
 """Three-way trained inference distinguishes unsupported from contradicted.

 A fixed .90 admission threshold is declared before qualification. Borderline
 facts are recorded as unresolved, not converted into confident state changes.
 """
 def __init__(self,model,tokenizer):
  self.model=model;self.tokenizer=tokenizer
  # DeBERTa pools position zero. Autoregressive left-padding would pool PAD
  # instead of CLS on shorter examples and silently corrupt the judgment.
  tokenizer.padding_side='right'
 def query(self,scene,batch_size=8):
  premise=re.sub(r'\b[Ss]he\b','Mari',scene)
  rows=[]
  mapping={str(v).lower():int(k) for k,v in self.model.config.id2label.items()}
  if set(mapping)!={'entailment','neutral','contradiction'}:raise ValueError('unrecognized NLI label semantics')
  for start in range(0,len(FACTS),batch_size):
   subset=FACTS[start:start+batch_size]
   inputs=self.tokenizer([premise]*len(subset),[claim+'.' for _,claim in subset],return_tensors='pt',padding=True,truncation=True,max_length=512)
   with torch.inference_mode():scores=self.model(**inputs).logits.softmax(-1).cpu().tolist()
   for (key,claim),score in zip(subset,scores):
    p={k:float(score[i]) for k,i in mapping.items()}
    rows.append({'key':key,'claim':claim,'entailed':p['entailment']>=.90,'probabilities':p,
                 'status':'entailed' if p['entailment']>=.90 else ('unresolved' if p['entailment']>=.5 else 'not_entailed')})
  return rows

def main():
 from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForSequenceClassification
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--corpus',required=True);ap.add_argument('--output',required=True);ap.add_argument('--nli',action='store_true');a=ap.parse_args()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 if a.nli:
  model=AutoModelForSequenceClassification.from_pretrained(a.model,dtype=torch.float32,device_map='cpu').eval()
  compiler=EntailmentCompiler(model,AutoTokenizer.from_pretrained(a.model))
 else:
  model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cpu',attn_implementation='sdpa').eval()
  compiler=FactCompiler(model,AutoTokenizer.from_pretrained(a.model))
 rows=[]
 for case in json.loads(pathlib.Path(a.corpus).read_text()):
  row={'case':case}
  try:row['result']=compiler.compile(case['scene'],case['spoken'])
  except Exception as e:row['error']=str(e)
  rows.append(row);pathlib.Path(a.output).write_text(json.dumps(rows,indent=2)+'\n');print(case['id'],row.get('error',row.get('result',{}).get('scene')),flush=True)
if __name__=='__main__':main()
