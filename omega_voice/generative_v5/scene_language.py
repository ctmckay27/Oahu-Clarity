"""Validated local language-to-event interface, separate from speech realization.

Source quotations are required for every change. The compiler produces only
typed events; nothing here is forwarded as a renderer instruction. Semantic
qualification is independent of structural validation.
"""
import copy,json,re
from ..causal_v4.runtime import compile_scene,digest,THOUGHTS,TACTICS,REL_KEYS,EMOTIONS,NUMERIC_PATHS,words

CONTRACT='''Translate the scene into JSON only, with one key "events" containing a list.
Every event has "at_word" (zero-based spoken-word index), "kind", and "source_quote"
(an exact substring of the scene supporting that event). Changes at the beginning
use at_word 0. Do not infer facts from the spoken words alone. Do not invent events.
Allowed kinds and fields:
- thought: mode = known, remembering, deciding, discovering, reconsidering,
  correcting, searching, suppressing, judging, realizing, withholding, observing.
- knowledge: status = known, beliefs, suspicions; proposition = string;
  confidence = 0..1.
- action: tactic = share, clarify, reassure, protect, challenge, set_boundary,
  invite, tease, correct, confront, persuade, admit, withhold; target = string.
- mask: dimension = fear, anger, amusement, tenderness, curiosity, shame, relief;
  internal = 0..1, display = 0..1, effort = 0..1.
- set: values = object of approved numeric state paths to 0..1. Paths:
  body_state.fatigue, body_state.exertion, body_state.breath_reserve,
  emotional_state.fear, emotional_state.anger, emotional_state.amusement,
  emotional_state.tenderness, emotional_state.curiosity, emotional_state.shame,
  emotional_state.relief, knowledge_state.certainty, listener_model.monitoring,
  listener_model.resistance, relationship.trust, relationship.familiarity,
  relationship.irritation, relationship.concern, relationship.authority,
  relationship.vulnerability, relationship.playfulness, relationship.distance,
  relationship.suspicion, relationship.expected_knowledge,
  interaction_state.urgency, subtext.disclosure, subtext.conflict.
- feedback: feedback = resistance, understood, misunderstood.
- interrupt or resume: no further fields.

Preserve internal state versus intended display: trying not to sound scared is
a mask event with elevated internal fear, low display and high effort. It is not
a request for a "scared voice". Remembering/searching precede a later realization;
do not move the later knowledge into the beginning. Relationship changes remain
separate dimensions. Intensity may use .25 low, .5 medium, .8 high; these are
bounded engineering conventions, not physiological measurements. If no supported
event follows from the scene, return an empty events list. No style prose, voice
description, speaker changes, or markup in the spoken text.

The scene's sentence number is NOT a spoken-word index. Every initial fact
happens at word 0, even when it appears in a later sentence of the scene.
Fear/fright is an emotional state, not a discovery or realization. A realization
requires newly obtained knowledge. An intention to hide a feeling requires a mask.

Example scene: Mari finds the mistake funny, but she wants to keep her amusement
out of her voice.
Example events: {"events":[{"kind":"mask","at_word":0,"source_quote":"Mari finds the mistake funny, but she wants to keep her amusement out of her voice.","dimension":"amusement","internal":0.7,"display":0.1,"effort":0.8}]}
Example scene: Mari is weary after working late. She does not feel angry.
Example events: {"events":[{"kind":"set","at_word":0,"source_quote":"Mari is weary after working late.","values":{"body_state.fatigue":0.8}}]}
'''

FIELDS={'thought':{'mode'},'knowledge':{'status','proposition','confidence'},
        'action':{'tactic','target'},'mask':{'dimension','internal','display','effort'},
        'set':{'values'},'feedback':{'feedback'},'interrupt':set(),'resume':set()}

def explicit_times(scene_text,spoken_text):
 # This interface currently supports initial scene facts and explicitly indexed
 # later observations. Unresolved free-form temporal anchors require expansion;
 # a model must not fabricate one event per word or scene sentence.
 indices={0,*[int(n) for n in re.findall(r'(?:spoken\s+)?word(?:\s+index)?\s+(\d+)',scene_text,re.I)]}
 if any(i>len(words(spoken_text)) for i in indices):raise ValueError('scene time outside supplied speech')
 return sorted(indices)

def event_grammar(scene_text,spoken_text):
 """Finite values and verbatim source spans are enforced during decoding."""
 spans=[scene_text]+[s.strip() for s in re.split(r'(?<=[.!?])\s+',scene_text) if len(s.strip())>=4]
 scalar={'type':'number','enum':[0,.1,.15,.2,.25,.3,.4,.5,.6,.7,.75,.8,.85,.9,.95,1]}
 string={'type':'string','maxLength':120}
 additions={
  'thought':{'mode':{'type':'string','enum':sorted(THOUGHTS)}},
  'knowledge':{'status':{'type':'string','enum':['known','beliefs','suspicions']},'proposition':string,'confidence':scalar},
  'action':{'tactic':{'type':'string','enum':sorted(TACTICS)},'target':string},
  'mask':{'dimension':{'type':'string','enum':sorted(EMOTIONS)},'internal':scalar,'display':scalar,'effort':scalar},
  'set':{'values':{'type':'object','properties':{k:scalar for k in NUMERIC_PATHS if NUMERIC_PATHS[k]==(0,1)},'additionalProperties':False,'minProperties':1}},
  'feedback':{'feedback':{'type':'string','enum':['resistance','understood','misunderstood']}},
  'interrupt':{},'resume':{},
 }
 options=[]
 for kind,extra in additions.items():
  props={'kind':{'type':'string','enum':[kind]},'at_word':{'type':'integer','enum':explicit_times(scene_text,spoken_text)},'source_quote':{'type':'string','enum':list(dict.fromkeys(spans))},**extra}
  options.append({'type':'object','properties':props,'required':list(props),'additionalProperties':False})
 return {'type':'object','properties':{'events':{'type':'array','items':{'anyOf':options},'maxItems':8}},'required':['events'],'additionalProperties':False}

class SceneCompilationError(ValueError):
 def __init__(self,message,raw):super().__init__(message);self.raw=raw

def parse_events(raw,scene_text,spoken_text,prior=None):
 raw=raw.strip()
 if raw.startswith('```'):raw=re.sub(r'^```(?:json)?\s*|\s*```$','',raw)
 data=json.loads(raw)
 if set(data)!={'events'} or not isinstance(data['events'],list) or len(data['events'])>40:raise ValueError('invalid compiler result')
 events=[];witnesses=[]
 for i,e in enumerate(data['events']):
  k=e.get('kind');quote=e.get('source_quote')
  if k not in FIELDS or set(e)!=FIELDS[k]|{'at_word','kind','source_quote'}:raise ValueError('uncontrolled event fields')
  if not isinstance(quote,str) or not quote.strip() or quote not in scene_text:raise ValueError('ungrounded scene quote')
  if len(quote)<4:raise ValueError('insufficient source span')
  if e['at_word'] not in explicit_times(scene_text,spoken_text):raise ValueError('unanchored scene event time')
  event={n:copy.deepcopy(v) for n,v in e.items() if n!='source_quote'}
  event.update(id='scene-'+digest([scene_text,spoken_text,i,e])[:16],source={'kind':'natural-language scene','text':quote,'scene_sha256':digest(scene_text)})
  events.append(event);witnesses.append({'event':event['id'],'source_quote':quote})
 scene={'events':events};plan=compile_scene(spoken_text,scene,prior)
 return {'scene':scene,'plan':plan,'witnesses':witnesses,'semantic_status':'requires independently qualified scene interpretation; source span is provenance, not proof of entailment'}

def compile_with_model(model,processor,scene_text,spoken_text,prior=None,text_only=False,progress_path=None):
 import torch
 from lmformatenforcer import JsonSchemaParser,TokenEnforcer,TokenEnforcerTokenizerData
 prompt=CONTRACT+'\nSCENE:\n'+scene_text+'\nSPOKEN TEXT:\n'+spoken_text
 conversation=[{'role':'user','content':prompt if text_only else [{'type':'text','text':prompt}]}]
 text=processor.apply_chat_template(conversation,tokenize=False,add_generation_prompt=True)
 inputs=processor(text=text,return_tensors='pt').to(model.device)
 if not hasattr(processor,'mari_grammar_tokens'):
  # Public core API avoids lm-format-enforcer 0.11.3's obsolete Transformers
  # tokenization_utils import. Preserve leading spaces and partial UTF-8 tokens.
  tok=processor if text_only else processor.tokenizer;prefix=tok.encode('0',add_special_tokens=False)[-1]
  special=set(tok.all_special_ids);entries=[]
  for token in range(len(tok)):
   if token in special:continue
   raw=tok.decode([token]);context=tok.decode([prefix,token])[1:]
   entries.append((token,context,len(context)>len(raw)))
  processor.mari_grammar_tokens=TokenEnforcerTokenizerData(entries,lambda ids:tok.decode(ids).rstrip('\ufffd'),tok.eos_token_id,False,len(tok))
 enforcer=TokenEnforcer(processor.mari_grammar_tokens,JsonSchemaParser(event_grammar(scene_text,spoken_text)))
 def allowed(batch_id,sent):return enforcer.get_allowed_tokens(sent.tolist()).allowed_tokens
 class Progress:
  def __init__(self):self.first=True;self.ids=[]
  def put(self,value):
   if self.first:self.first=False;return
   self.ids.extend(value.reshape(-1).tolist())
   if progress_path and len(self.ids)%20==0:self.save()
  def save(self):
   import pathlib
   pathlib.Path(progress_path).write_text(json.dumps({'status':'incomplete generation; not admitted','tokens':len(self.ids),'raw':processor.decode(self.ids) if text_only else processor.tokenizer.decode(self.ids)},indent=2)+'\n')
  def end(self):
   if progress_path:self.save()
 with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=650,do_sample=False,prefix_allowed_tokens_fn=allowed,streamer=Progress())
 raw=processor.batch_decode(out[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
 try:result=parse_events(raw,scene_text,spoken_text,prior)
 except Exception as e:raise SceneCompilationError(str(e),raw) from e
 result['raw_model_output']=raw;result['decoding']='typed-event JSON grammar; verbatim source span enumeration';return result

def main():
 import argparse,pathlib,time,torch
 from transformers import AudioFlamingo3ForConditionalGeneration,AutoProcessor,AutoModelForCausalLM,AutoTokenizer
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--corpus',required=True);ap.add_argument('--output',required=True);ap.add_argument('--kind',choices=['text','audio'],default='audio');a=ap.parse_args()
 torch.set_num_threads(5);torch.set_num_interop_threads(1)
 cls=AutoModelForCausalLM if a.kind=='text' else AudioFlamingo3ForConditionalGeneration
 model=cls.from_pretrained(a.model,dtype=torch.bfloat16,device_map='cpu',low_cpu_mem_usage=True,attn_implementation='sdpa').eval()
 processor=(AutoTokenizer if a.kind=='text' else AutoProcessor).from_pretrained(a.model);rows=[]
 print('SCENE_MODEL_LOADED',a.kind,flush=True)
 for case in json.loads(pathlib.Path(a.corpus).read_text()):
  start=time.monotonic();row={'case':case}
  print('SCENE_CASE',case['id'],flush=True)
  try:row['result']=compile_with_model(model,processor,case['scene'],case['spoken'],text_only=a.kind=='text',progress_path=a.output+'.in_progress.json')
  except Exception as e:
   row['error']=str(e)
   if hasattr(e,'raw'):row['raw_model_output']=e.raw
  row['elapsed_s']=time.monotonic()-start;rows.append(row)
  pathlib.Path(a.output).write_text(json.dumps(rows,indent=2)+'\n');print(case['id'],row.get('error',row['result']['scene'] if 'result' in row else 'failed'),flush=True)
if __name__=='__main__':main()
