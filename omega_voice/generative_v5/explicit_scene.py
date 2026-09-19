"""Compositional explicit scene facts, with subject, negation and time scope.

This is a bounded English interface, not unrestricted pragmatic inference.
Dependency parsing proposes structure; declared predicate rules license events.
Unresolved statements about Mari block admission rather than inventing motives.
No linguistic model, source sentence or acting note is sent to the renderer.
"""
import argparse,json,pathlib,re,hashlib,importlib.metadata
from ..causal_v4.runtime import compile_scene,digest

EMOTION={'afraid':'fear','frightened':'fear','scared':'fear','fearful':'fear','terrified':'fear',
 'angry':'anger','furious':'anger','amused':'amusement','tender':'tenderness','curious':'curiosity',
 'ashamed':'shame','embarrassed':'shame','relieved':'relief'}
THOUGHT={'remember':'remembering','decide':'deciding','reconsider':'reconsidering','realize':'realizing',
 'correct':'correcting','withhold':'withholding','judge':'judging','observe':'observing'}
ACTION={'reassure':'reassure','clarify':'clarify','protect':'protect','challenge':'challenge',
 'tease':'tease','admit':'admit','persuade':'persuade','invite':'invite','confront':'confront'}
REPORT={'say','claim','pretend','imagine','deny','suppose','wish'}

def strata(scene):
 marks=list(re.finditer(r'\b(?:at\s+)?(?:spoken\s+)?word(?:\s+index)?\s+(\d+)\s*[,;:]?',scene,re.I))
 out=[];start=0;at=0
 for m in marks:
  if scene[start:m.start()].strip():out.append((at,scene[start:m.start()].strip()))
  at=int(m.group(1));start=m.end()
 if scene[start:].strip():out.append((at,scene[start:].strip()))
 if any(a>b for (a,_),(b,_) in zip(out,out[1:])):raise ValueError('nonmonotonic scene chronology')
 return out

def governor_subject(token):
 seen=set();node=token
 while node.i not in seen:
  seen.add(node.i)
  subjects=[c for c in node.children if c.dep_ in {'nsubj','nsubjpass'}]
  if subjects:return subjects[0]
  if node.head==node:break
  node=node.head
 return None

def negative(token):
 nodes=[token]
 if token.dep_ in {'acomp','attr','xcomp'}:nodes.append(token.head)
 return any(c.dep_=='neg' or c.lower_ in {'never','neither'} for node in nodes for c in node.children)

def nonasserted(token):
 for node in [token,*token.ancestors]:
  if any(c.lower_ in {'if','unless','whether'} for c in node.children):return True
  if node!=token and node.lemma_.lower() in REPORT:return True
 return False

def concealment(token):
 """Resolve the concealed dimension inside this predicate's own clause."""
 lemma=token.lemma_.lower()
 if lemma not in {'hide','conceal','suppress','keep','try'} or negative(token):return None
 scoped=[]
 def visit(node):
  scoped.append(node)
  for child in node.children:
   if child.dep_ in {'advcl','conj','parataxis'}:continue
   visit(child)
 visit(token)
 words={x.lower_ for x in scoped};lemmas={x.lemma_.lower() for x in scoped}
 if lemma=='try' and not ('sound' in lemmas and 'not' in words):return None
 if lemma=='keep' and not (('from' in words and bool(lemmas&{'hear','notice','see'})) or {'out','of','voice'}<=words):return None
 dimensions={EMOTION[x] for x in words if x in EMOTION}|(words&{'fear','anger','amusement'})
 if len(dimensions)!=1:return None
 return next(iter(dimensions)),{x.i for x in scoped}

class ExplicitSceneCompiler:
 def __init__(self,nlp,policy='bounded_thought_recovery_v1'):
  self.nlp=nlp
  if policy not in {'bounded_thought_recovery_v1','epistemic_focus_v2'}:raise ValueError('unknown scene temporal policy')
  self.temporal_policy=policy
  if nlp.meta.get('version')!='3.8.0' or nlp.meta.get('lang')!='en' or importlib.metadata.version('spacy')!='3.8.7':raise ValueError('unselected dependency parser version')
  self.provenance={'spacy':'3.8.7','model':'en_core_web_sm','version':'3.8.0',
   'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
   'pipeline_sha256':hashlib.sha256(nlp.to_bytes()).hexdigest(),'temporal_policy':policy}
 def compile(self,scene,text,prior=None,context=None):
  events=[];audit=[];unresolved=[];principal='mari'
  context=context or {}
  if set(context)-{'assertion'}:raise ValueError('unknown conversational context field')
  assertion=context.get('assertion')
  if assertion:
   if self.temporal_policy!='epistemic_focus_v2':raise ValueError('claim context requires epistemic_focus_v2')
   if set(assertion)-{'proposition','mode','confidence','source'}:raise ValueError('unknown assertion context field')
   if not isinstance(assertion.get('source'),dict) or not assertion['source'].get('text'):raise ValueError('assertion context needs source')
   events.append(dict(assertion,id='claim-'+digest(assertion)[:16],at_word=0,kind='assertion'))
  for at,source in strata(scene):
   doc=self.nlp(source);facts=[]
   for sent in doc.sents:
    sentence=sent.text;tokens=list(sent);used=set();mari_roots=set();before_principal=principal
    def is_mari(token):
     subject=governor_subject(token)
     if subject is None:return False
     if subject.lower_=='mari':return True
     if subject.lower_=='she':return before_principal=='mari'
     return False
    def mark(token):
     used.add(token.i)
     for parent in token.ancestors:
      if parent in sent:used.add(parent.i)
    def add(token,key,value=True):
     facts.append({'key':key,'value':value,'source':sentence,'predicate':token.text,'predicate_offset':token.idx});mark(token)
    # Explicit concealment is a relation between internal and displayed state.
    # It does not license inferred reassurance, protection or boundary tactics.
    concealed={t.i:concealment(t) for t in tokens if is_mari(t) and not nonasserted(t)}
    concealed={k:v for k,v in concealed.items() if v}
    for token in tokens:
     if token.pos_ not in {'VERB','ADJ','AUX'}:continue
     if not is_mari(token):continue
     if token.dep_ in {'ROOT','conj','advcl'}:mari_roots.add(token.i)
     if nonasserted(token):
      unresolved.append({'at_word':at,'source':sentence,'reason':'conditional or reported Mari state requires resolved scene reality'});mark(token);continue
     lemma=token.lemma_.lower();neg=negative(token);subtree=' '.join(x.text for x in token.subtree).lower()
     if token.i in concealed:
      add(token,'conceal_'+concealed[token.i][0]);continue
     emotion=EMOTION.get(token.lower_)
     if emotion and token.dep_ in {'ROOT','acomp','attr','conj','xcomp'}:
      if any(emotion==dimension and token.i in scope for dimension,scope in concealed.values()):mark(token);continue
      add(token,'emotion.'+emotion,0. if neg else .75);continue
     if lemma in {'frighten','scare'} and any(c.dep_=='nsubjpass' for c in token.children):
      add(token,'emotion.fear',0. if neg else .75);continue
     if neg:
      if lemma=='remember':
       objects=[c for c in token.children if c.dep_=='dobj']
       if objects:add(token,'unknown',' '.join(x.text for c in objects for x in c.subtree))
      # Negated actions are not performed actions. Keep the explicit audit.
      if lemma in ACTION or lemma in THOUGHT or lemma in {'trust','search','run','know'}:mark(token)
      continue
     if lemma=='search' or (lemma=='look' and 'for' in subtree and any(x in subtree for x in ['word','name'])):add(token,'thought','remembering' if 'memory' in subtree else 'searching')
     elif lemma in THOUGHT:
      if lemma!='correct' or any(x in subtree for x in ['herself','her earlier','her previous']):add(token,'thought',THOUGHT[lemma])
     elif lemma in ACTION:add(token,'action',ACTION[lemma])
     elif lemma in {'trust','distrust'}:add(token,'relationship.'+('trust' if lemma=='trust' else 'suspicion'),.8)
     elif lemma=='know':
      if any(c.dep_=='dobj' and c.lower_ in {'listener','them','you'} for c in token.children):add(token,'relationship.familiarity',.8 if 'year' in subtree else .6)
      else:
       complements=[c for c in token.children if c.dep_ in {'ccomp','dobj','xcomp'}]
       if complements:
        prop=' '.join(x.text for c in complements for x in c.subtree)
        add(token,'knowledge',prop)
     elif lemma in {'spot','notice','see'}:add(token,'observation',sentence)
     elif lemma in {'run','sprint','climb'}:add(token,'body_state.exertion',.75)
     elif token.lower_ in {'exhausted','weary','tired','fatigued'}:add(token,'body_state.fatigue',.8)
     elif token.lower_=='short' and 'breath' in subtree:add(token,'body_state.breath_reserve',.3)
     elif lemma=='monitor':add(token,'listener_model.monitoring',.8)
     elif lemma=='feel':
      for child in token.children:
       if child.dep_=='dobj' and child.lower_ in {'fear','anger','amusement','tenderness','curiosity','shame','relief'}:
        add(token,'emotion.'+child.lower_,0. if any(c.lower_=='no' for c in child.children) else .75)
     elif token.lower_=='irritated':add(token,'relationship.irritation',.8)
     elif token.lower_=='urgent':add(token,'interaction_state.urgency',.75)
     elif token.lower_=='familiar':add(token,'relationship.familiarity',.8)
    for root in mari_roots-used:
     token=doc[root]
     if token.lemma_.lower() not in {'be','have'} or any(c.dep_ in {'acomp','attr','xcomp'} for c in token.children):
      unresolved.append({'at_word':at,'source':sentence,'reason':'unlicensed Mari predicate '+token.text})
    root_subj=governor_subject(sent.root)
    if root_subj is not None:
     if root_subj.lower_=='mari':principal='mari'
     elif root_subj.lower_ not in {'she','it','there','nothing'}:principal='other'
    audit.append({'at_word':at,'source':sentence,'facts':[x for x in facts if x['source']==sentence]})
   def emit(kind,source,**fields):
    event={'id':'explicit-'+digest([at,kind,source,fields])[:16],'at_word':at,'kind':kind,
           'source':{'kind':'explicit scene predicate','text':source,'scene_sha256':digest(scene)},**fields}
    if event not in events:events.append(event)
   masks={f['key'][8:] for f in facts if f['key'].startswith('conceal_')}
   observed=any(f['key']=='observation' for f in facts)
   for fact in facts:
    key,value,quote=fact['key'],fact['value'],fact['source']
    if key.startswith('conceal_'):emit('mask',quote,dimension=key[8:],internal=.75,display=.1,effort=.8)
    elif key.startswith('emotion.'):
     if key[8:] not in masks:emit('set',quote,values={'emotional_state.'+key[8:]:value})
    elif '.' in key:emit('set',quote,values={key:value})
    elif key=='thought':emit('thought',quote,mode=value)
    elif key=='action':emit('action',quote,tactic=value,target='current listener')
    elif key=='knowledge':
     # A definite answer can refer to the explicitly supplied current claim.
     # Other world facts retain their own proposition and cannot change it.
     proposition=assertion['proposition'] if assertion and value.strip().lower()=='the answer' else value
     emit('knowledge',quote,status='known',confidence=.95,proposition=proposition)
     if observed:emit('thought',quote,mode='realizing')
    elif key=='unknown':emit('knowledge',quote,status='unresolved',confidence=0.,proposition=value)
   # Observation without an explicit epistemic result licenses observation,
   # not an invented proposition or a claim of certainty.
   if observed and not any(f['key']=='knowledge' for f in facts):
    emit('thought',next(f['source'] for f in facts if f['key']=='observation'),mode='observing')
  resolved={'events':events}
  return {'scene':resolved,'plan':compile_scene(text,resolved,prior,policy=self.temporal_policy),'conversational_context':context,
   'audit':audit,'unresolved':unresolved,'admitted':not unresolved,
   'provenance':self.provenance,
   'scope':'explicit licensed English predicates; unsupported material statements block admission; no pragmatic objective inference'}

 def runtime_scene(self,scene,text,prior=None):
  result=self.compile(scene,text,prior)
  if not result['admitted']:raise ValueError('unresolved scene reality: '+json.dumps(result['unresolved']))
  return result['scene'],result['audit']

def main():
 import spacy
 ap=argparse.ArgumentParser();ap.add_argument('corpus');ap.add_argument('output');a=ap.parse_args()
 compiler=ExplicitSceneCompiler(spacy.load('en_core_web_sm'));rows=[]
 for case in json.loads(pathlib.Path(a.corpus).read_text()):
  try:row={'case':case,'result':compiler.compile(case['scene'],case['spoken'])}
  except Exception as e:row={'case':case,'error':str(e)}
  rows.append(row);pathlib.Path(a.output).write_text(json.dumps(rows,indent=2)+'\n');print(case['id'],row.get('error',row['result']['scene'] if 'result' in row else None),flush=True)
if __name__=='__main__':main()
