"""Resolve a source-grounded proposition mismatch into a scoped spoken focus.

This deliberately rejects paraphrase, ambiguous reference and multi-slot
changes. It does not infer an intention or choose a dramatic action.
"""
import difflib
from .runtime import words

def tokens(text):return [m.group().lower() for m in words(text)]

def resolve_contrast(text,event):
 expected=event.get('expected');observed=event.get('observed')
 if not all(isinstance(x,str) and x.strip() for x in [expected,observed]):raise ValueError('contrast requires two explicit propositions')
 a,b=tokens(expected),tokens(observed);opcodes=difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes();changes=[x for x in opcodes if x[0]!='equal']
 if not changes:return {'start_word':event['at_word'],'until_word':event['at_word'],'changed_words':[],'reason':'no mismatch'}
 if len(changes)!=1:raise ValueError('contrast has multiple unresolved semantic slots')
 tag,i,j,k,l=changes[0]
 if tag!='replace' or not 1<=j-i<=3 or not 1<=l-k<=3 or len(a)-(j-i)<2:raise ValueError('contrast is outside bounded slot substitution')
 spoken=tokens(text);matches=[n for n in range(len(spoken)-len(b)+1)if spoken[n:n+len(b)]==b]
 if len(matches)!=1:raise ValueError('observed proposition lacks one exact spoken scope')
 start=matches[0]+k;end=matches[0]+l
 if event['at_word']>start:raise ValueError('contrast discovered after its proposed spoken effect')
 return {'start_word':start,'until_word':end,'changed_words':b[k:l],'expected_words':a[i:j],'proposition_start_word':matches[0],'proposition_until_word':matches[0]+len(b),'reason':'single explicitly observed expectation mismatch'}
