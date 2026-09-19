"""Compile source-grounded finality to measured native boundary control.

Only the boundary-contour actuator is under qualification. Other coordinates
remain explicitly unresolved; this module is not a completed performance route.
"""
import math
import numpy as np
from ..causal_v4.runtime import words,verify_plan
from ..causal_v4.evaluate import normalized

def boundaries(text):
 matches=words(text);result=[]
 for i,m in enumerate(matches):
  following=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)]
  if i+1==len(matches) or any(c in following for c in '.!?;:'):
   result.append(i)
 return result

def compile_finality(plan,alignment,duration):
 verify_plan(plan)
 aligned=alignment['words'];n=len(plan['words'])
 if not alignment.get('exact_words') or len(aligned)!=n or normalized(plan['text'])!=[a['word'] for a in aligned]:
  raise ValueError('exact text alignment required')
 if any(not 0<=a['start']<a['end']<=duration+.05 for a in aligned):raise ValueError('invalid word time')
 if any(a['start']>b['start'] for a,b in zip(aligned,aligned[1:])):raise ValueError('nonmonotonic alignment')
 # Native weights are applied to the next autoregressive frame, hence -1.
 weights=np.zeros((min(8192,math.ceil((duration+2)/.08)),1),np.float32)
 segments=[];phrase_start=0
 for word in boundaries(plan['text']):
  knot=plan['trajectory'][word];f=knot['controls']['finality']
  strength=float(np.clip((f-.7)*(.7/.3),-.7,.7))
  cause_words=[e['event']['at_word'] for e in plan['journal'] if e['event']['at_word']<=word]
  latest=max(cause_words) if cause_words else phrase_start
  start=max(aligned[phrase_start]['start'],aligned[min(latest,word)]['start'],aligned[word]['end']-.55)
  end=aligned[word]['end']
  first=max(0,math.floor(start/.08)-1);last=max(first+1,math.ceil(end/.08)-1)
  weights[first:last,0]=strength
  segments.append({'word':word,'start_s':start,'end_s':end,'native_frames':[first,last],
                   'finality_intent':f,'strength':strength,'causes':knot['causes'],
                   'interpretation':'boundary contour only; acoustic direction is not a judgment of certainty'})
  phrase_start=word+1
 return weights,{'segments':segments,'plan_hash':plan['plan_hash'],
                 'alignment_source_sha256':alignment['source_sha256'],
                 'status':'diagnostic native contour route; full performance unresolved'}

def alignment_error(segments,observed):
 """Check whether actuation still addresses the intended final words."""
 if not observed.get('exact_words'):return float('inf')
 return max((abs(s['end_s']-observed['words'][s['word']]['end']) for s in segments),default=0)
