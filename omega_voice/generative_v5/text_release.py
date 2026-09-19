"""Source-bound lexical availability for the native incremental generator.

This controls information access, not an emotion or intended acoustic effect.
The renderer must still demonstrate intelligibility and the causal consequence.
"""
import csv,math,pathlib,struct
import numpy as np
from ..causal_v4.runtime import verify_plan,words
from ..causal_v4.renderer import sha

def write_release(path,token_ids,earliest_frames):
 ids=list(token_ids)+[151673];frames=list(earliest_frames)
 if not token_ids or len(ids)!=len(frames) or len(token_ids)>8190:raise ValueError('release length mismatch')
 if frames[0]!=0 or any(not isinstance(x,int) or x<0 or x>8190 for x in frames):raise ValueError('invalid release frame')
 if any(a>b for a,b in zip(frames,frames[1:])):raise ValueError('nonmonotonic release')
 if any(not isinstance(x,int) or x<0 or x>151935 for x in ids):raise ValueError('invalid token identity')
 p=pathlib.Path(path);p.write_bytes(struct.pack('<2I',0x314c524d,len(token_ids))+np.asarray(list(zip(ids,frames)),dtype='<u4').tobytes())
 return {'path':str(p),'sha256':sha(p),'token_ids':ids,'earliest_frames':frames}

def read_release(path):
 b=pathlib.Path(path).read_bytes()
 if len(b)<8:raise ValueError('truncated release')
 magic,n=struct.unpack('<2I',b[:8])
 if magic!=0x314c524d or not 1<=n<=8190 or len(b)!=8+8*(n+1):raise ValueError('invalid release packet')
 pairs=np.frombuffer(b[8:],dtype='<u4').reshape(-1,2)
 return pairs[:,0].tolist(),pairs[:,1].tolist()

def compile_release(plan,tokenizer,path):
 verify_plan(plan)
 if not plan.get('realization_timeline'):raise ValueError('measured source clock required')
 encoded=tokenizer(plan['text'],add_special_tokens=False,return_offsets_mapping=True)
 ids=list(encoded['input_ids']);offsets=encoded['offset_mapping'];spans=words(plan['text'])
 gates=[]
 for item in plan['journal']:
  event=item['event'];kind=event['kind']
  if kind=='knowledge' or (kind=='thought' and event.get('mode') in {'realizing','correcting','deciding','reconsidering'}):
   at=event['at_word']
   if at==0:continue
   knot=next(k for k in plan['trajectory'] if k['at_word']==at)
   origin=plan['initial_state']['time_s'];frame=math.ceil((knot['time_s']-origin)/.08)
   gates.append({'at_word':at,'release_frame':frame,'cause':event,'state_hash':knot['state_hash']})
 frames=[0]+[i-1 for i in range(1,len(ids)+1)]
 for i,(start,end) in enumerate(offsets):
  if i==0:continue
  for gate in gates:
   if end>spans[gate['at_word']].start():frames[i]=max(frames[i],gate['release_frame'])
 for i in range(1,len(frames)):frames[i]=max(frames[i],frames[i-1])
 record=write_release(path,ids,frames)
 return dict(record,gates=gates,source_timeline=plan['realization_timeline'],
  scope='diagnostic lexical release from source events; phonation during withheld text remains under qualification')

def verify_consumption(packet,audit):
 ids,frames=read_release(packet)
 with pathlib.Path(audit).open() as f:rows=list(csv.DictReader(f))
 observed=[(int(x['token_index']),int(x['feed_frame'])) for x in rows]
 if [x[0] for x in observed]!=list(range(len(ids))):raise ValueError('not every scheduled token was consumed exactly once')
 if observed[0]!=(0,-1):raise ValueError('incorrect first-token prefill')
 for i,frame in observed[1:]:
  if frame<frames[i]:raise ValueError('future text leaked into generator')
 if any(b[1]<=a[1] for a,b in zip(observed,observed[1:])):raise ValueError('release chronology changed')
 return {'consumed_tokens':len(ids),'last_feed_frame':observed[-1][1],'audit_sha256':sha(audit),'no_early_consumption':True}
