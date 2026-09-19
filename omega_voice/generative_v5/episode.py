"""Causal information units joined through one recoverable performance session.

Units are supplied at changes in available information or interaction. This
layer does not split every clause, rewrite text, invent hesitation, or provide
future unit text to an earlier renderer call.
"""
import pathlib,json
import numpy as np,soundfile as sf
from .session import durable_json
from ..causal_v4.renderer import sha

def realize(session,units,directory):
 directory=pathlib.Path(directory);directory.mkdir(parents=True,exist_ok=True)
 if (directory/'episode.wav').exists():raise FileExistsError('episode already realized')
 saved=json.loads((directory/'units.json').read_text()) if (directory/'units.json').exists() else []
 if saved:
  current=json.loads(session.state_path.read_text())
  if current['last_request']!=saved[-1]['id'] or current['receipt_sha256']!=saved[-1]['receipt_sha256']:raise ValueError('episode/session recovery boundary differs')
 receipts=[];parts=[];texts=[];cursor=0
 for i,unit in enumerate(units):
  if set(unit)-{'id','text','scene','seed','boundary_cause'}:raise ValueError('unknown episode unit field')
  cause=unit.get('boundary_cause')
  if i and (not isinstance(cause,dict) or not cause.get('text')):raise ValueError('each later information boundary requires a source cause')
  if i<len(saved):
   item=saved[i];receipt_path=session.directory/unit['id']/'receipt.json'
   if item['id']!=unit['id'] or sha(receipt_path)!=item['receipt_sha256']:raise ValueError('completed episode unit provenance changed')
   receipt=json.loads(receipt_path.read_text())
   if not receipt['quality_admitted'] or receipt['delivered_plan']['text']!=unit['text'] or receipt['delivered_plan']['scene']!=unit.get('scene',{}):raise ValueError('completed unit meaning changed')
  else:receipt=session.render_turn(unit['id'],unit['text'],unit.get('scene'),unit.get('seed',88000+i),diagnostic=True)
  audio=pathlib.Path(receipt['trials'][-1]['audio']);y,sr=sf.read(audio,dtype='int16')
  if sr!=24000 or y.ndim!=1:raise ValueError('unqualified episode audio format')
  if i<len(saved) and sha(audio)!=saved[i]['audio_sha256']:raise ValueError('completed episode waveform changed')
  receipts.append({'id':unit['id'],'boundary_cause':cause,'start_sample':cursor,'end_sample':cursor+len(y),
    'audio_sha256':sha(audio),'receipt':str(session.directory/unit['id']/'receipt.json'),
    'receipt_sha256':sha(session.directory/unit['id']/'receipt.json'),'parent_state_hash':receipt['parent_state_hash'],
    'unresolved_channels':receipt['unresolved_channels']})
  cursor+=len(y);parts.append(y);texts.append(unit['text']);durable_json(directory/'units.json',receipts)
 joined=directory/'episode.wav';sf.write(joined,np.concatenate(parts),24000,subtype='PCM_16')
 q=session.evaluator.evaluate(joined,' '.join(texts))
 result={'role':'diagnostic continuous causal episode','full_completion':False,'units':receipts,'quality':q,
   'join':'unaltered PCM concatenation; no synthetic pause, crossfade, or random microbehavior',
   'waveform_sha256':sha(joined),'all_text':' '.join(texts),'all_units_admitted':True,
   'episode_quality_admitted':q['quality_screen_pass'] and q['wer']==0}
 durable_json(directory/'receipt.json',result);return result
