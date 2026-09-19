"""Causal-unit successor to rejected full-utterance flow conversion.

Only the completed thought unit is delivered; future state cannot regenerate
it. The fixed anchor still determines style. A verified delivered unit becomes
reference context for the next unit, without later audio in that context.
"""
import argparse,json,pathlib
import numpy as np,soundfile as sf
from .session import durable_json
from .seed_vc_diagnostic import ConversionProbe,sha
from .alignment import ForcedAligner
from ..causal_v4.evaluate import Evaluator

FULL='I thought the spare key was upstairs. I can see it beside the blue bowl.'
PREFIX='I thought the spare key was upstairs.'
SUFFIX='I can see it beside the blue bowl.'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('stage',choices=['prepare','render','score']);a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/seed_causal_continuation';d.mkdir(exist_ok=True)
 if a.stage=='prepare':
  ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json')
  s=r/'continuation/factive_scene/0_no_result.wav';t=r/'continuation/factive_scene/0_related.wav';q=ev.evaluate(s,FULL)
  if q['wer']!=0:raise ValueError('source transcript not verified')
  alignment=al.align(ev.load(s),FULL,sha(s),True);x,sr=sf.read(s,dtype='int16');y,sry=sf.read(t,dtype='int16');assert sr==sry==24000
  left=alignment['words'][6]['end'];right=alignment['words'][7]['start'];event_sample=int(round(right*sr))
  # A real quiet interval between verified complete words is required. Select
  # its quietest 20ms window, rather than cutting phonation or adding silence.
  width=round(.02*sr);lo=int(np.ceil(left*sr));hi=int(np.floor(right*sr));candidates=[]
  peak=float(np.abs(x.astype(float)).max()/32768);threshold=max(.001,.01*peak)
  for start in range(lo,hi-width+1,24):
   rms=float(np.sqrt(np.mean((x[start:start+width].astype(float)/32768)**2)))
   if rms<=threshold:candidates.append((rms,start+width//2))
  if not candidates:raise ValueError('no verified quiet thought boundary')
  rms,cut=min(candidates)
  if not np.array_equal(x[:event_sample],y[:event_sample]):raise ValueError('source causality already failed')
  sf.write(d/'prefix.source.wav',x[:cut],sr,subtype='PCM_16')
  for k,z in [('no_result',x),('related',y)]:sf.write(d/(k+'.source.wav'),z[cut:],sr,subtype='PCM_16')
  durable_json(d/'PREDECLARED.json',{'source_paths':[str(s),str(t)],'source_hashes':[sha(s),sha(t)],'full_text':FULL,'prefix_text':PREFIX,'suffix_text':SUFFIX,'alignment':alignment,'quality':q,'source_cut_sample':cut,'source_event_sample':event_sample,'source_boundary_quiet_rms':rms,'quiet_threshold':threshold,'prefix_source_sha256':sha(d/'prefix.source.wav'),'source_future_state_has_no_prefix_difference':True,'gates':{'WER':0,'speaker_similarity_min':.6013473320007324,'pre_event_PCM_exact':True,'no_future_reference_content':True,'contour_direction_preserved':True,'UTMOS_loss_max':.15},'scope':'diagnostic causal thought-unit flow realization; no personality admission','full_completion':False})
  print('prepared',left,right,cut/sr,rms,flush=True);return
 pre=json.loads((d/'PREDECLARED.json').read_text())
 if a.stage=='render':
  probe=ConversionProbe(r)
  prefix=d/'prefix.wav'
  if not prefix.exists():probe.render(d/'prefix.source.wav',prefix)
  else:
   rec=json.loads(prefix.with_suffix('.receipt.json').read_text());assert sha(prefix)==rec['output_sha256'] and sha(d/'prefix.source.wav')==rec['source_sha256']
  # Check the delivered prefix before it is allowed to condition a successor.
  ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');q=ev.evaluate(prefix,PREFIX)
  durable_json(d/'PREFIX_QUALITY.json',q)
  if q['wer']!=0 or not q['identity_pass']:raise ValueError('unverified prefix cannot condition delivery')
  px,sr=sf.read(prefix,dtype='int16');peak=float(np.abs(px.astype(float)).max()/32768);tail_rms=float(np.sqrt(np.mean((px[-240:].astype(float)/32768)**2)))
  if tail_rms>max(.001,.01*peak):raise ValueError('generated prefix does not end at quiet boundary')
  rows=[]
  for k in ['no_result','related']:
   suffix=d/(k+'.suffix.wav')
   rec=probe.render(d/(k+'.source.wav'),suffix,heard_prefix=prefix)
   sy,ss=sf.read(suffix,dtype='int16');assert ss==sr==24000
   sf.write(d/(k+'.joined.wav'),np.concatenate([px,sy]),sr,subtype='PCM_16')
   rows.append({'case':k,'prefix_sha256':sha(prefix),'prefix_samples':len(px),'prefix_tail_rms':tail_rms,'suffix_receipt':rec,'joined_sha256':sha(d/(k+'.joined.wav'))})
   durable_json(d/'RESULTS.json',rows)
  return
 from .acoustics import measure
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');rows=[]
 for row in json.loads((d/'RESULTS.json').read_text()):
  k=row['case'];p=d/(k+'.joined.wav');q=ev.evaluate(p,FULL);sq=ev.evaluate(d/(k+'.suffix.wav'),SUFFIX)
  alignment=al.align(ev.load(p),FULL,sha(p),q['wer']==0) if q['wer']==0 else None
  rows.append({'case':k,'quality':q,'suffix_quality':sq,'alignment':alignment,'acoustics':measure(p),'checks':{'WER':q['wer']==0 and sq['wer']==0,'identity':q['identity_pass'] and sq['identity_pass'],'no_clipping':q['audio']['clipping_fraction']==0}});durable_json(d/'QUALITY_RESULTS.json',rows);print(k,rows[-1]['checks'],rows[-1]['acoustics'],flush=True)
 px,_=sf.read(d/'prefix.wav',dtype='int16');x,_=sf.read(d/'no_result.joined.wav',dtype='int16');y,_=sf.read(d/'related.joined.wav',dtype='int16')
 checks={'quality':all(all(z['checks'].values())for z in rows),'prefix_PCM_exact':np.array_equal(x[:len(px)],px) and np.array_equal(y[:len(px)],px),'distinct_successor':not np.array_equal(x[len(px):],y[len(px):]),'contour_direction_preserved':rows[1]['acoustics']['final_slope_semitones_per_s']<rows[0]['acoustics']['final_slope_semitones_per_s']}
 durable_json(d/'ASSESSMENT.json',{'checks':checks,'all_pass':all(checks.values()),'UTMOS_pending':True,'scope':'causal continuation diagnostic, no character admission','full_completion':False})
if __name__=='__main__':main()
