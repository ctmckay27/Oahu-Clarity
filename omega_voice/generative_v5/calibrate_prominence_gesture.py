"""Independent reference-text calibration for a coupled local prominence gesture."""
import argparse,pathlib,json,hashlib,collections,itertools
import numpy as np,librosa,parselmouth,soundfile as sf
from .alignment import ForcedAligner,AlignmentRejected
from .session import durable_json
from ..causal_v4.renderer import sha
from ..causal_v4.evaluate import normalized

def pitch_shape(path,word):
 y,sr=sf.read(path);s=parselmouth.Sound(y,sampling_frequency=sr);p=s.to_pitch_ac(time_step=.005,pitch_floor=70,pitch_ceiling=500);hz=p.selected_array['frequency'];ts=p.xs();mask=(hz>0)&(ts>=word['start'])&(ts<=word['end']);x=ts[mask];f=hz[mask]
 if len(f)<10:return None
 grid=np.linspace(0,1,33);phase=(x-word['start'])/(word['end']-word['start']);v=np.interp(grid,phase,12*np.log2(f));trend=np.linspace(v[0],v[-1],len(v));part=y[round(word['start']*sr):round(word['end']*sr)];level=20*np.log10(max(1e-8,np.sqrt(np.mean(part*part)))/max(1e-8,np.sqrt(np.mean(y*y))))
 return {'residual':(v-trend).tolist(),'level_relative_dB':float(level),'voiced_frames':len(f),'absolute_pitch_st':v.tolist()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/coupled_prominence_calibration';d.mkdir(exist_ok=True);sources=[r/'continuation/prosodic_evaluator/FEATURES.json',r/'continuation/paired_emphasis_evaluator/FRESH_FEATURES.json',r/'continuation/matched_emphasis/RESULTS.json'];used={tuple(x['case']['src_sentence'])for p in sources for x in json.loads(p.read_text())};meta=pathlib.Path('/tmp/mari-emphassess-data/gold_df.json');groups=collections.defaultdict(list)
 for line in meta.read_text().splitlines():
  x=json.loads(line);groups[tuple(x['src_sentence'])].append(x)
 eligible=[k for k,v in groups.items()if k not in used and k[-1]=='.'and len({tuple(x['gold_emphasis'])for x in v})>=2];chosen=sorted(eligible,key=lambda k:hashlib.sha256(json.dumps(k).encode()).hexdigest())[:4];cases=[x for k in chosen for x in sorted(groups[k],key=lambda x:x['id'])];durable_json(d/'PREDECLARED.json',{'cases':cases,'selection':'first four hash-ordered declarative text groups with alternative emphasis labels, excluding all observer-training/qualification and original-duration-calibration texts; all voices/versions','geometry':'median added-emphasis change in pitch residual after removing each word linear endpoint trend; reference relative level difference. Native endpoint contour and existing duration mapping preserved.','no_Mari_fit':True,'observer_model_sha256':sha(r/'continuation/paired_emphasis_evaluator/MODEL.joblib'),'full_completion':False});al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');done=json.loads((d/'REFERENCES.json').read_text())if(d/'REFERENCES.json').exists()else [];seen={x['case']['id']for x in done};prov=json.loads((r/'continuation/emphasis_inspection/DATASET_PROVENANCE.json').read_text());paths={pathlib.Path(x['name']).stem:x for x in prov['files']if x['name'].endswith('.wav')}
 for c in cases:
  if c['id']in seen:continue
  p=pathlib.Path(paths[c['id']]['path']);assert sha(p)==paths[c['id']]['sha256'];y,_=librosa.load(p,sr=16000)
  try:alignment=al.align(y,' '.join(c['src_sentence']),sha(p),True);admitted=True
  except AlignmentRejected as e:alignment=e.report;admitted=False
  gold=[];offset=0
  for i,token in enumerate(c['src_sentence']):
   n=len(normalized(token))
   if i in c['gold_emphasis']:gold.extend(range(offset,offset+n))
   offset+=n
  done.append({'case':c,'path':str(p),'audio_sha256':sha(p),'gold':gold,'alignment':alignment,'admitted':admitted,'shapes':[pitch_shape(p,w)for w in alignment['words']]if admitted else None});durable_json(d/'REFERENCES.json',done);print(len(done),'of',len(cases),c['id'],admitted,flush=True)
 grouped=collections.defaultdict(list)
 for x in done:
  if x['admitted']:grouped[(tuple(x['case']['src_sentence']),x['case']['voice'])].append(x)
 pairs=[]
 for key,rs in grouped.items():
  for a,b in itertools.permutations(rs,2):
   for i in set(b['gold'])-set(a['gold']):
    aa=a['shapes'][i];bb=b['shapes'][i]
    if aa is None or bb is None:continue
    delta=np.asarray(bb['residual'])-np.asarray(aa['residual']);pairs.append({'before':a['case']['id'],'after':b['case']['id'],'word':b['alignment']['words'][i]['word'],'pitch_residual_delta_st':delta.tolist(),'level_relative_delta_dB':bb['level_relative_dB']-aa['level_relative_dB']})
 if not pairs:raise ValueError('no admitted calibration pairs')
 profile=np.median([x['pitch_residual_delta_st']for x in pairs],axis=0);level=float(np.median([x['level_relative_delta_dB']for x in pairs]));durable_json(d/'PAIRS.json',pairs);result={'schema':'mari-coupled-prominence-calibration/1.0','source_sha256':sha(d/'REFERENCES.json'),'selection_sha256':sha(d/'PREDECLARED.json'),'pairs_sha256':sha(d/'PAIRS.json'),'implementation_sha256':sha(__file__),'reference_cases':len(done),'admitted_reference_cases':sum(x['admitted']for x in done),'pairs':len(pairs),'normalized_time':np.linspace(0,1,33).tolist(),'pitch_residual_delta_st':profile.tolist(),'relative_level_delta_dB':level,'reference_strength':.55,'no_Mari_fit':True,'observer_training_text_overlap':False,'physiological_model_claim':False,'full_completion':False};durable_json(d/'CALIBRATION.json',result);print('PROFILE',profile.tolist(),'LEVEL',level,'PAIRS',len(pairs),flush=True)
if __name__=='__main__':main()
