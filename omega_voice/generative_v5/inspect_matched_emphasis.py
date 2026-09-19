"""Fixed same-text/same-voice reference geometry, independent of Mari scoring.

The published references are synthetic, not human recordings. Their controlled
emphasis contrast can identify missing acoustic channels; it cannot establish
Mari personality or justify adopting any reference speaker identity.
"""
import argparse,json,pathlib,collections,itertools
import numpy as np,soundfile as sf,librosa,parselmouth
from .alignment import ForcedAligner,AlignmentRejected
from .session import durable_json
from ..causal_v4.renderer import sha
from ..causal_v4.evaluate import normalized

def features(path,alignment):
 y,sr=sf.read(path,dtype='float64');sound=parselmouth.Sound(y,sampling_frequency=sr);pitch=sound.to_pitch_ac(time_step=.005,pitch_floor=70,pitch_ceiling=500);hz=pitch.selected_array['frequency'];times=pitch.xs();voiced=hz>0;median=float(np.median(hz[voiced]));whole=np.sqrt(np.mean(y*y));out=[]
 for w in alignment['words']:
  start,end=w['start'],w['end'];mask=(times>=start)&(times<=end)&voiced;f=hz[mask];part=y[round(start*sr):round(end*sr)];rms=np.sqrt(np.mean(part*part));out.append({'word':w['word'],'start':start,'end':end,'duration':end-start,'voiced_seconds':float(mask.sum()*.005),'F0_median_relative_st':float(12*np.log2(np.median(f)/median)) if len(f)else None,'F0_high_relative_st':float(12*np.log2(np.percentile(f,90)/median))if len(f)else None,'level_relative_dB':float(20*np.log10(max(1e-9,rms)/max(1e-9,whole)))})
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/matched_emphasis';d.mkdir(exist_ok=True);data=pathlib.Path('/tmp/mari-emphassess-data');meta=data/'gold_df.json';rows=[json.loads(x)for x in meta.read_text().splitlines()];by_text=collections.defaultdict(list)
 for row in rows:by_text[tuple(row['src_sentence'])].append(row)
 # First two lexical transcript groups, all four voices, all emphasis versions.
 # Selection is fixed without loading scores or auditioning a reference voice.
 keys=sorted(k for k,v in by_text.items()if len({tuple(x['gold_emphasis'])for x in v})>=2)[:2];cases=[x for k in keys for x in sorted(by_text[k],key=lambda z:z['id'])];provenance=json.loads((r/'continuation/emphasis_inspection/DATASET_PROVENANCE.json').read_text());paths={pathlib.Path(x['name']).stem:x for x in provenance['files']if x['name'].endswith('.wav')};durable_json(d/'PREDECLARED.json',{'cases':cases,'metadata_sha256':sha(meta),'selection':'first two lexical transcript groups with multiple emphasis positions, all voices and versions; no model-scored selection','source_type':'synthetic annotated speech; no Mari identity change','purpose':'distinguish missing duration/relative-focus structure from simply increasing pitch or gain'})
 al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');out=[]
 for row in cases:
  p=pathlib.Path(paths[row['id']]['path']);assert sha(p)==paths[row['id']]['sha256'];wave,_=librosa.load(p,sr=16000);text=' '.join(row['src_sentence']);gold=[];offset=0
  for i,t in enumerate(row['src_sentence']):
   n=len(normalized(t))
   if i in row['gold_emphasis']:gold.extend(range(offset,offset+n))
   offset+=n
  try:
   alignment=al.align(wave,text,sha(p),True);record={'case':row,'path':str(p),'gold':gold,'alignment':alignment,'features':features(p,alignment),'admitted_clock':True}
  except AlignmentRejected as error:record={'case':row,'path':str(p),'gold':gold,'admitted_clock':False,'rejected_alignment':error.report}
  out.append(record);durable_json(d/'RESULTS.json',out);print(row['id'],record['admitted_clock'],flush=True)
 pairs=[]
 for k in keys:
  for voice in sorted({x['voice']for x in by_text[k]}):
   group=[x for x in out if tuple(x['case']['src_sentence'])==k and x['case']['voice']==voice and x['admitted_clock']]
   for emphasized,other in itertools.permutations(group,2):
    for i in set(emphasized['gold'])-set(other['gold']):
     u,v=emphasized['features'][i],other['features'][i]
     if u['F0_high_relative_st'] is None or v['F0_high_relative_st'] is None:continue
     pairs.append({'voice':voice,'text':' '.join(k),'word':u['word'],'emphasized':emphasized['case']['id'],'other':other['case']['id'],'duration_ratio':u['duration']/v['duration'],'voiced_duration_ratio':u['voiced_seconds']/max(.005,v['voiced_seconds']),'F0_high_delta_st':u['F0_high_relative_st']-v['F0_high_relative_st'],'F0_median_delta_st':u['F0_median_relative_st']-v['F0_median_relative_st'],'level_delta_dB':u['level_relative_dB']-v['level_relative_dB']})
 durable_json(d/'PAIRS.json',pairs);summary={k:{'median':float(np.median([p[k]for p in pairs])),'q25':float(np.percentile([p[k]for p in pairs],25)),'q75':float(np.percentile([p[k]for p in pairs],75))}for k in ['duration_ratio','voiced_duration_ratio','F0_high_delta_st','F0_median_delta_st','level_delta_dB']};durable_json(d/'SUMMARY.json',{'pairs':len(pairs),'admitted_clocks':sum(x['admitted_clock']for x in out),'cases':len(out),'geometry':summary,'scope':'matched synthetic prosodic mechanism observations, not human physiology or Mari character'});print(summary,flush=True)
if __name__=='__main__':main()
