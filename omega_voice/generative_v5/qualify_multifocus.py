"""Qualify the observer on published multiple-emphasis controls before repair."""
import argparse,pathlib,json,collections
import librosa
from .alignment import ForcedAligner,AlignmentRejected
from .session import durable_json
from .emphasis_evaluator import EmphasisEvaluator,word_observations
from ..causal_v4.renderer import sha
from ..causal_v4.evaluate import normalized

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('stage',choices=['align','infer']);a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/multifocus_qualification';d.mkdir(exist_ok=True)
 if a.stage=='align':
  data=pathlib.Path('/tmp/mari-emphassess-data');meta=data/'gold_df.json';rows=[json.loads(x)for x in meta.read_text().splitlines()];groups=collections.defaultdict(list)
  for row in rows:
   if len(row['gold_emphasis'])>=2:groups[tuple(row['src_sentence'])].append(row)
  keys=sorted(groups)[:2];cases=[]
  for key in keys:
   for voice in sorted({x['voice']for x in groups[key]}):cases.append(sorted([x for x in groups[key]if x['voice']==voice],key=lambda x:x['id'])[0])
  prov=json.loads((r/'continuation/emphasis_inspection/DATASET_PROVENANCE.json').read_text());paths={pathlib.Path(x['name']).stem:x for x in prov['files']if x['name'].endswith('.wav')}
  durable_json(d/'PREDECLARED.json',{'cases':cases,'selection':'first two lexical groups with multiple gold emphasis labels, first id per voice, all four voices; no score-based selection','metadata_sha256':sha(meta),'source':'published synthetic EmphAssess annotations, not human listening or Mari identity assets','gates':{'precision_min':.8,'recall_min':.7},'question':'Does the previously qualified single-focus observer retain its targets when more than one word is emphasized?','full_completion':False})
  al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');out=[]
  for case in cases:
   p=pathlib.Path(paths[case['id']]['path']);assert sha(p)==paths[case['id']]['sha256'];wave,_=librosa.load(p,sr=16000);gold=[];offset=0
   for i,t in enumerate(case['src_sentence']):
    n=len(normalized(t))
    if i in case['gold_emphasis']:gold.extend(range(offset,offset+n))
    offset+=n
   try:alignment=al.align(wave,' '.join(case['src_sentence']),sha(p),True);record={'case':case,'path':str(p),'gold':gold,'alignment':alignment,'admitted_clock':True}
   except AlignmentRejected as error:record={'case':case,'path':str(p),'gold':gold,'admitted_clock':False,'rejected_alignment':error.report}
   out.append(record);durable_json(d/'ALIGNED_CONTROLS.json',out);print(case['id'],record['admitted_clock'],flush=True)
  return
 cases=json.loads((d/'ALIGNED_CONTROLS.json').read_text());ev=EmphasisEvaluator(r);results=[];tp=fp=fn=0
 for case in cases:
  if not case['admitted_clock']:raise ValueError('unqualified clock; do not silently drop failed control')
  result=ev.infer(case['path']);words=word_observations(result,case['alignment']);pred={x['index']for x in words if x['emphasized']};gold=set(case['gold']);tp+=len(pred&gold);fp+=len(pred-gold);fn+=len(gold-pred);row={'id':case['case']['id'],'gold':sorted(gold),'predicted':sorted(pred),'words':words,'result':result};results.append(row);durable_json(d/'RESULTS.json',results);print(row['id'],gold,pred,flush=True)
 precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn);durable_json(d/'ASSESSMENT.json',{'TP':tp,'FP':fp,'FN':fn,'precision':precision,'recall':recall,'admitted_for_multiple_acoustic_emphasis':precision>=.8 and recall>=.7,'scope':'published synthetic multiple-focus acoustic control qualification only; no character admission','full_completion':False});print('precision',precision,'recall',recall,flush=True)
if __name__=='__main__':main()
