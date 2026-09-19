"""Fixed upstream emphasis labels before observing any Mari intervention."""
import argparse,pathlib,json
import numpy as np,soundfile as sf
from .session import durable_json
from .alignment import ForcedAligner
from .emphasis_evaluator import EmphasisEvaluator,word_observations
from ..causal_v4.renderer import sha
from ..causal_v4.evaluate import normalized

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('stage',choices=['align','infer']);a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/emphasis_qualification';d.mkdir(exist_ok=True);src=r/'continuation/emphasis_inspection/source';cases=[json.loads(x)for x in(src/'test/test_data/df_test.json').read_text().splitlines()]
 if a.stage=='align':
  import librosa
  al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');rows=[]
  for case in cases:
   path=src/case['tgt_audiopath'];text=' '.join(case['src_sentence']);wave,_=librosa.load(path,sr=16000)
   # The fixed published transcript metadata is an independent source. These
   # controls have synthesis annotations independent of our inference.
   alignment=al.align(wave,text,sha(path),True);gold=[];offset=0
   for i,token in enumerate(case['src_sentence']):
    n=len(normalized(token))
    if i in case['gold_emphasis']:gold.extend(range(offset,offset+n))
    offset+=n
   rows.append({'id':case['id'],'path':str(path),'source_case':case,'gold_indices':gold,'alignment':alignment,'transcript_authority':'fixed published synthetic-speech transcript/emphasis annotation, facebookresearch/emphassess pinned test data'});durable_json(d/'ALIGNED_CONTROLS.json',rows);print(case['id'],alignment['minimum_word_probability'],flush=True)
  durable_json(d/'PREDECLARED.json',{'controls':len(cases),'word_overlap_threshold':.5,'gates':{'precision_min':.8,'recall_min':.7,'non_speech_positive_frame_fraction_max':.01},'source_pipeline_documented_baseline':{'precision':1,'recall':.71,'F1':.83},'adaptation':'qualified CTC word clocks on same-language published synthetic-speech transcript metadata; no SimAlign/translation/WhisperX inference','scope':'acoustic word emphasis; no intent, Mari, naturalness or character judgment'});return
 pre=json.loads((d/'PREDECLARED.json').read_text());ev=EmphasisEvaluator(r);durable_json(d/'EXECUTION_PROVENANCE.json',ev.provenance);rows=[];TP=FP=FN=0
 for case in json.loads((d/'ALIGNED_CONTROLS.json').read_text()):
  result=ev.infer(case['path']);words=word_observations(result,case['alignment']);pred={w['index']for w in words if w['emphasized']};gold=set(case['gold_indices']);TP+=len(pred&gold);FP+=len(pred-gold);FN+=len(gold-pred)
  rows.append({'id':case['id'],'gold':sorted(gold),'predicted':sorted(pred),'words':words,'result':result});durable_json(d/'RESULTS.json',rows);print(case['id'],'gold',gold,'predicted',pred,flush=True)
 negatives=[]
 for k,w in [('silence',np.zeros(32000)),('tone',.1*np.sin(2*np.pi*220*np.arange(32000)/16000))]:
  path=d/(k+'.wav');sf.write(path,w,16000,subtype='PCM_16');result=ev.infer(path);negatives.append({'id':k,'result':result});durable_json(d/'NEGATIVE_RESULTS.json',negatives)
 precision=TP/max(1,TP+FP);recall=TP/max(1,TP+FN);checks={'precision':precision>=pre['gates']['precision_min'],'recall':recall>=pre['gates']['recall_min'],'negative_rejection':all(x['result']['positive_frame_fraction']<=.01 for x in negatives)}
 durable_json(d/'ASSESSMENT.json',{'TP':TP,'FP':FP,'FN':FN,'precision':precision,'recall':recall,'F1':2*precision*recall/max(1e-8,precision+recall),'checks':checks,'admitted_for_acoustic_emphasis':all(checks.values()),'full_completion':False,'scope':pre['scope']});print(checks,flush=True)
if __name__=='__main__':main()
