"""Discriminate temporal-pooling failure from absent short-event identity cues.

Original unmodified-waveform gates remain failed. Fixed padding/repetition are
analysis transforms only; no generated waveform is changed or admitted here.
"""
import argparse,pathlib,json,hashlib,math
import numpy as np,soundfile as sf,torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier
from transformers import WavLMForXVector,Wav2Vec2FeatureExtractor
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);out=r/'continuation/event_duration';out.mkdir(exist_ok=True)
 modeldir=pathlib.Path('/tmp/mari-wavlm-sv');meta=json.loads((r/'continuation/wavlm_event_identity/MODEL_PROVENANCE.json').read_text())
 assert all(sha(modeldir/x['path'])==x['sha256']for x in meta['files'])
 assert tuple(int(x)for x in torch.__version__.split('+')[0].split('.')[:2])>=(2,6)
 torch.set_num_threads(2);ecapa=EncoderClassifier.from_hparams(source=str(r/'models/ecapa'),savedir=str(r/'models/ecapa'),run_opts={'device':'cpu'})
 wavlm=WavLMForXVector.from_pretrained(modeldir).eval();processor=Wav2Vec2FeatureExtractor.from_pretrained(modeldir)
 def load(p):
  y,sr=sf.read(p,dtype='float32');assert y.ndim==1;g=math.gcd(sr,16000);return resample_poly(y,16000//g,sr//g)
 def embed(y,model):
  with torch.inference_mode():
   if model=='ECAPA':v=ecapa.encode_batch(torch.from_numpy(y).unsqueeze(0)).squeeze().numpy()
   else:v=wavlm(**processor(y,sampling_rate=16000,return_tensors='pt')).embeddings[0].numpy()
  if not np.all(np.isfinite(v)) or np.linalg.norm(v)<1e-12:raise ValueError('nonfinite/zero embedding')
  return v/np.linalg.norm(v)
 (out/'PREDECLARED.json').write_text(json.dumps({'role':'causal evaluator diagnostic only','transforms':['right_zero_pad_to_1.5s','cycle_to_1.5s'],'reason':'WavLM TDNN produced single-frame variance NaN on a .32s input; distinguish pooling context from absent identity cues','original_failed_gates_unchanged':True,'event_admission':False},indent=2)+'\n')
 examples=json.loads((r/'continuation/backchannel_reference/HUMAN_BACKCHANNELS.json').read_text())['examples'];rows=[]
 for model in ['ECAPA','WavLM']:
  enroll={x:embed(load(r/f'continuation/event_identity/{x}.enrollment.wav'),model)for x in 'ABD'};enroll['Mari']=embed(load(r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav'),model)
  for case in examples:
   assert sha(case['path'])==case['sha256'];y=load(case['path']);assert len(y)<24000
   for method in ['zero','cycle']:
    z=np.pad(y,(0,24000-len(y)))if method=='zero'else np.tile(y,math.ceil(24000/len(y)))[:24000]
    row={'model':model,'id':case['id'],'transform':method,'expected':case['id'].split('.')[1]}
    try:
     v=embed(z.astype(np.float32),model);scores={k:float(v@w)for k,w in enroll.items()};choice=max(scores,key=scores.get);row.update(choice=choice,scores=scores,correct=choice==row['expected'])
    except Exception as e:row.update(error=repr(e),correct=False)
    rows.append(row);(out/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n');print(row,flush=True)
 result={model+'_'+method:sum(x['correct']for x in rows if x['model']==model and x['transform']==method)for model in ['ECAPA','WavLM']for method in ['zero','cycle']}
 (out/'ASSESSMENT.json').write_text(json.dumps({'correct_of_six':result,'identity_evaluator_admitted':False,'scope':'only fixed diagnostic transforms; any future admission requires new held-out duration/speaker/negative controls'},indent=2)+'\n')
if __name__=='__main__':main()
