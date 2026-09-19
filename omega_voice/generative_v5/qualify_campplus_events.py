"""Test short-event reference statistics on fixed human speaker controls.

CAMPPlus also conditions SeedVC, so its scores alone can never independently
admit SeedVC output identity even if this qualification succeeds.
"""
import argparse,pathlib,json,sys,subprocess
import numpy as np,soundfile as sf,torch,torchaudio
from .session import durable_json
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/campplus_event_identity';d.mkdir(exist_ok=True);engine=pathlib.Path('/tmp/mari-seed-vc-engine')
 if subprocess.check_output(['git','-C',str(engine),'rev-parse','HEAD'],text=True).strip()!='51383efd921027683c89e5348211d93ff12ac2a8':raise ValueError('changed upstream')
 if subprocess.check_output(['git','-C',str(engine),'status','--porcelain'],text=True).strip():raise ValueError('upstream modified')
 ck=pathlib.Path('/tmp/mari-seed-vc-models/speaker/campplus_cn_common.bin')
 if sha(ck)!='3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8':raise ValueError('changed model')
 sys.path.insert(0,str(engine));from modules.campplus.DTDNN import CAMPPlus
 torch.set_num_threads(3);model=CAMPPlus(feat_dim=80,embedding_size=192).eval();state=torch.load(ck,map_location='cpu',weights_only=True)
 state={k.replace('xvector.stats','stats',1) if k.startswith('xvector.stats') else k.replace('xvector.dense','dense',1) if k.startswith('xvector.dense') else k:v for k,v in state.items()};torch.nn.Module.load_state_dict(model,state,strict=True)
 def embedding(path):
  x,sr=sf.read(path,dtype='float32');x=torch.from_numpy(x)[None];x=torchaudio.functional.resample(x,sr,16000)
  feat=torchaudio.compliance.kaldi.fbank(x,num_mel_bins=80,dither=0,sample_frequency=16000);feat-=feat.mean(dim=0,keepdim=True)
  with torch.inference_mode():z=model(feat[None],torch.tensor([feat.shape[0]//2],dtype=torch.int32))[0].numpy()
  if not np.all(np.isfinite(z))or np.linalg.norm(z)<1e-8:raise ValueError('invalid event embedding')
  return z/np.linalg.norm(z)
 cases=json.loads((r/'continuation/backchannel_reference/HUMAN_BACKCHANNELS.json').read_text())['examples'];enroll={k:embedding(r/f'continuation/event_identity/{k}.enrollment.wav')for k in ['A','B','D']};enroll['Mari']=embedding(r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 durable_json(d/'PREDECLARED.json',{'cases':cases,'gate':'all six fixed AMI events choose annotated enrolled speaker with positive margin; negative controls if positives pass','prohibited':'scores cannot independently admit outputs of the renderer conditioned with this same network','weight_sha256':sha(ck),'implementation_sha256':sha(__file__),'full_completion':False})
 rows=[]
 for c in cases:
  if sha(c['path'])!=c['sha256']:raise ValueError('changed event')
  try:
   z=embedding(c['path']);scores={k:float(z@v)for k,v in enroll.items()};expected=c['id'].split('.')[1];choice=max(scores,key=scores.get);margin=scores[expected]-max(v for k,v in scores.items()if k!=expected);row={'id':c['id'],'expected':expected,'choice':choice,'scores':scores,'margin':margin,'correct':choice==expected and margin>0}
  except Exception as exc:row={'id':c['id'],'error':repr(exc),'correct':False}
  rows.append(row);durable_json(d/'RESULTS.json',rows);print(row,flush=True)
 durable_json(d/'ASSESSMENT.json',{'correct':sum(x['correct']for x in rows),'all_positives':all(x['correct']for x in rows),'independent_seedvc_identity_admission':False,'qualified':False,'reason':'negative controls and independent output evaluation still required'if all(x['correct']for x in rows)else'fails fixed short human positives','full_completion':False})
if __name__=='__main__':main()
