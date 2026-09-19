"""Qualify an inspectable source/filter resynthesis route on Mari's carrier.

No voice is selected here. Unmodified-feature reconstruction must retain the
selected carrier before physical control or nonlexical synthesis is considered.
"""
import argparse,pathlib,sys,json,subprocess,hashlib,time
import numpy as np
import torch,torchaudio as ta,soundfile as sf
from safetensors.torch import load_file

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--source',default='/tmp/mari-hifiglot');a=ap.parse_args()
 r=pathlib.Path(a.root);src=pathlib.Path(a.source);d=r/'continuation/source_filter';d.mkdir(exist_ok=True)
 rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip()
 if rev!='78951625e8f0a113f401fef7032e98d58cfa7dac' or subprocess.check_output(['git','diff','HEAD'],cwd=src):raise ValueError('unverified source')
 weights=pathlib.Path('/tmp/mari-hifiglot-checkpoints/model.safetensors')
 if sha(weights)!='8955e6f43593099b912c25e2e5f9abd2120950e2927b34c5aaebbe3df9eb14c4':raise ValueError('checkpoint mismatch')
 sys.path.insert(0,str(src/'src'))
 from neural_formant_synthesis.third_party.hifi_gan.env import AttrDict
 from neural_formant_synthesis.feature_extraction import feature_extractor,Normaliser,MedianPool1d
 from neural_formant_synthesis.models import SourceFilterFormantSynthesisGenerator
 torch.set_num_threads(4);torch.manual_seed(114514)
 h=AttrDict(json.loads((src/'configs/hifiglot/config_hifigan.json').read_text()))
 f=AttrDict(json.loads((src/'configs/hifiglot/config_feature_map.json').read_text()))
 model=SourceFilterFormantSynthesisGenerator(f,h,pretrained_fm=None,freeze_fm=False)
 model.load_state_dict(load_file(weights),strict=True);model.eval()
 extract=feature_extractor(h.sampling_rate,h.win_size,h.hop_size,10000,4)
 median=MedianPool1d(kernel_size=7,stride=1,padding=0,same=True);norm=Normaliser()
 p=r/'continuation/calibration/unchanged.wav';x,sr=sf.read(p,dtype='float64');x=ta.functional.resample(torch.tensor(x),sr,h.sampling_rate)
 started=time.monotonic()
 formants,energy,centroid,tilt,pitch,voicing=extract(x)
 formants=median(formants.T.unsqueeze(1)).squeeze(1).T;pitch=pitch.squeeze(0);voicing=voicing.squeeze(0)
 np.savez_compressed(d/'carrier_features.npz',formants=formants.numpy(),energy=energy.numpy(),centroid=centroid.numpy(),tilt=tilt.numpy(),pitch=pitch.numpy(),voicing=voicing.numpy())
 pitch_n,formants_n,tilt_n,centroid_n,energy_n=norm(pitch.clone(),formants.clone(),tilt.clone(),centroid.clone(),energy.clone())
 feat=torch.cat((pitch_n[:,None],formants_n,tilt_n[:,None],centroid_n[:,None],energy_n[:,None],voicing[:,None]),dim=-1).T.float()[None]
 if not torch.isfinite(feat).all():raise ValueError('nonfinite extracted feature; no silent repair')
 with torch.inference_mode():
  torch.manual_seed(114514);y,_=model(feat,pitch[None]);torch.manual_seed(114514);repeat,_=model(feat,pitch[None])
 if not torch.isfinite(y).all():raise ValueError('nonfinite resynthesis')
 output=d/'unmodified_features.wav'
 if output.exists():raise FileExistsError(output)
 y24=ta.functional.resample(y.squeeze().cpu(),h.sampling_rate,24000)
 sf.write(output,y24.numpy(),24000,subtype='FLOAT')
 record={'role':'unmodified-feature source-filter diagnostic; not selected backend or completed voice',
  'input':str(p),'input_sha256':sha(p),'audio':str(output),'audio_sha256':sha(output),
  'source_repository':'PupuAI/HiFi-Glot','source_revision':rev,'weights_sha256':sha(weights),
  'seed':114514,'source_sample_rate':h.sampling_rate,'features_shape':list(feat.shape),
  'repeat_exact':bool(torch.equal(y,repeat)),'peak':float(y24.abs().max()),'duration_s':len(y24)/24000,
  'elapsed_s':time.monotonic()-started,'torch':torch.__version__,'full_completion':False,'feature_intervention':None}
 (d/'RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
