"""Isolated wide-content/fixed-Mari-reference conversion capability probe.

This is not a selected voice route. No AR/style conversion, donor identity
selection, text directions, hosted service or silent backend substitution.
"""
import argparse,pathlib,json,hashlib,sys,subprocess,time,os
import numpy as np,soundfile as sf
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
class ConversionProbe:
 def __init__(self,root):
  self.root=pathlib.Path(root);self.engine=pathlib.Path('/tmp/mari-seed-vc-engine');self.models=pathlib.Path('/tmp/mari-seed-vc-models');self.evidence=self.root/'continuation/seed_vc_inspection'
  self.revision='51383efd921027683c89e5348211d93ff12ac2a8'
  assert subprocess.check_output(['git','-C',str(self.engine),'rev-parse','HEAD'],text=True).strip()==self.revision
  assert not subprocess.check_output(['git','-C',str(self.engine),'status','--porcelain'],text=True).strip()
  self.manifest=json.loads((self.evidence/'MODEL_PROVENANCE.json').read_text());assert self.manifest['complete']
  for f in self.manifest['files']:assert sha(f['path'])==f['sha256']
  self.anchor=self.root/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav'
  assert sha(self.anchor)=='73bb894fc75e18dddf76e3327f3bf48e161b7dd8b30fbe4949c3ac7318c29567'
  os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
  sys.path.insert(0,str(self.engine))
  import torch,yaml
  from hydra.utils import instantiate
  from omegaconf import OmegaConf
  self.torch=torch;torch.set_num_threads(4);torch.manual_seed(93800)
  if tuple(int(x)for x in torch.__version__.split('+')[0].split('.')[:2])<(2,6):raise ValueError('safe torch loader required')
  cfg=OmegaConf.load(self.engine/'configs/v2/vc_wrapper.yaml')
  # The inspected timbre-only path uses the wide tokenizer, CFM, reference
  # encoder and vocoder. Do not load unused AR or narrow/style-conversion parts.
  cfg.ar=None;cfg.ar_length_regulator=None;cfg.content_extractor_narrow=None
  cfg.content_extractor_wide.tokenizer_name=str(self.models/'tokenizer')
  cfg.content_extractor_wide.ssl_model_name=str(self.models/'hubert')
  cfg.vocoder.pretrained_model_name_or_path=str(self.models/'vocoder')
  model=instantiate(cfg).cpu().float().eval();loaded=[]
  def load(module,state,label,allowed_missing=()):
   miss,extra=module.load_state_dict(state,strict=False)
   if extra or any(not any(k.startswith(p)for p in allowed_missing)for k in miss):raise ValueError('unloaded conversion weights '+label+repr((miss,extra)))
   loaded.append({'part':label,'missing':list(miss),'unexpected':list(extra),'allowed_missing_prefixes':list(allowed_missing)})
  strip=lambda state:{k.removeprefix('module.'):v for k,v in state.items()}
  ck=torch.load(self.models/'cfm/cfm_small.pth',map_location='cpu',weights_only=True)['net']
  load(model.cfm,strip(ck['cfm']),'cfm');load(model.cfm_length_regulator,strip(ck['length_regulator']),'length_regulator')
  load(model.content_extractor_wide,torch.load(self.models/'astral/bsq2048_light.pth',map_location='cpu',weights_only=True),'content_extractor',('ssl_model.',))
  load(model.style_encoder,torch.load(self.models/'speaker/campplus_cn_common.bin',map_location='cpu',weights_only=True),'reference_encoder')
  model.vocoder.remove_weight_norm();self.model=model
  self.provenance={'upstream_revision':self.revision,'model_manifest':self.manifest,'loads':loaded,'ar_enabled':False,'style_conversion':False,'anonymization':False,'target_anchor_sha256':sha(self.anchor),'dtype':'float32','device':'cpu','source_sha256':sha(__file__),'role':'isolated diagnostic; not production or completed voice'}
 def render(self,source,destination,seed=93800):
  import librosa
  torch=self.torch;m=self.model;p=pathlib.Path(destination)
  if p.exists():raise FileExistsError(p)
  source=pathlib.Path(source);sr=m.sr;start=time.monotonic();torch.manual_seed(seed)
  x=librosa.load(source,sr=sr)[0];ref=librosa.load(self.anchor,sr=sr)[0]
  if x.ndim!=1 or not np.all(np.isfinite(x))or not .15<=len(x)/sr<=15:raise ValueError('unsupported diagnostic input')
  sx=librosa.resample(x,orig_sr=sr,target_sr=16000);sy=librosa.resample(ref,orig_sr=sr,target_sr=16000)
  with torch.inference_mode():
   tx=torch.from_numpy(x)[None];ty=torch.from_numpy(ref)[None];xx=torch.from_numpy(sx)[None];yy=torch.from_numpy(sy)[None]
   xm=m.mel_fn(tx);ym=m.mel_fn(ty)
   _,xc,_=m.content_extractor_wide(xx,[len(sx)]);_,yc,_=m.content_extractor_wide(yy,[len(sy)])
   style=m.compute_style(yy)
   content,_=m.cfm_length_regulator(xc,ylens=torch.tensor([xm.shape[-1]]));reference,_=m.cfm_length_regulator(yc,ylens=torch.tensor([ym.shape[-1]]))
   joined=torch.cat([reference,content],1)
   mel=m.cfm.inference(joined,torch.tensor([joined.shape[1]]),ym,style,25,inference_cfg_rate=[.7,.7],random_voice=False)
   # CFM sees reference first; only newly realized source duration is emitted.
   mel=mel[:,:,ym.shape[-1]:];y=m.vocoder(mel).reshape(-1).numpy()
  if not np.all(np.isfinite(y))or len(y)==0:raise ValueError('invalid conversion output')
  p.parent.mkdir(parents=True,exist_ok=True);sf.write(p,y,sr,subtype='PCM_16')
  receipt={'source':str(source),'source_sha256':sha(source),'output_sha256':sha(p),'source_seconds':len(x)/sr,'output_seconds':len(y)/sr,'peak':float(np.max(np.abs(y))),'seed':seed,'diffusion_steps':25,'cfg':[.7,.7],'source_content_frames':int(xc.shape[-1]),'target_content_frames':int(yc.shape[-1]),'elapsed_seconds':time.monotonic()-start,'provenance':self.provenance,'quality_and_identity_admitted':False}
  p.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items()if k!='provenance'}),flush=True);return receipt
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('source');ap.add_argument('output');a=ap.parse_args();probe=ConversionProbe(a.root);probe.render(a.source,a.output)
if __name__=='__main__':main()
