"""Previously qualified, pinned UTMOS quality screen, not a character judge."""
import pathlib,sys,subprocess
from ..causal_v4.renderer import sha

class NaturalnessObserver:
 def __init__(self):
  import torch
  self.torch=torch;torch.set_num_threads(3);engine=pathlib.Path('/tmp/mari-speechmos')
  rev=subprocess.check_output(['git','-C',str(engine),'rev-parse','HEAD'],text=True).strip()
  if rev!='ed25eacbfa42b99156c36ebec67a733b5dbb9b79':raise ValueError('unselected naturalness implementation')
  weights=pathlib.Path('/root/.cache/torch/hub/checkpoints/utmos22_strong_step7459_v1.pt')
  if sha(weights)!='38aa51ab79e2a4e09a1449758a4b37e9cbb2e8235a49662a732d33a9ba1e9bff':raise ValueError('unselected naturalness weights')
  sys.path.insert(0,str(engine));from speechmos.utmos22.strong.model import UTMOS22Strong
  self.model=UTMOS22Strong();self.model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True);self.model.eval()
  self.provenance={'model':'UTMOS22Strong','revision':rev,'weights_sha256':sha(weights),'implementation_sha256':sha(__file__),'torch':torch.__version__,'scope':'predicted MOS screen; no direct listening, character, or causal appropriateness judgment'}
 def score(self,path):
  import soundfile as sf,numpy as np
  y,sr=sf.read(path,dtype='float32')
  if y.ndim!=1 or not np.isfinite(y).all():raise ValueError('malformed speech')
  with self.torch.inference_mode():value=float(self.model(self.torch.from_numpy(y)[None],sr)[0])
  return {'path':str(path),'sha256':sha(path),'predicted_MOS':value}
