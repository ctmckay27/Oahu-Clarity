"""Real executable rejection and byte-parity checks, including fast-math NaNs."""
import argparse,json,os,pathlib,struct,subprocess
import numpy as np
from .native import render,write_trajectory
from ..causal_v4.renderer import sha
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();root=pathlib.Path(a.root);d=root/'continuation/native_integrity';d.mkdir(exist_ok=True)
 text='I understand what happened. Give me a moment to check the sequence.'
 base=render(root,text,d/'cold.wav')
 zero=d/'zero.mtraj';write_trajectory(zero,np.zeros((1,29,2048)),np.zeros((100,1)))
 identical=render(root,text,d/'zero.wav',trajectory=zero)
 expected='063a844698d830af35e868d8daba0f8d228410eed786c60410eed9698a05ba10'
 parity=base['audio']['sha256']==identical['audio']['sha256']==expected
 command=base['command'];records=[]
 payload=bytearray(zero.read_bytes())
 variants={'nan':payload[:],'truncated':payload[:-4],'wrong_model':payload[:]}
 struct.pack_into('<I',variants['nan'],20,0x7fc00000)
 struct.pack_into('<I',variants['wrong_model'],8,28)
 for name,data in variants.items():
  p=d/(name+'.mtraj');p.write_bytes(data);out=d/(name+'.wav');cmd=command[:-1]+[str(out)]
  env={k:v for k,v in os.environ.items() if not k.startswith(('MARI_','QWEN_'))};env['MARI_TRAJECTORY']=str(p)
  r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=120)
  (d/(name+'.log')).write_text(r.stdout+r.stderr)
  records.append({'case':name,'returncode':r.returncode,'audio_exists':out.exists(),'passed':r.returncode!=0 and not out.exists() and 'MARI:' in r.stderr})
 result={'cold_start_no_control_and_zero_exact_parity':parity,'expected_audio_sha256':expected,'invalid_packet_rejection':records,'all_passed':parity and all(r['passed'] for r in records)}
 (d/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
 if not result['all_passed']:raise RuntimeError('native integrity qualification failed')
if __name__=='__main__':main()
