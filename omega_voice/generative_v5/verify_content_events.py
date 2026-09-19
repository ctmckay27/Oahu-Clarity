"""Discriminate first-codebook event control from identity leakage.

AMI excerpts are donor-event diagnostics, never selected speaker assets. No
trained/semantic interpretation of their first codebook is assumed.
"""
import argparse,pathlib,json,os,subprocess,re
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
from .reference_calibration import encode_reference
from .native import verify_runtime
from .session import durable_json
from ..causal_v4.renderer import sha,inspect_audio
from ..causal_v4.runtime import PROFILE

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/content_events';d.mkdir(exist_ok=True)
 engine=pathlib.Path('/tmp/mari-native-content');build=json.loads((engine/'MARI_CONTENT_BUILD_RECEIPT.json').read_text());verify_runtime(r)
 if sha(engine/'qwen_tts')!=build['binary_sha256'] or sha(engine/'qwen_tts.c')!=build['source_sha256']:raise ValueError('diagnostic engine changed')
 profile=r/'recovered/production_release/production/MARI_VOICE_V1_PROFILE.bin'
 if sha(profile)!=PROFILE:raise ValueError('selected profile changed')
 durable_json(d/'BUILD_PROVENANCE.json',build)
 source=json.loads((r/'continuation/backchannel_reference/HUMAN_BACKCHANNELS.json').read_text())
 cases=source['examples'][::2] # first fixed example from each independently annotated channel
 durable_json(d/'PREDECLARED_SCOPE.json',{'selection':'first already acquired annotated event per channel; remaining examples held out',
  'source_license':source['license'],'attribution':source['attribution'],'donor_role':'acoustic codebook transfer diagnostic only; no Mari identity selection',
  'gate':'ordinary route byte parity; exact first-codebook delivery; event class survives; identity requires separately qualified short-event evidence',
  'not_assumed':['codebook0 contains semantics only','residual prediction alone enforces selected speaker','short-event ECAPA matches full-sentence threshold'],
  'full_completion':False})
 rows=[]
 def run(name,text,codes=None):
  out=d/(name+'.wav')
  if out.exists():raise FileExistsError(out)
  cmd=[str(engine/'qwen_tts'),'-d',str(r/'models/qwen-custom'),'--load-voice',str(profile),'--xvector-only','-l','English','--text',text,
   '--seed','88000','--temperature','.42','--top-k','40','--top-p','.95','--rep-penalty','1.05','-j4','-o',str(out),'--no-compose']
  env={k:v for k,v in os.environ.items() if not k.startswith(('MARI_','QWEN_'))}
  if codes:
   env.update(MARI_EMPTY_LEXICAL_EVENT='1',MARI_EVENT_CONTENT_CODES=str(codes));cmd+=['--max-tokens','64','--eos-strategy','off']
  p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=900);log=p.stdout+p.stderr;out.with_suffix('.log').write_text(log)
  row={'id':name,'command':cmd,'returncode':p.returncode,'log_sha256':sha(out.with_suffix('.log')),'text':text,'full_completion':False}
  if p.returncode:raise RuntimeError(log[-2500:])
  if out.exists():row['audio']=inspect_audio(out)[2]
  if codes:
   expected=np.loadtxt(codes,dtype=int,ndmin=1).tolist();got=[int(v) for v in re.findall(r'MARI_CONTENT_FRAME frame=\d+ code0=(\d+) native_residual=true',log)]
   row.update(delivered_codebook0=got,exact_input_delivery=got==expected,native_residual_audit='codebooks1to15=native' in log,
    input_codebook0_sha256=sha(codes),no_instructions='(instruct=0,' in log)
   if not row['exact_input_delivery'] or not row['native_residual_audit'] or not row['no_instructions']:raise ValueError('event route audit failed')
  else:
   row['ordinary_parity']=out.exists() and sha(out)=='063a844698d830af35e868d8daba0f8d228410eed786c60410eed9698a05ba10'
   if not row['ordinary_parity']:raise ValueError('ordinary route drift')
  rows.append(row);durable_json(d/'RESULTS.json',rows);print(name,row.get('ordinary_parity'),row.get('audio',{}).get('duration_s'),flush=True)
 run('ordinary_parity','I understand what happened. Give me a moment to check the sequence.')
 for case in cases:
  original=pathlib.Path(case['path'])
  if sha(original)!=case['sha256']:raise ValueError('AMI source changed')
  y,sr=sf.read(original)
  if sr!=16000:raise ValueError('AMI source clock changed')
  wav=d/(case['id']+'.source24k.wav');sf.write(wav,resample_poly(y,3,2),24000,subtype='PCM_16')
  codes=d/(case['id']+'.codes');encode_reference(r,wav,case['annotation'],codes)
  allcodes=np.loadtxt(codes,dtype=int,ndmin=2);first=d/(case['id']+'.code0');np.savetxt(first,allcodes[:,0],fmt='%d')
  durable_json(d/(case['id']+'.source.json'),{'source':case,'resampled_audio_sha256':sha(wav),'codes_sha256':sha(codes),'code0_sha256':sha(first),
   'role':'donor event acoustic diagnostic; does not alter original evaluator-only provenance or select donor identity'})
  run(case['id'],'',first)
if __name__=='__main__':main()
