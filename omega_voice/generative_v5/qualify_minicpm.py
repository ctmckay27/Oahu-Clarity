"""Blinded, source-hashed audio-critic controls. Responses are evidence, not passes."""
import argparse,hashlib,json,pathlib,subprocess,time,math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('destination');a=ap.parse_args()
 root=pathlib.Path(a.root).resolve();out=pathlib.Path(a.destination).resolve();out.mkdir(parents=True,exist_ok=True)
 c=root/'continuation';engine=pathlib.Path('/tmp/mari-minicpm-engine');models=pathlib.Path('/tmp/mari-minicpm-gguf')
 receipt=json.loads((c/'minicpm_inspection/CRITIC_BUILD_RECEIPT.json').read_text());binary=engine/'build/bin/mari-audio-critic'
 assert sha(binary)==receipt['binary_sha256']
 for row in json.loads((c/'minicpm_inspection/MODEL_PROVENANCE.json').read_text())['files']:
  assert sha(models/row['path'])==row['sha256']
 validity='Describe only what is audible. Is there intelligible speech, unintelligible vocal sound, or silence? If speech is intelligible, give the exact words. Do not guess a language or invent a transcript.'
 rhythm='Describe the audible rhythm and phrasing of the voice. Is the timing flowing, read aloud, conspicuously acted, or mechanically interrupted? Give specific audible evidence in two sentences. If there is no speech, say so.'
 pitch='How does the vocal pitch move over the final word: upward, downward, or approximately level? Describe the sound, not the punctuation or meaning of the words.'
 jobs=[('carrier',c/'calibration/unchanged.wav',validity),('silence',c/'speechjudge_qualification/silence.wav',validity),('reversed',c/'speechjudge_qualification/reversed.wav',validity),('carrier_rhythm',c/'calibration/unchanged.wav',rhythm),('mechanical_gaps',c/'qwen_delivery_qualification/negative_disrupted.wav',rhythm),('retained_rejected',c/'retained_v2_negative/01_search_resolution.wav',rhythm),('human_annotated',c/'drame_inspection/source/data/realism_data/role-eval_v1_0001.wav',rhythm)]
 jobs += [('contour_'+str(i),c/'isolated_calibration'/name,pitch) for i,name in enumerate(['0_fall.reference.wav','0_rise.reference.wav','2_rise.reference.wav','2_fall.reference.wav'])]
 plan={'purpose':'qualification before any Mari performance verdict','blinding':'model receives numeric filenames, no state or answer labels','gates':{'input_validity':'all three known waveform types correctly identified without invented speech','rhythm':'distinguish inserted mechanical interruptions from human-annotated scene delivery; retained Carl-rejected acting cannot establish a pass','contour':'all four measured signs must agree before using this critic for pitch'},'jobs':[{'id':k,'source':str(p),'source_sha256':sha(p),'prompt':q}for k,p,q in jobs],'model_build':receipt}
 (out/'PREDECLARED.json').write_text(json.dumps(plan,indent=2)+'\n');results=[]
 for i,(key,src,prompt) in enumerate(jobs):
  x,sr=sf.read(src,dtype='float32');assert x.ndim==1 and np.all(np.isfinite(x))
  if sr!=16000:
   g=math.gcd(sr,16000);x=resample_poly(x,16000//g,sr//g)
  wav=out/f'{i:03}.wav';sf.write(wav,x,16000,subtype='PCM_16')
  task=out/f'{i:03}.prompt.txt';task.write_text(prompt);answer=out/f'{i:03}.answer.txt';log=out/f'{i:03}.log'
  if answer.exists():raise FileExistsError(answer)
  cmd=[str(binary),str(models/'MiniCPM-o-4_5-Q8_0.gguf'),str(models/'audio/MiniCPM-o-4_5-audio-F16.gguf'),str(wav),str(task),str(answer)]
  start=time.time()
  with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  row={'id':key,'returncode':r.returncode,'elapsed_seconds':time.time()-start,'input_sha256':sha(wav),'log_sha256':sha(log),'audio_delivery_audit':'MARI_CRITIC_AUDIO_PREFILL positions=' in log.read_text(errors='replace'),'answer':answer.read_text()if answer.exists()else None}
  results.append(row);(out/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(row),flush=True)
  if r.returncode or not row['audio_delivery_audit']:raise RuntimeError('critic execution failed; not a performance verdict')
if __name__=='__main__':main()
