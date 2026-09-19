"""Isolated acoustic references for native actuator identification.

PSOLA derivatives are calibration stimuli only. Final speech must be generated
freely by the frozen Mari carrier with the resulting numeric actuator.
"""
import argparse,json,os,pathlib,struct,subprocess,tempfile
import numpy as np
import parselmouth
from parselmouth.praat import call
from .native import render,read_sequence,verify_runtime
from .acoustics import measure
from ..causal_v4.renderer import sha,inspect_audio
from ..causal_v4.runtime import PROFILE

def contour_reference(source,output,sign):
 sound=parselmouth.Sound(str(source));m=measure(source);end=m['voiced_end_s'];start=max(0,end-.55)
 manipulation=call(sound,'To Manipulation',.01,70,500);tier=call(manipulation,'Extract pitch tier')
 points=[]
 for i in range(1,call(tier,'Get number of points')+1):
  t=call(tier,'Get time from index',i);hz=call(tier,'Get value at index',i)
  u=np.clip((t-start)/(end-start),0,1)
  # Zero mean over the target interval, bounded -1..+3 semitones. No time warp.
  semitones=sign*(9*u*u-6*u) if t>=start else 0
  points.append((t,hz*2**(semitones/12)))
 call(tier,'Remove points between',sound.xmin,sound.xmax)
 for t,hz in points:call(tier,'Add point',t,hz)
 call([tier,manipulation],'Replace pitch tier')
 out=call(manipulation,'Get resynthesis (overlap-add)');out.save(str(output),'WAV')
 _,_,info=inspect_audio(output)
 return {'source_sha256':sha(source),'output':info,'target_window_s':[start,end],'contour_sign':sign,'method':'Praat PSOLA, paired zero-mean local contour derivative, identical duration','scope':'calibration reference, never selected carrier'}

def encode_reference(root,audio,text,codes_out):
 root=pathlib.Path(root);engine=root/'continuation/engine';model=root/'models/qwen-custom';p=root/'recovered/production_release/production/MARI_VOICE_V1_PROFILE.bin'
 lock=verify_runtime(root)
 if sha(p)!=PROFILE:raise ValueError('identity profile mismatch')
 env={k:v for k,v in os.environ.items() if not k.startswith(('MARI_','QWEN_'))}
 receipt=json.loads((engine/'MARI_CODEC_BUILD_RECEIPT.json').read_text())
 if sha(engine/'mari_codec_encode')!=receipt['binary_sha256']:raise ValueError('codec binary changed')
 if sha(pathlib.Path(__file__).with_name('codec_encode.c'))!=receipt['source_sha256']:raise ValueError('codec source changed')
 cmd=[str(engine/'mari_codec_encode'),str(model),str(audio),str(codes_out)]
 proc=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=900)
 pathlib.Path(codes_out).with_suffix('.encode.log').write_text(proc.stdout+proc.stderr)
 if proc.returncode or 'MARI_CODEC_EXTRACT' not in proc.stderr:raise RuntimeError('reference encoder failure: '+proc.stderr[-1500:])
 codes=np.loadtxt(codes_out,dtype=np.int32,ndmin=2)
 if codes.shape[1]!=16 or not 0<len(codes)<=8192 or (codes<0).any() or (codes>2047).any():raise ValueError('invalid codec output')
 record={'audio_sha256':sha(audio),'text':text,'frames':len(codes),'codes_sha256':sha(codes_out),'encoder_binary_sha256':receipt['binary_sha256'],'scope':'diagnostic codec references only; does not extract or alter speaker profile'}
 pathlib.Path(codes_out).with_suffix('.encode.json').write_text(json.dumps(record,indent=2)+'\n');return record

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();root=pathlib.Path(a.root);d=root/'continuation/isolated_calibration';d.mkdir(exist_ok=True)
 texts=['The package is ready.','You left it by the window.','The meeting starts at seven.','That is the correct address.']
 rows=[];deltas=[];means=[]
 for i,text in enumerate(texts):
  source=root/f'continuation/calibration/pair_{i}_statement.wav';pair={}
  for label,sign in [('fall',-1),('rise',1)]:
   p=d/f'{i}_{label}';ref=p.with_suffix('.reference.wav');codes=p.with_suffix('.codes')
   record=contour_reference(source,ref,sign);record['acoustics']=measure(ref)
   if not codes.exists():encode_reference(root,ref,text,codes)
   replay=p.with_suffix('.replay.wav')
   if not replay.exists():render(root,text,replay,90500+i,capture=True,teacher_codes=codes)
   seq=read_sequence(replay.with_suffix('.qseq'));start,end=record['target_window_s']
   # Same text, waveform timing and identity; compare exactly corresponding frames.
   window=[max(0,int(start/.08)),min(len(seq),int(end/.08)+1)]
   pair[label]={'reference':record,'replay_acoustics':measure(replay),'window':window,'activation':seq[window[0]:window[1]].mean(0)}
  contrast=pair['fall']['replay_acoustics']['final_slope_semitones_per_s']-pair['rise']['replay_acoustics']['final_slope_semitones_per_s']
  if contrast < -2:
   deltas.append(pair['fall']['activation']-pair['rise']['activation']);means.append((pair['fall']['activation']+pair['rise']['activation'])/2)
  row={'pair':i,'replayed_fall_minus_rise':contrast,'accepted':contrast < -2,**{k:{n:v for n,v in item.items() if n!='activation'} for k,item in pair.items()}}
  rows.append(row);(d/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n');print('isolated contrast',i,contrast,flush=True)
 if len(deltas)<3:raise RuntimeError('insufficient verified isolated contrasts')
 v=np.mean(deltas,0);n=np.mean(means,0);v-=v.mean(1,keepdims=True);v-=n*((v*n).sum(1)/(n*n).sum(1))[:,None]
 v[:21]=0;v[26:]=0
 np.savez(d/'isolated_finality_bank.npz',directions=v[None],paired_deltas=np.stack(deltas))
 print('ISOLATED_BANK_READY',np.linalg.norm(v,axis=1).tolist(),flush=True)
if __name__=='__main__':main()
