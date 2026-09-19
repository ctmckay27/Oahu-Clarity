"""Numeric control packets for the native Talker. No textual style channel."""
from pathlib import Path
import hashlib,json,os,re,struct,subprocess,time
import numpy as np
from ..causal_v4.renderer import inspect_audio,sha
from ..causal_v4.runtime import ANCHOR,PROFILE

def read_sequence(path):
 with open(path,'rb') as f:
  head=f.read(12)
  if len(head)!=12:raise ValueError('truncated capture header')
  magic,l,d=struct.unpack('<3I',head)
  if (magic,l,d)!=(0x3151534d,29,2048):raise ValueError('wrong capture model or format')
  x=np.frombuffer(f.read(),dtype='<f4').copy()
 if not x.size or x.size%(l*d):raise ValueError('truncated activation capture')
 if not np.isfinite(x).all():raise ValueError('nonfinite activation capture')
 return x.reshape(-1,l,d)

def write_trajectory(path,banks,weights,onset=None):
 b=np.asarray(banks,dtype='<f4');w=np.asarray(weights,dtype='<f4')
 if b.ndim!=3 or w.ndim!=2 or b.shape[0]!=w.shape[1]:raise ValueError('trajectory shape mismatch')
 if not np.isfinite(b).all() or not np.isfinite(w).all() or not b.size or not w.size or np.abs(w).max()>2 or np.abs(b).max()>100:raise ValueError('invalid trajectory')
 if b.shape[1:]!=(29,2048) or not 0<len(b)<=16 or not 0<len(w)<=8192:raise ValueError('wrong model shape')
 o=None if onset is None else np.asarray(onset,dtype='<f4')
 if o is not None and (o.shape!=(len(b),) or not np.isfinite(o).all() or np.abs(o).max()>2):raise ValueError('invalid onset trajectory')
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('wb') as f:
  f.write(struct.pack('<5I',0x3152544d if o is None else 0x3252544d,*b.shape,len(w)));f.write(b.tobytes())
  if o is not None:f.write(o.tobytes())
  f.write(w.tobytes())
 return {'path':str(p),'sha256':sha(p),'banks':len(b),'frames':len(w),'layers':29,'dim':2048}

def verify_runtime(root):
 from .build_native import verify_source
 root=Path(root);engine=root/'continuation/engine';model=root/'models/qwen-custom'
 verify_source(engine)
 receipt=json.loads((engine/'MARI_BUILD_RECEIPT.json').read_text())
 if sha(engine/'qwen_tts')!=receipt['binary_sha256']:raise ValueError('binary mismatch')
 for rel,h in receipt['source_hashes'].items():
  if sha(engine/rel)!=h:raise ValueError('engine source changed: '+rel)
 prov=json.loads((model/'MARI_PINNED_MODEL_RECEIPT.json').read_text())
 if prov['revision']!='0c0e3051f131929182e2c023b9537f8b1c68adfe':raise ValueError('model revision mismatch')
 for rel,h in prov['files'].items():
  if sha(model/rel)!=h:raise ValueError('model bytes changed: '+rel)
 return {'build':receipt,'model':prov}

def render(root,text,out,seed=88000,trajectory=None,capture=False,teacher_codes=None,reference=None,incremental_text=False):
 root=Path(root);out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise FileExistsError(out)
 if not text.strip() or re.search(r'[\[\]<>]',text):raise ValueError('invalid spoken text')
 engine=root/'continuation/engine';model=root/'models/qwen-custom'
 production=root/'recovered/production_release/production'
 if sha(production/'MARI_VOICE_V1_PROFILE.bin')!=PROFILE or sha(production/'MARI_VOICE_V1_ANCHOR.wav')!=ANCHOR:raise ValueError('identity asset mismatch')
 lock=verify_runtime(root);receipt=lock['build']
 cmd=[str(engine/'qwen_tts'),'-d',str(model),'--load-voice',str(production/'MARI_VOICE_V1_PROFILE.bin'),'--xvector-only','-l','English','--text',text,'--seed',str(seed),'--temperature','.42','--top-k','40','--top-p','.95','--rep-penalty','1.05','-j4','-o',str(out)]
 reference_record=None
 if reference:
  # This route uses the selected speaker embedding unchanged and places only
  # an explicitly hash-bound diagnostic recording in the acoustic prefix.
  if teacher_codes:raise ValueError('reference conditioning and forced reconstruction are separate routes')
  if set(reference)!={'audio','text','sha256','source_carrier_sha256','role'}:raise ValueError('incomplete reference provenance')
  ref=Path(reference['audio']);_,refsr,refinfo=inspect_audio(ref)
  if sha(ref)!=reference['sha256'] or reference['source_carrier_sha256']!=ANCHOR:raise ValueError('reference lineage/hash mismatch')
  if refsr!=24000 or not reference['text'].strip() or re.search(r'[\[\]<>]',reference['text']):raise ValueError('invalid reference format or transcript')
  cmd.extend(['--emo-ref',str(ref),'--emo-ref-text',reference['text']])
  reference_record=dict(reference,audio_info=refinfo)
 env={k:v for k,v in os.environ.items() if not k.startswith(('QWEN_','MARI_'))}
 if not isinstance(incremental_text,bool):raise ValueError('incremental text flag must be boolean')
 if incremental_text:
  env['QWEN_TTS_STREAM_LAYOUT']='1';env['QWEN_DUMP_CODE0']=str(out.with_suffix('.code0.txt'))
 if trajectory:env['MARI_TRAJECTORY']=str(Path(trajectory).resolve())
 if teacher_codes:
  if trajectory:raise ValueError('calibration replay and free-generation control are separate routes')
  codes=np.loadtxt(teacher_codes,dtype=np.int32,ndmin=2)
  if codes.ndim!=2 or codes.shape[1]!=16 or not 1<=len(codes)<=8192 or (codes<0).any() or (codes>2047).any():raise ValueError('invalid teacher-forcing codes')
  env['QWEN_TF_CODES']=str(Path(teacher_codes).resolve())
 if capture:
  env['QWEN_ACT_MAP']=str(out.with_suffix('.qamp'));env['MARI_ACT_SEQUENCE']=str(out.with_suffix('.qseq'))
 start=time.monotonic();r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=900)
 out.with_suffix('.log').write_text(r.stdout+r.stderr)
 if r.returncode:raise RuntimeError('native renderer failure: '+r.stderr[-1500:])
 if trajectory and 'MARI_NATIVE_TRAJECTORY' not in r.stderr:raise RuntimeError('trajectory did not enter native renderer')
 if incremental_text and not re.search(r'stream_common=\d+, trailing_text=[1-9][0-9]*',r.stderr):raise RuntimeError('incremental text layout did not enter native renderer')
 if reference and ('Emotion-by-example:' not in r.stderr or not re.search(r'icl_codes=[1-9][0-9]*',r.stderr)):
  raise RuntimeError('reference conditioning did not enter native renderer; output is not admitted')
 if teacher_codes and f'teacher-forcing replay: {len(codes)} reference frames' not in r.stderr:raise RuntimeError('teacher forcing did not enter renderer')
 _,_,audio=inspect_audio(out)
 capture_info=None
 if capture:
  seq=read_sequence(out.with_suffix('.qseq'))
  expected=round(audio['duration_s']/.08)
  if len(seq)!=expected:raise RuntimeError('capture/frame count mismatch')
  capture_info={'frames':len(seq),'sha256':sha(out.with_suffix('.qseq'))}
 record={'role':'native generative diagnostic','command':cmd,'explicit_environment':{k:v for k,v in env.items() if k.startswith(('QWEN_','MARI_'))},'audio':audio,'text':text,'seed':seed,'elapsed_wall_s':time.monotonic()-start,'binary_sha256':receipt['binary_sha256'],'prose_instructions':False,'trajectory_sha256':sha(trajectory) if trajectory else None}
 record.update(runtime_lock=lock,capture=capture_info,reference=reference_record)
 if incremental_text:
  codes=out.with_suffix('.code0.txt')
  if not codes.is_file() or not codes.stat().st_size:raise RuntimeError('incremental token evidence absent')
  record.update(text_availability='one text token per generation frame; diagnostic qualification pending',code0_sha256=sha(codes))
 if teacher_codes:
  if round(audio['duration_s']/.08)!=len(codes):raise RuntimeError('teacher-forcing replay length mismatch')
  record.update(role='teacher-forced calibration reconstruction; not free generation',teacher_codes_sha256=sha(teacher_codes))
 out.with_suffix('.receipt.json').write_text(json.dumps(record,indent=2)+'\n')
 return record
