"""Test native acoustic continuation with no lexical target or acting prose."""
import argparse,pathlib,json,subprocess,os
import soundfile as sf
from .native import verify_runtime
from .session import durable_json
from ..causal_v4.renderer import sha,inspect_audio
from ..causal_v4.runtime import ANCHOR,PROFILE

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/native_events';d.mkdir(exist_ok=True)
 lock=verify_runtime(r);engine=pathlib.Path('/tmp/mari-native-event');build=json.loads((engine/'MARI_EVENT_BUILD_RECEIPT.json').read_text())
 if sha(engine/'qwen_tts')!=build['binary_sha256'] or sha(engine/'qwen_tts.c')!=build['source_sha256']:raise ValueError('isolated renderer changed')
 assets=r/'recovered/production_release/production';profile=assets/'MARI_VOICE_V1_PROFILE.bin'
 if sha(profile)!=PROFILE or sha(assets/'MARI_VOICE_V1_ANCHOR.wav')!=ANCHOR:raise ValueError('identity changed')
 source=json.loads((r/'continuation/nonlexical_world/SOURCE.json').read_text());word=source['word'];x,sr=sf.read(source['source_audio'],dtype='int16')
 if sha(source['source_audio'])!=source['source_audio_sha256']:raise ValueError('reference source changed')
 reference=d/'measured_word_context.wav';sf.write(reference,x[round(word['start']*sr):round(word['end']*sr)],sr,subtype='PCM_16')
 cases=[('ordinary_parity','I understand what happened. Give me a moment to check the sequence.',None,None),
  ('native_event_measured_context','',reference,'moment'),
  ('native_event_nasal_context','',r/'continuation/nonlexical_world/receipt.wav','')]
 rows=[]
 for name,text,ref,reftext in cases:
  out=d/(name+'.wav')
  if out.exists():raise FileExistsError(out)
  cmd=[str(engine/'qwen_tts'),'-d',str(r/'models/qwen-custom'),'--load-voice',str(profile),'--xvector-only','-l','English','--text',text,
   '--seed','88000','--temperature','.42','--top-k','40','--top-p','.95','--rep-penalty','1.05','-j4','-o',str(out)]
  env={k:v for k,v in os.environ.items() if not k.startswith(('QWEN_','MARI_'))}
  if ref:
   cmd += ['--emo-ref',str(ref),'--emo-ref-text',reftext,'--no-compose','--max-tokens','16'];env['MARI_EMPTY_LEXICAL_EVENT']='1'
  run=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=900);log=run.stdout+run.stderr;out.with_suffix('.log').write_text(log)
  row={'id':name,'command':cmd,'returncode':run.returncode,'log_sha256':sha(out.with_suffix('.log')),
   'binary_sha256':build['binary_sha256'],'selected_profile_sha256':PROFILE,'reference_sha256':sha(ref) if ref else None,
   'input_spoken_text':text,'acting_prose':False,'full_completion':False}
  if out.exists():
   try:row['audio']=inspect_audio(out)[2]
   except Exception as exc:row['audio_error']=str(exc)
  if not ref:
   row['ordinary_parity']=out.exists() and sha(out)=='063a844698d830af35e868d8daba0f8d228410eed786c60410eed9698a05ba10'
   if not row['ordinary_parity']:raise ValueError('isolated change affected ordinary route')
  else:row['native_route_audit']={'empty_lexical_flag':'MARI_EMPTY_LEXICAL_EVENT explicit' in log,'zero_text_content':'text_content: 0)' in log,
    'acoustic_prefix':'Emotion-by-example:' in log,'no_instructions':'(instruct=0,' in log,'natural_eos':'EOS at frame' in log}
  rows.append(row);durable_json(d/'RESULTS.json',rows);print(name,run.returncode,row.get('ordinary_parity'),row.get('audio',{}).get('duration_s'),flush=True)
 durable_json(d/'BUILD_PROVENANCE.json',build)
if __name__=='__main__':main()
