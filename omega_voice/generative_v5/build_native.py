"""Build exact upstream revision plus the tracked Mari native integration patch."""
import argparse,hashlib,json,pathlib,shutil,subprocess
REV='e391ec5467b0218eeb175f4888ad65b259d1e7c7'
def verify_source(engine):
 e=pathlib.Path(engine);src=pathlib.Path(__file__).parent
 if subprocess.check_output(['git','-C',str(e),'rev-parse','HEAD'],text=True).strip()!=REV:raise RuntimeError('wrong engine revision')
 actual=subprocess.check_output(['git','-C',str(e),'diff','HEAD'])
 if actual!=(src/'engine.patch').read_bytes():raise RuntimeError('engine modifications differ from recorded patch')
 if (e/'mari_native_controls.h').read_bytes()!=(src/'native_controls.h').read_bytes():raise RuntimeError('unrecorded native control change')
 return True

def main():
 import scipy_openblas32 as blas
 ap=argparse.ArgumentParser();ap.add_argument('engine');a=ap.parse_args();e=pathlib.Path(a.engine).resolve();src=pathlib.Path(__file__).parent
 if subprocess.check_output(['git','-C',str(e),'rev-parse','HEAD'],text=True).strip()!=REV:raise RuntimeError('wrong engine revision')
 patch=src/'engine.patch'
 if not (e/'mari_native_controls.h').exists():
  subprocess.run(['git','-C',str(e),'apply','--check',str(patch)],check=True)
  subprocess.run(['git','-C',str(e),'apply',str(patch)],check=True)
 shutil.copyfile(src/'native_controls.h',e/'mari_native_controls.h')
 verify_source(e)
 flags='-I'+blas.get_include_dir()+' -Dcblas_sgemm=scipy_cblas_sgemm -Dopenblas_set_num_threads=scipy_openblas_set_num_threads -Dopenblas_get_num_threads=scipy_openblas_get_num_threads'
 lib=blas.get_lib_dir();cmd=['make','-C',str(e),'blas','SIMD=portable','EXTRA_CFLAGS='+flags,'LDLIBS=-lm -lpthread -L'+lib+' -Wl,-rpath,'+lib+' -lscipy_openblas']
 subprocess.run(cmd,check=True)
 receipt={'engine_base':REV,'patch_sha256':hashlib.sha256(patch.read_bytes()).hexdigest(),'native_controls_sha256':hashlib.sha256((src/'native_controls.h').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256((e/'qwen_tts').read_bytes()).hexdigest(),'command':cmd,'scope':'CPU single-request diagnostic. Other backends and serving are not qualified.'}
 receipt['source_hashes']={p:hashlib.sha256((e/p).read_bytes()).hexdigest() for p in ['qwen_tts.c','qwen_tts_talker.c','mari_native_controls.h']}
 (e/'MARI_BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
 # Separate calibration codec utility; no speaker/profile creation and no TTS fallback.
 objects=[str(p) for p in sorted(e.glob('*.o')) if p.name!='main.o']
 codec_cmd=['cc','-O2','-I'+str(e),str(src/'codec_encode.c'),*objects,str(e/'vendor/lz4.o'),str(e/'third_party/ingot/libingot.a'),'-o',str(e/'mari_codec_encode'),'-lm','-lpthread','-L'+lib,'-Wl,-rpath,'+lib,'-lscipy_openblas']
 subprocess.run(codec_cmd,check=True)
 (e/'MARI_CODEC_BUILD_RECEIPT.json').write_text(json.dumps({'command':codec_cmd,'source_sha256':hashlib.sha256((src/'codec_encode.c').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256((e/'mari_codec_encode').read_bytes()).hexdigest(),'engine_base':REV},indent=2)+'\n')
if __name__=='__main__':main()
