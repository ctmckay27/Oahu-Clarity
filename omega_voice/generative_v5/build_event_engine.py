"""Isolated native empty-lexical-target diagnostic; production engine untouched."""
import pathlib,argparse,shutil,subprocess,json,hashlib
from .native import verify_runtime

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('destination');a=ap.parse_args();r=pathlib.Path(a.root);dst=pathlib.Path(a.destination)
 if dst.exists():raise FileExistsError(dst)
 lock=verify_runtime(r);base=r/'continuation/engine';shutil.copytree(base,dst)
 source=dst/'qwen_tts.c';s=source.read_text()
 needle='    if (!text_tokens || text_token_len == 0) {'
 if s.count(needle)!=1:raise ValueError('native source insertion mismatch')
 replacement='''    const char *mari_event_env = getenv("MARI_EMPTY_LEXICAL_EVENT");
    int mari_empty_event = mari_event_env != NULL;
    if (mari_empty_event && (strcmp(mari_event_env, "1") != 0 || text[0] != 0 ||
        !ctx->voice_clone || !ctx->speaker_embedding || !ctx->emo_ref_path ||
        !ctx->emo_ref_text || (ctx->instruct && ctx->instruct[0]) ||
        getenv("MARI_TEXT_RELEASE") || ctx->stream_layout_active)) {
        fprintf(stderr, "Error: invalid isolated nonlexical diagnostic request\\n");
        free(text_tokens); free(instruct_tokens); free(ref_text_tokens); return -1;
    }
    if (mari_empty_event) fprintf(stderr, "MARI_EMPTY_LEXICAL_EVENT explicit selected-profile acoustic continuation; semantic qualification pending\\n");
    if (!text_tokens || (text_token_len == 0 && !mari_empty_event)) {'''
 s=s.replace(needle,replacement)
 needle='if (ctx->emo_ref_path && ctx->emo_ref_text && ref_text_tokens && ref_text_token_len > 0) {'
 if s.count(needle)!=1:raise ValueError('native reference insertion mismatch')
 s=s.replace(needle,'if (ctx->emo_ref_path && ctx->emo_ref_text && (mari_empty_event || (ref_text_tokens && ref_text_token_len > 0))) {')
 source.write_text(s)
 # The copied object files come from the verified base. The changed C source
 # rebuilds its object; exact unchanged dependencies and build command persist.
 import scipy_openblas32 as blas
 lib=blas.get_lib_dir();flags='-I'+blas.get_include_dir()+' -Dcblas_sgemm=scipy_cblas_sgemm -Dopenblas_set_num_threads=scipy_openblas_set_num_threads -Dopenblas_get_num_threads=scipy_openblas_get_num_threads'
 cmd=['make','-C',str(dst),'blas','SIMD=portable','EXTRA_CFLAGS='+flags,'LDLIBS=-lm -lpthread -L'+lib+' -Wl,-rpath,'+lib+' -lscipy_openblas']
 subprocess.run(cmd,check=True)
 receipt={'role':'isolated empty-target acoustic continuation diagnostic; no production selection','base_runtime':lock,
  'source_sha256':sha(source),'binary_sha256':sha(dst/'qwen_tts'),'implementation_sha256':sha(__file__),'command':cmd,
  'base_binary_unchanged':sha(base/'qwen_tts')==lock['build']['binary_sha256'],'full_completion':False}
 (dst/'MARI_EVENT_BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
 subprocess.run(['git','-C',str(dst),'diff','HEAD','--','qwen_tts.c'],stdout=(dst/'MARI_EVENT_COMBINED.patch').open('w'),check=True)
 print(receipt['binary_sha256'],flush=True)
if __name__=='__main__':main()
