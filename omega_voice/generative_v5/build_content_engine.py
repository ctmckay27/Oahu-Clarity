"""Isolated first-codebook event transfer, with semantic/identity scope unqualified.

No ordinary speech route changes. An observed codebook-0 sequence is an acoustic
constraint, NOT asserted to be a disentangled semantic representation. The
frozen selected-speaker native model predicts codebooks 1..15. Tests decide
whether donor identity leaks or the event survives.
"""
import argparse,pathlib,subprocess,json,sys
from .build_event_engine import sha

def replace_once(s,old,new):
 if s.count(old)!=1:raise ValueError('source insertion mismatch: '+old[:70])
 return s.replace(old,new)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('destination');a=ap.parse_args();dst=pathlib.Path(a.destination)
 subprocess.run([sys.executable,'-m','omega_voice.generative_v5.build_event_engine',a.root,str(dst)],check=True)
 original=json.loads((dst/'MARI_EVENT_BUILD_RECEIPT.json').read_text());source=dst/'qwen_tts.c';s=source.read_text()
 s=replace_once(s,'    const char *mari_event_env = getenv("MARI_EMPTY_LEXICAL_EVENT");','''    const char *mari_content_path = getenv("MARI_EVENT_CONTENT_CODES");
    const char *mari_event_env = getenv("MARI_EMPTY_LEXICAL_EVENT");
    int mari_content_codes[64], mari_content_frames = 0;
    if (mari_content_path) {
        if (!mari_event_env || strcmp(mari_event_env,"1") || !mari_content_path[0] ||
            getenv("QWEN_TF_CODES") || getenv("MARI_TRAJECTORY")) {
            fprintf(stderr,"Error: incompatible event-content route\\n"); return -1;
        }
        FILE *mf = fopen(mari_content_path,"r");
        if (!mf) { fprintf(stderr,"Error: missing event-content codes\\n"); return -1; }
        int v, result;
        while ((result=fscanf(mf,"%d",&v)) == 1) {
            if (v<0 || v>2047 || mari_content_frames>=64) {
                fclose(mf); fprintf(stderr,"Error: invalid event-content codes\\n"); return -1;
            }
            mari_content_codes[mari_content_frames++]=v;
        }
        fclose(mf);
        if (result!=EOF || !mari_content_frames || mari_content_frames>ctx->max_tokens) {
            fprintf(stderr,"Error: malformed/empty/over-budget event-content codes\\n"); return -1;
        }
        fprintf(stderr,"MARI_CONTENT_EVENT frames=%d codebook0=observed codebooks1to15=native semantic_and_identity_transfer=unqualified\\n",mari_content_frames);
    }''')
 s=replace_once(s,'!ctx->voice_clone || !ctx->speaker_embedding || !ctx->emo_ref_path ||\n        !ctx->emo_ref_text || (ctx->instruct && ctx->instruct[0]) ||',
  '!ctx->voice_clone || !ctx->speaker_embedding ||\n        (!mari_content_path && (!ctx->emo_ref_path || !ctx->emo_ref_text)) || (ctx->instruct && ctx->instruct[0]) ||')
 s=replace_once(s,'    for (int frame = 0; frame < max_frames; frame++) {\n        qwen_census_frame();','''    for (int frame = 0; frame < max_frames; frame++) {
        if (mari_content_path && frame>=mari_content_frames) break;
        qwen_census_frame();''')
 s=replace_once(s,'        if (code0_fp) fprintf(code0_fp, "%d\\n", code0);','''        if (mari_content_path) {
            code0=mari_content_codes[frame];
            fprintf(stderr,"MARI_CONTENT_FRAME frame=%d code0=%d native_residual=true\\n",frame,code0);
        }
        if (code0_fp) fprintf(code0_fp, "%d\\n", code0);''')
 source.write_text(s);subprocess.run(original['command'],check=True)
 receipt={'role':'isolated codebook0 event transfer diagnostic; not production or semantic/identity admission',
  'base_build':original,'source_sha256':sha(source),'binary_sha256':sha(dst/'qwen_tts'),'implementation_sha256':sha(__file__),
  'full_completion':False,'native_residual_codebooks':list(range(1,16))}
 (dst/'MARI_CONTENT_BUILD_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
 with (dst/'MARI_CONTENT_COMBINED.patch').open('w') as f:subprocess.run(['git','-C',str(dst),'diff','HEAD','--','qwen_tts.c'],stdout=f,check=True)
 print(receipt['binary_sha256'],flush=True)
if __name__=='__main__':main()
