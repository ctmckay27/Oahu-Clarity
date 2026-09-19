"""Build pinned local audio critic; this is never a Mari renderer."""
import argparse,hashlib,json,pathlib,shutil,subprocess
REV='64d092c60db4b4ee45768476bd752f03fdcc98ea'
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('engine');p.add_argument('cmake');p.add_argument('receipt');a=p.parse_args()
 e=pathlib.Path(a.engine).resolve(); own=pathlib.Path(__file__).parent
 assert subprocess.check_output(['git','-C',str(e),'rev-parse','HEAD'],text=True).strip()==REV
 target=e/'tools/omni/omni.cpp'
 s=subprocess.check_output(['git','-C',str(e),'show','HEAD:tools/omni/omni.cpp'],text=True)
 # Scope this patch to synchronous user-audio input. Propagate each failed
 # eval, never let a skipped waveform be judged as if it had been supplied.
 start=s.index('            if (aud_fname.length() > 0) {',s.index('bool stream_prefill('))
 end=s.index('\n        }\n        else {',start)
 chunk=s[start:end]
 old='''                    eval_string(ctx_omni, ctx_omni->params, "<|audio_start|>", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                    prefill_with_emb(ctx_omni, ctx_omni->params, embeds->embed, embeds->n_pos, ctx_omni->params->n_batch, &ctx_omni->n_past);
                    eval_string(ctx_omni, ctx_omni->params, "<|audio_end|>", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                    omni_embed_free(embeds);'''
 new='''                    bool delivered = eval_string(ctx_omni, ctx_omni->params, "<|audio_start|>", ctx_omni->params->n_batch, &ctx_omni->n_past, false)
                        && prefill_with_emb(ctx_omni, ctx_omni->params, embeds->embed, embeds->n_pos, ctx_omni->params->n_batch, &ctx_omni->n_past)
                        && eval_string(ctx_omni, ctx_omni->params, "<|audio_end|>", ctx_omni->params->n_batch, &ctx_omni->n_past, false);
                    omni_embed_free(embeds);
                    if (!delivered) return false;'''
 assert chunk.count(old)==1
 chunk=chunk.replace(old,new).replace('LOG_WRN("%s: audio encoding failed, skipping audio for this frame\\n", __func__);','LOG_ERR("MARI_CRITIC: audio encoding failed; no judgment\\n");\n                    return false;')
 target.write_text(s[:start]+chunk+s[end:])
 shutil.copyfile(own/'minicpm_critic.cpp',e/'tools/omni/mari-audio-critic.cpp')
 shutil.copyfile(own/'minicpm_scene.cpp',e/'tools/omni/mari-scene-compiler.cpp')
 cm=e/'tools/omni/CMakeLists.txt'
 base=subprocess.check_output(['git','-C',str(e),'show','HEAD:tools/omni/CMakeLists.txt'],text=True)
 cm.write_text(base+'\nadd_executable(mari-audio-critic mari-audio-critic.cpp)\ntarget_link_libraries(mari-audio-critic PRIVATE llama-common omni Threads::Threads)\ntarget_compile_features(mari-audio-critic PRIVATE cxx_std_17)\nadd_executable(mari-scene-compiler mari-scene-compiler.cpp)\ntarget_link_libraries(mari-scene-compiler PRIVATE llama-common llama Threads::Threads)\ntarget_compile_features(mari-scene-compiler PRIVATE cxx_std_17)\n')
 commands=[[a.cmake,'-S',str(e),'-B',str(e/'build'),'-DCMAKE_BUILD_TYPE=Release','-DGGML_CUDA=OFF','-DGGML_METAL=OFF','-DLLAMA_CURL=OFF','-DLLAMA_BUILD_TESTS=OFF'],[a.cmake,'--build',str(e/'build'),'--target','mari-audio-critic','mari-scene-compiler','-j4']]
 for cmd in commands:subprocess.run(cmd,check=True)
 receipt={'upstream_repository':'https://github.com/tc-mb/llama.cpp-omni','upstream_revision':REV,'purpose':'unqualified audio critic; no Mari speech generation','commands':commands,'source_sha256':sha(own/'minicpm_critic.cpp'),'builder_sha256':sha(__file__),'patched_omni_sha256':sha(target),'binary_sha256':sha(e/'build/bin/mari-audio-critic')}
 receipt['scene_source_sha256']=sha(own/'minicpm_scene.cpp');receipt['scene_binary_sha256']=sha(e/'build/bin/mari-scene-compiler')
 pathlib.Path(a.receipt).write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
