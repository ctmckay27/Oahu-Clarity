// Audio-input diagnostic critic. No speech generation, voice reference or fallback.
#include "common.h"
#include "omni.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <filesystem>

int main(int argc, char **argv) {
    if (argc != 6) { std::cerr << "model audio_encoder input.wav prompt.txt output.txt\n"; return 2; }
    for (int i=1; i<5; ++i) if (!std::filesystem::is_regular_file(argv[i])) return 3;
    if (std::filesystem::exists(argv[5])) return 4;
    std::ifstream pf(argv[4]);
    std::string prompt((std::istreambuf_iterator<char>(pf)), {});
    if (prompt.empty() || prompt.size()>12000 || prompt.find("<|")!=std::string::npos) return 5;
    common_params params;
    params.model.path=argv[1]; params.apm_model=argv[2];
    params.n_ctx=4096; params.n_gpu_layers=0; params.n_batch=512;
    params.cpuparams.n_threads=4; params.cpuparams_batch.n_threads=4;
    params.n_predict=160; params.sampling.seed=88000; params.sampling.temp=0;
    common_init();
    auto *ctx=omni_init(&params,1,false,"",0,"cpu",false);
    if (!ctx || !ctx->ctx_audio || !ctx->ctx_llama || ctx->use_tts) return 6;
    ctx->async=false; ctx->ref_audio_path="";
    ctx->audio_voice_clone_prompt="<|im_start|>system\nYou analyze only the supplied audio. Report uncertainty rather than inventing sounds.";
    ctx->audio_assistant_prompt="<|im_end|>\n<|im_start|>user\n"+prompt+"\n";
    if (!stream_prefill(ctx,"","",0)) return 7;
    int before=ctx->n_past;
    if (!stream_prefill(ctx,argv[3],"",1) || ctx->n_past<=before+2) return 8;
    std::cerr << "MARI_CRITIC_AUDIO_PREFILL positions=" << ctx->n_past-before << " tts=false reference=false\n";
    if (!stream_decode(ctx,"",0)) return 9;
    std::string result;
    { std::lock_guard<std::mutex> guard(ctx->text_mtx);
      for (const auto &piece:ctx->text_queue) if(piece!="__END_OF_TURN__") result+=piece; }
    if(result.empty()) return 10;
    std::ofstream out(argv[5]); out<<result<<"\n"; out.close();
    omni_free(ctx);
    return out ? 0 : 11;
}
