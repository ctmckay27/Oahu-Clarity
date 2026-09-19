// Pinned local language compiler. Typed JSON is validated again by Python.
// No audio encoder, voice model, network or renderer instructions here.
#include "llama.h"
#include "common.h"
#include "sampling.h"
#include "json-schema-to-grammar.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <iterator>
static std::string read(const char *p) {
    std::ifstream f(p); return {(std::istreambuf_iterator<char>(f)), {}};
}
int main(int argc,char **argv) {
    if(argc!=5) return 2;
    for(int i=1;i<4;++i) if(!std::filesystem::is_regular_file(argv[i]))return 3;
    if(std::filesystem::exists(argv[4]))return 4;
    const std::string prompt=read(argv[2]);
    if(prompt.empty()||prompt.size()>24000)return 5;
    auto schema=nlohmann::ordered_json::parse(read(argv[3]));
    common_init(); ggml_backend_load_all();
    auto mp=llama_model_default_params();mp.n_gpu_layers=0;
    auto *model=llama_model_load_from_file(argv[1],mp);if(!model)return 6;
    auto cp=llama_context_default_params();cp.n_ctx=4096;cp.n_batch=512;cp.n_threads=4;cp.n_threads_batch=4;
    auto *ctx=llama_init_from_model(model,cp);if(!ctx)return 7;
    auto tokens=common_tokenize(ctx,prompt,false,true);
    if(tokens.size()+800>4096)return 8;
    for(size_t at=0;at<tokens.size();at+=512) {
        auto n=std::min(size_t(512),tokens.size()-at);
        if(llama_decode(ctx,llama_batch_get_one(tokens.data()+at,n)))return 9;
    }
    common_params_sampling sp;sp.seed=88000;sp.temp=0;
    sp.grammar={COMMON_GRAMMAR_TYPE_USER,json_schema_to_grammar(schema)};
    auto *sampler=common_sampler_init(model,sp);if(!sampler)return 10;
    auto *vocab=llama_model_get_vocab(model);std::string result;bool ended=false;int count=0;
    for(;count<800;++count) {
        auto token=common_sampler_sample(sampler,ctx,-1,true);
        common_sampler_accept(sampler,token,true);
        if(llama_vocab_is_eog(vocab,token)){ended=true;break;}
        result+=common_token_to_piece(ctx,token,true);
        if(llama_decode(ctx,llama_batch_get_one(&token,1)))return 11;
    }
    if(!ended)return 12; // truncated generation is never an admitted scene
    try { auto parsed=nlohmann::ordered_json::parse(result); if(!parsed.is_object())return 13; }
    catch(...) {return 14;}
    std::ofstream out(argv[4]);out<<result<<"\n";out.close();
    std::cerr<<"MARI_SCENE_TEXT_ONLY tokens="<<tokens.size()<<" generated="<<count<<" eos=true\n";
    common_sampler_free(sampler);llama_free(ctx);llama_model_free(model);llama_backend_free();
    return out?0:15;
}
