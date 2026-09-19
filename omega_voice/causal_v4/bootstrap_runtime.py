"""Reproduce the tested local CPU engine without system package installation.

Downloads exact public model/engine revisions. No user data are uploaded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ENGINE_REV="e391ec5467b0218eeb175f4888ad65b259d1e7c7"
MODEL_REV="0c0e3051f131929182e2c023b9537f8b1c68adfe"

def main():
    import scipy_openblas32 as blas
    from huggingface_hub import snapshot_download
    ap=argparse.ArgumentParser();ap.add_argument("--directory",required=True);a=ap.parse_args()
    root=Path(a.directory).resolve();root.mkdir(parents=True,exist_ok=True)
    engine=root/"qwen3-tts"
    if not engine.exists():subprocess.run(["git","clone","https://github.com/gabriele-mastrapasqua/qwen3-tts.git",str(engine)],check=True)
    dirty=subprocess.check_output(["git","-C",str(engine),"status","--porcelain","--untracked-files=no"],text=True)
    if dirty.strip():raise RuntimeError("refuse to overwrite modified engine source")
    subprocess.run(["git","-C",str(engine),"fetch","origin",ENGINE_REV],check=True)
    subprocess.run(["git","-C",str(engine),"checkout","--detach",ENGINE_REV],check=True)
    flags="-I"+blas.get_include_dir()+" -Dcblas_sgemm=scipy_cblas_sgemm -Dopenblas_set_num_threads=scipy_openblas_set_num_threads -Dopenblas_get_num_threads=scipy_openblas_get_num_threads"
    lib=blas.get_lib_dir()
    cmd=["make","-C",str(engine),"blas","SIMD=portable","EXTRA_CFLAGS="+flags,
         "LDLIBS=-lm -lpthread -L"+lib+" -Wl,-rpath,"+lib+" -lscipy_openblas"]
    subprocess.run(cmd,check=True)
    subprocess.run([str(engine/"qwen_tts"),"--self-test"],check=True)
    model=root/"qwen-custom"
    snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",revision=MODEL_REV,local_dir=model,
                      allow_patterns=["*.json","*.safetensors","merges.txt","speech_tokenizer/*"])
    files={}
    for p in sorted(model.rglob("*")):
        if p.is_file() and ".cache" not in p.parts and p.name!="MARI_PINNED_MODEL_RECEIPT.json":
            h=hashlib.sha256()
            with p.open("rb") as f:
                for b in iter(lambda:f.read(1048576),b""):h.update(b)
            files[str(p.relative_to(model))]=h.hexdigest()
    (model/"MARI_PINNED_MODEL_RECEIPT.json").write_text(json.dumps({"model":"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice","revision":MODEL_REV,"files":files,"source":"exact-revision snapshot_download"},indent=2))
    (root/"BUILD_RECEIPT.json").write_text(json.dumps({"engine_revision":ENGINE_REV,"command":cmd,"model_revision":MODEL_REV},indent=2))

if __name__=="__main__":main()
