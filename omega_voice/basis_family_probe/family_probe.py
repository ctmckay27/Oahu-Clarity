"""Bounded reference-voice family experiment; no training or production promotion.

Run in an isolated environment with downloaded, hash-verified reference assets.
No cloud speech API, donor voice, automatic fallback or style prompt is used.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, random, subprocess, sys, time, traceback
from pathlib import Path

ANCHOR_HASH = '73bb894fc75e18dddf76e3327f3bf48e161b7dd8b30fbe4949c3ac7318c29567'
PROFILE_HASH = '9c49146ae2bd3c5a32c686ad1705fd912391af4a39db0d63e9dc353640d304fa'
QWEN_C_REV = 'e391ec5467b0218eeb175f4888ad65b259d1e7c7'
BASE_REV = 'fd4b254389122332181a7c3db7f27e918eec64e3'
CUSTOM_REV = '0c0e3051f131929182e2c023b9537f8b1c68adfe'
REF_TEXT = ('I understand the problem. Give me the facts in the order they happened, '
            'not the order they were explained later. Then we can decide what actually matters.')
TEXTS = [
    'The spare key is in the drawer.',
    'Yeah, leave it there for now.',
    'Was that before lunch or after?',
    'I thought you meant the smaller box.',
    'Give me a second. I had it a minute ago.',
    "It's upstairs. No, sorry, downstairs by the door.",
    'The appointment is at ten thirty on Thursday.',
    "I saw your message. I just haven't answered yet.",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def run(cmd: list[str], log: Path, timeout: int = 1800, cwd: Path | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as f:
        f.write('COMMAND ' + json.dumps([str(x) for x in cmd]) + '\n'); f.flush()
        cp = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, cwd=cwd)
    if cp.returncode:
        raise RuntimeError(f'command failed ({cp.returncode}); see {log}')


def assets(root: Path) -> tuple[Path, Path]:
    def get(name: str, expected: str) -> Path:
        good = [p for p in root.rglob(name) if digest(p) == expected]
        if not good:
            raise RuntimeError(f'No hash-verified {name}; no substitute reference permitted')
        return sorted(good)[0].resolve()
    return get('MARI_VOICE_V1_ANCHOR.wav', ANCHOR_HASH), get('MARI_VOICE_V1_PROFILE.bin', PROFILE_HASH)


def cases() -> list[dict]:
    rows = [{'id': f'U{i:02d}', 'text': text, 'seed': 172600 + i, 'replicate': 0}
            for i, text in enumerate(TEXTS, 1)]
    for i in (2, 6):
        rows.append({'id': f'U{i:02d}_r1', 'text': TEXTS[i-1], 'seed': 173600+i, 'replicate': 1})
    return rows


def snapshot(repo: str, destination: Path, receipt: Path, revision: str | None = None,
             allow: list[str] | None = None) -> Path:
    from huggingface_hub import HfApi, snapshot_download
    revision = revision or HfApi().model_info(repo).sha
    p = Path(snapshot_download(repo, revision=revision, local_dir=str(destination),
                               allow_patterns=allow, max_workers=4)).resolve()
    files = [{'path': str(f.relative_to(p)), 'bytes': f.stat().st_size, 'sha256': digest(f)}
             for f in sorted(p.rglob('*')) if f.is_file() and '.cache' not in f.parts]
    write(receipt, {'repo': repo, 'revision': revision, 'files': files})
    return p


def environment(out: Path) -> None:
    data = {'python': sys.version, 'platform': platform.platform(), 'cpus': os.cpu_count(),
            'run_id': os.getenv('GITHUB_RUN_ID'), 'run_attempt': os.getenv('GITHUB_RUN_ATTEMPT'),
            'source_commit': os.getenv('GITHUB_SHA'), 'cpu_info': Path('/proc/cpuinfo').read_text(),
            'mem_info': Path('/proc/meminfo').read_text(), 'threads': 4}
    write(out / 'RUNTIME.json', data)
    import importlib.metadata
    packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    (out / 'pip-freeze.txt').write_text('\n'.join(packages) + '\n')


def record_wave(out: Path, route: str, case: dict, started: float, extra: dict) -> None:
    import numpy as np
    import soundfile as sf
    p = out / route / (case['id'] + '.wav')
    y, sr = sf.read(p, always_2d=True)
    if not len(y) or not np.isfinite(y).all() or np.max(np.abs(y)) == 0:
        raise RuntimeError('Empty, silent or nonfinite waveform')
    write(p.with_suffix('.json'), {'status': 'RENDERED_NOT_PERCEPTUALLY_ACCEPTED',
          'route': route, **case, 'waveform_sha256': digest(p), 'sample_rate': sr,
          'channels': y.shape[1], 'duration_s': len(y)/sr,
          'peak': float(np.max(np.abs(y))), 'clipped_samples': int((np.abs(y) >= 1).sum()),
          'wall_s': time.monotonic()-started, 'source_anchor_sha256': ANCHOR_HASH,
          'reference_transcript': REF_TEXT, 'acting_instruction': None,
          'trajectory_consumed': False, 'training_performed': False, **extra})


def fail_case(out: Path, route: str, case: dict, exc: Exception) -> None:
    write(out / route / (case['id'] + '.failure.json'), {
        'status': 'FAILED_NO_SUBSTITUTE', 'route': route, **case,
        'error': repr(exc), 'traceback': traceback.format_exc()})


def native(anchor: Path, profile: Path, out: Path, work: Path) -> None:
    e = work / 'qwen-c'
    run(['git', 'clone', '--no-checkout', 'https://github.com/gabriele-mastrapasqua/qwen3-tts.git', str(e)], out/'clone.log')
    run(['git', '-C', str(e), 'checkout', '--detach', QWEN_C_REV], out/'checkout.log')
    run(['git', '-C', str(e), 'submodule', 'update', '--init', '--recursive'], out/'submodules.log')
    import scipy_openblas32 as blas
    inc, lib = blas.get_include_dir(), blas.get_lib_dir()
    flags = '-I'+inc+' -Dcblas_sgemm=scipy_cblas_sgemm -Dopenblas_set_num_threads=scipy_openblas_set_num_threads -Dopenblas_get_num_threads=scipy_openblas_get_num_threads'
    run(['make', '-C', str(e), 'blas', 'SIMD=portable', 'EXTRA_CFLAGS='+flags,
         'LDLIBS=-lm -lpthread -L'+lib+' -Wl,-rpath,'+lib+' -lscipy_openblas'], out/'build.log')
    engine = e / 'qwen_tts'
    write(out/'NATIVE_BINARY.json', {'engine_revision': QWEN_C_REV, 'sha256': digest(engine),
          'policy': 'original unactuated production CLI; not a CPC trajectory run'})
    model = snapshot('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', work/'custom',
                     out/'MODEL.json', CUSTOM_REV, ['*.json', '*.safetensors', '*.txt'])
    route = 'native_custom_xvector'; (out/route).mkdir(exist_ok=True)
    for c in cases():
        t = time.monotonic(); p = out/route/(c['id']+'.wav')
        cmd = [str(engine), '-d', str(model), '--load-voice', str(profile), '--xvector-only',
               '-l', 'English', '--text', c['text'], '--seed', str(c['seed']),
               '--temperature', '0.42', '--top-k', '40', '--top-p', '0.95', '--rep-penalty', '1.05',
               '--max-duration', '20', '-j4', '-o', str(p)]
        try:
            run(cmd, p.with_suffix('.log'), 600)
            record_wave(out, route, c, t, {'profile_sha256': digest(profile), 'command': cmd,
                         'engine_sha256': digest(engine), 'model_revision': CUSTOM_REV})
        except Exception as exc: fail_case(out, route, c, exc)


def qwen_reference(anchor: Path, profile: Path, out: Path, work: Path) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    model_dir = snapshot('Qwen/Qwen3-TTS-12Hz-1.7B-Base', work/'base', out/'MODEL.json', BASE_REV,
                         ['*.json', '*.safetensors', '*.txt'])
    model = Qwen3TTSModel.from_pretrained(str(model_dir), device_map='cpu',
                 dtype=torch.bfloat16, attn_implementation='sdpa')
    prompts = {kind: model.create_voice_clone_prompt(ref_audio=str(anchor), ref_text=REF_TEXT,
                  x_vector_only_mode=(kind == 'base_xvector')) for kind in ['base_xvector', 'base_icl']}
    prompt_audit = {}
    for kind, prompt in prompts.items():
        item = prompt[0]
        audit = {'python_type': str(type(item)), 'x_vector_only_mode': item.x_vector_only_mode,
                 'icl_mode': item.icl_mode}
        for field in ['ref_code', 'ref_spk_embedding']:
            x = getattr(item, field, None)
            if x is not None:
                x = x.detach().cpu().contiguous()
                audit[field] = {'shape': list(x.shape), 'sha256': hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest()}
        prompt_audit[kind] = audit
    write(out/'REFERENCE_CONDITIONING.json', prompt_audit)
    for kind in prompts: (out/kind).mkdir(exist_ok=True)
    for c in cases():
        for kind, prompt in prompts.items():
            t=time.monotonic()
            try:
                random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
                wavs, sr = model.generate_voice_clone(text=c['text'], language='English',
                  voice_clone_prompt=prompt, do_sample=True, temperature=0.7, top_k=50, top_p=1.0,
                  repetition_penalty=1.05, max_new_tokens=384)
                sf.write(out/kind/(c['id']+'.wav'), wavs[0], sr, subtype='PCM_16')
                record_wave(out, kind, c, t, {'model_revision': BASE_REV, 'torch': torch.__version__,
                     'prompt_audit': prompt_audit[kind], 'max_new_tokens':384,
                     'historical_bin_profile_loaded': False, 'same_anchor_reembedded': True})
            except Exception as exc: fail_case(out, kind, c, exc)


def f5(anchor: Path, profile: Path, out: Path, work: Path) -> None:
    import soundfile as sf
    import torch
    from f5_tts.api import F5TTS
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    md=snapshot('SWivid/F5-TTS',work/'f5',out/'MODEL.json', allow=['F5TTS_v1_Base/*'])
    voc=snapshot('charactr/vocos-mel-24khz',work/'vocos',out/'VOCODER.json',allow=['*.json','*.yaml','*.bin'])
    m=F5TTS(model='F5TTS_v1_Base',ckpt_file=str(md/'F5TTS_v1_Base/model_1250000.safetensors'),
       vocab_file=str(md/'F5TTS_v1_Base/vocab.txt'),device='cpu',vocoder_local_path=str(voc))
    route='f5_reference';(out/route).mkdir(exist_ok=True)
    for c in cases():
        t=time.monotonic()
        try:
            y,sr,_=m.infer(ref_file=str(anchor),ref_text=REF_TEXT,gen_text=c['text'],
                 nfe_step=32,cfg_strength=2,sway_sampling_coef=-1,speed=1,
                 cross_fade_duration=0,remove_silence=False,seed=c['seed'])
            sf.write(out/route/(c['id']+'.wav'),y,sr,subtype='PCM_16')
            record_wave(out,route,c,t,{'license_scope':'noncommercial research only; no production promotion',
               'ref_preprocessing':'upstream native preprocessing, not byte-identical feature extraction',
               'nfe_step':32,'cfg_strength':2,'speed':1,'no_acting_prompt':True})
        except Exception as exc: fail_case(out,route,c,exc)


def fish(anchor: Path, profile: Path, out: Path, work: Path) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from fish_speech.models.text2semantic.llama import DualARTransformer, precompute_freqs_cis
    from fish_speech.models.text2semantic.inference import decode_one_token_ar, generate_long, load_codec_model, encode_audio, decode_to_audio
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    md=snapshot('fishaudio/s2-pro',work/'fish',out/'MODEL.json',
                allow=['*.json','*.safetensors','*.pth','*.model','*.txt','LICENSE*'])
    with torch.device('meta'):
        model=DualARTransformer.from_pretrained(str(md),load_weights=True,max_length=4096)
    model.freqs_cis=precompute_freqs_cis(model.config.max_seq_len,model.config.head_dim,model.config.rope_base)
    model.fast_freqs_cis=precompute_freqs_cis(model.config.num_codebooks,model.config.fast_head_dim,model.config.rope_base)
    missing=[n for n,p in model.named_parameters() if p.is_meta]
    missing += [n for n,p in model.named_buffers() if p.is_meta]
    if missing: raise RuntimeError('Unmaterialized state: '+str(missing))
    model=model.to(device='cpu',dtype=torch.bfloat16).eval();model._cache_setup_done=False
    codec=load_codec_model(md/'codec.pth','cpu',torch.float32)
    ref=encode_audio(anchor,codec,'cpu').cpu()
    write(out/'CPU_LOADER.json',{'meta_initialization':True,'all_parameters_materialized':True,
       'max_seq_len':4096,'model_dtype':'bfloat16','codec_dtype':'float32','quantized':False,
       'reference_code_shape':list(ref.shape),'torch':torch.__version__})
    route='fish_s2_reference';(out/route).mkdir(exist_ok=True)
    for c in cases():
        t=time.monotonic()
        try:
            random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
            generated=[]
            for response in generate_long(model=model,device='cpu',decode_one_token=decode_one_token_ar,
               text=c['text'],prompt_text=[REF_TEXT],prompt_tokens=[ref],max_new_tokens=384,
               top_p=0.9,top_k=30,temperature=1.0,compile=False,chunk_length=300):
                if response.action=='sample':generated.append(response.codes)
            if len(generated)!=1:raise RuntimeError('Expected one short unstitched generation, got '+str(len(generated)))
            codes=generated[0];np.save(out/route/(c['id']+'.codes.npy'),codes.cpu().numpy())
            y=decode_to_audio(codes.to('cpu'),codec).cpu().float().numpy()
            sf.write(out/route/(c['id']+'.wav'),y,codec.sample_rate,subtype='PCM_16')
            record_wave(out,route,c,t,{'license_scope':'research only; no production promotion',
               'code_frames':int(codes.shape[-1]),'max_new_tokens':384,'meta_loader':True,'max_seq_len':4096})
        except Exception as exc: fail_case(out,route,c,exc)


def main() -> int:
    p=argparse.ArgumentParser();p.add_argument('route',choices=['native','qwen_reference','f5','fish'])
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    a.out=a.out.resolve();a.work=a.work.resolve();a.out.mkdir(parents=True,exist_ok=True);a.work.mkdir(parents=True,exist_ok=True)
    environment(a.out);write(a.out/'CASES.json',cases())
    write(a.out/'SCOPE.json',{'experiment':'basis-family discrimination','production_promotion':False,
      'human_acceptance':False,'training':False,'acting_instructions':False,'certification_set_used':False})
    try:
        anchor,profile=assets(a.assets)
        write(a.out/'INPUTS.json',{'anchor_sha256':digest(anchor),'profile_sha256':digest(profile),'text':REF_TEXT})
        globals()[a.route](anchor,profile,a.out,a.work)
        write(a.out/'EXECUTION.json',{'status':'FINISHED_INSPECT_PER_CASE_RESULTS',
              'wav_count':len(list(a.out.rglob('*.wav'))),'failure_count':len(list(a.out.rglob('*.failure.json')))})
        return 0
    except Exception as exc:
        write(a.out/'EXECUTION.json',{'status':'ROUTE_FAILED_NO_FALLBACK','error':repr(exc),'traceback':traceback.format_exc()})
        traceback.print_exc();return 1

if __name__=='__main__':raise SystemExit(main())
