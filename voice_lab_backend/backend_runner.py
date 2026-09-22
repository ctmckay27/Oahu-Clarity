"""Renderer/conditioning investigation, not identity certification."""
import argparse,copy,importlib.util,json,pathlib,shutil
spec=importlib.util.spec_from_file_location('base','voice_lab_sealed/runner.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
b.MODEL_REV='fd4b254389122332181a7c3db7f27e918eec64e3'
source={c['id']:c for c in b.CASES};cases=[]
for key in ['O01','F05','L01']:
    for mode in ['profile','reference_xvector','reference_icl']:
        c=copy.deepcopy(source[key]);c.update(id='X'+key+'_'+mode,category='renderer_conditioning_probe',mode=mode,comparison_group=key,source_case=key,actual_model='Qwen/Qwen3-TTS-12Hz-1.7B-Base');cases.append(c)
b.CASES=cases
original_prepare=b.prepare
def prepare():
    original_prepare()
    p=b.OUT/'environment.json';d=json.loads(p.read_text());d['model']='Qwen/Qwen3-TTS-12Hz-1.7B-Base';d['comparison_model']='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice';d['reference_conditioning_control']='same recovered G1-1 waveform / profile; no generic substitute';b.write(p,d)
b.prepare=prepare
p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','evaluate']);a=p.parse_args()
if a.action=='generate':
    b.generate(0,1);shutil.copy2(__file__,b.OUT/'backend_runner.py')
else:b.evaluate()
