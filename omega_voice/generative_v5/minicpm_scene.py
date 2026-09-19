"""Local source-quoted scene compiler candidate. Syntax is not semantic proof."""
import argparse,hashlib,json,pathlib,subprocess,time
from .scene_language import CONTRACT,event_grammar,parse_events
from ..causal_v4.runtime import compile_scene
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
class LocalSceneCompiler:
 def __init__(self,root,model='/tmp/mari-minicpm-gguf/MiniCPM-o-4_5-Q8_0.gguf',binary='/tmp/mari-minicpm-engine/build/bin/mari-scene-compiler'):
  self.root=pathlib.Path(root);self.model=pathlib.Path(model);self.binary=pathlib.Path(binary)
  self.build=json.loads((self.root/'continuation/minicpm_inspection/COMBINED_BUILD_RECEIPT.json').read_text())
  assert sha(self.binary)==self.build['scene_binary_sha256']
  assert sha(self.model)=='ae6af22ad3b7f1a7bf667922af84b9eb3e2199cc86f402702eaf9b054054788d'
 def compile(self,scene,text,destination,prior=None,policy='linguistic_scope_v4'):
  out=pathlib.Path(destination);out.mkdir(parents=True,exist_ok=False)
  if '<|' in scene or '<|' in text:raise ValueError('reserved model delimiters in input')
  # The model has no audio or voice instructions. Its only output is a
  # grammar-constrained causal representation validated by the runtime.
  prompt='<|im_start|>system\n'+CONTRACT+'<|im_end|>\n<|im_start|>user\nSCENE:\n'+scene+'\nSPOKEN TEXT:\n'+text+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
  (out/'prompt.txt').write_text(prompt);(out/'schema.json').write_text(json.dumps(event_grammar(scene,text)))
  cmd=[str(self.binary),str(self.model),str(out/'prompt.txt'),str(out/'schema.json'),str(out/'raw.json')]
  start=time.monotonic()
  with (out/'run.log').open('w')as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  receipt={'command':cmd,'returncode':r.returncode,'elapsed_seconds':time.monotonic()-start,'build':self.build,'scene':scene,'text':text,'policy':policy,'model_sha256':'ae6af22ad3b7f1a7bf667922af84b9eb3e2199cc86f402702eaf9b054054788d','source_sha256':sha(__file__)}
  (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
  if r.returncode:raise RuntimeError('scene model execution failed: '+str(r.returncode))
  raw=(out/'raw.json').read_text();result=parse_events(raw,scene,text,prior)
  result['plan']=compile_scene(text,result['scene'],prior,policy=policy)
  result['raw_model_output']=raw;result['provenance']=receipt
  result['admitted_as_general_compiler']=False
  (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('corpus');ap.add_argument('output');a=ap.parse_args()
 out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True);compiler=LocalSceneCompiler(a.root);rows=[]
 for i,case in enumerate(json.loads(pathlib.Path(a.corpus).read_text())):
  row={'case':case}
  try:row['result']=compiler.compile(case['scene'],case['spoken'],out/f'{i:03}')
  except Exception as e:row['error']=repr(e)
  rows.append(row);(out/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n');print(case['id'],row.get('error',row.get('result',{}).get('scene')),flush=True)
if __name__=='__main__':main()
