"""Factive scene result -> focused knowledge -> local final-contour evidence."""
import argparse,pathlib,json,subprocess
import numpy as np,soundfile as sf
from .session import durable_json,timeline_from_evaluation
from .alignment import ForcedAligner
from .physical_finality import realize
from .acoustics import measure
from ..causal_v4.runtime import compile_scene
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/factive_scene';d.mkdir(exist_ok=True)
 sources=[{'path':r/'continuation/text_release/carrier_parity.wav','text':'I thought the spare key was upstairs. I can see it beside the blue bowl.','proposition':'the spare key is beside the blue bowl','boundary':7},
 {'path':r/'continuation/scene_session/known_other_text/turn1/carrier.wav','text':'The package is beside the window. I will bring it to you.','proposition':'the package is beside the window','boundary':3}]
 cases=[]
 for i,x in enumerate(sources):
  prefix=f"At spoken word index {x['boundary']}, "
  scenes={'no_result':prefix+'Mari realizes something.','unrelated':prefix+'Mari realizes that the office is closed.','related':prefix+'Mari realizes that '+x['proposition']+'.','irrelevant':prefix+'Mari realizes that '+x['proposition']+'. There is a blue chair in the room.'}
  for name,scene in scenes.items():cases.append({'source':i,'id':name,'scene':scene,'spoken':x['text'],'context':{'assertion':{'proposition':x['proposition'],'mode':'assert','confidence':.3,'source':{'text':'Mari is making a provisional claim about the current location.'}}}})
 durable_json(d/'WAVE_CORPUS.json',cases)
 code="""import json,sys,spacy
from omega_voice.generative_v5.explicit_scene import ExplicitSceneCompiler
c=ExplicitSceneCompiler(spacy.load('en_core_web_sm'),policy='listener_causal_v6');out=[]
for case in json.load(sys.stdin):
 r=c.compile(case['scene'],case['spoken'],context=case['context'])
 if not r['admitted']:raise ValueError(r['unresolved'])
 out.append(dict(case=case,result=r))
print(json.dumps(out))
"""
 p=subprocess.run(['/tmp/mari-parserenv/bin/python','-c',code],input=json.dumps(cases),text=True,capture_output=True,check=True);compiled=json.loads(p.stdout);durable_json(d/'WAVE_COMPILED.json',compiled)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');rows=[];checks=[]
 for i,x in enumerate(sources):
  q=ev.evaluate(x['path'],x['text']);alignment=al.align(ev.load(x['path']),x['text'],sha(x['path']),q['wer']==0);clock=timeline_from_evaluation(dict(q,alignment=alignment));at=round(clock['words'][x['boundary']]['start']*24000)
  for item in [y for y in compiled if y['case']['source']==i]:
   name=item['case']['id'];plan=compile_scene(x['text'],item['result']['scene'],timeline=clock,policy='listener_causal_v6');out=d/f'{i}_{name}.wav';receipt=realize(x['path'],out,plan);quality=ev.evaluate(out,x['text']);row={'source':i,'case':name,'plan':plan,'mechanism':receipt,'sha256':sha(out),'quality':quality,'acoustics':measure(out),'full_completion':False};rows.append(row);durable_json(d/'WAVE_RESULTS.json',rows);print(i,name,quality['wer'],quality['speaker_similarity'],flush=True)
  local={y['case']:y for y in rows if y['source']==i};a0,_=sf.read(d/f'{i}_no_result.wav',dtype='int16');b,_=sf.read(d/f'{i}_related.wav',dtype='int16')
  check={'source':i,'all_quality':all(y['quality']['quality_screen_pass']and y['quality']['wer']==0 for y in local.values()),'no_result_not_certainty':local['no_result']['plan']['final_state']['knowledge_state']['certainty']==.3,'unrelated_exact':local['no_result']['sha256']==local['unrelated']['sha256'],'irrelevant_exact':local['related']['sha256']==local['irrelevant']['sha256'],'no_future_audio_leak':bool(np.array_equal(a0[:at],b[:at])),'relevant_changes_audio':local['related']['sha256']!=local['no_result']['sha256'],'knowledge_persists':x['proposition']in local['related']['plan']['final_state']['knowledge_state']['known'],'contour_strengthens':local['related']['acoustics']['final_slope_semitones_per_s']<local['no_result']['acoustics']['final_slope_semitones_per_s']};checks.append(check)
 durable_json(d/'WAVE_ASSESSMENT.json',{'checks':checks,'all_pass':all(all(v for k,v in x.items()if k!='source')for x in checks),'scope':'finite factive complements and focused epistemic acoustic consequences; no perceptual character admission','full_completion':False})
if __name__=='__main__':main()
