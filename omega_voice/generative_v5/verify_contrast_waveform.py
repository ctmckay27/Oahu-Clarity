"""Two texts, chosen correction versus observation, irrelevant controls."""
import argparse,pathlib,json
import numpy as np,soundfile as sf
from ..causal_v4.runtime import compile_scene
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha
from .alignment import ForcedAligner
from .session import timeline_from_evaluation,durable_json
from .prominence import realize

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/contrast_focus';d.mkdir(exist_ok=True)
 sources=[{'source':r/'continuation/text_release/carrier_parity.wav','text':'I thought the spare key was upstairs. I can see it beside the blue bowl.','expected':'I can see it beside the red bowl.','observed':'I can see it beside the blue bowl.','at':7,'focus':13}, {'source':r/'continuation/scene_session/known_other_text/turn1/carrier.wav','text':'The package is beside the window. I will bring it to you.','expected':'The package is beside the door.','observed':'The package is beside the window.','at':3,'focus':5}]
 durable_json(d/'WAVE_PREDECLARED.json',{'sources':sources,'gates':{'WER':0,'identity_min':.6013473320007324,'outside_focus_PCM_exact':True,'no_control_PCM_exact':True,'irrelevant_PCM_exact':True,'observed_emphasis_increase':True,'quality_loss_UTMOS_max':.15},'scope':'source proposition contrast and bounded acoustic prominence; no character admission','full_completion':False})
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');rows=[]
 for i,case in enumerate(sources):
  s=case['source'];text=case['text'];q=ev.evaluate(s,text);alignment=al.align(ev.load(s),text,sha(s),q['wer']==0);timeline=timeline_from_evaluation(dict(q,alignment=alignment));contrast={'id':'discrepancy','kind':'contrast','at_word':case['at'],'expected':case['expected'],'observed':case['observed'],'perspective':'listener','source':{'text':'The listener expected: '+case['expected']+' Mari now observes: '+case['observed']}};action={'id':'chosen-correction','kind':'action','at_word':case['at'],'tactic':'correct','target':'resolve the listener misconception','source':{'text':'Mari chooses to correct the location or identifying detail the listener had wrong.'}}
  cases={'baseline':{},'observed_only':{'events':[contrast]},'correct':{'events':[contrast,action]},'irrelevant':{'events':[contrast,action],'metadata':{'wall_color':'green'}}}
  for k,scene in cases.items():
   p=compile_scene(text,scene,timeline=timeline,policy='contrast_focus_v7');out=d/f'{i}_{k}.wav';receipt=realize(s,out,p);quality=ev.evaluate(out,text);actual=al.align(ev.load(out),text,sha(out),quality['wer']==0) if quality['wer']==0 else None
   x,_=sf.read(s,dtype='int16');y,_=sf.read(out,dtype='int16');mask=np.zeros(len(x),dtype=bool)
   for w in receipt['windows']:mask[int(np.ceil(w['start']*24000)):int(np.ceil(w['end']*24000))]=True
   checks={'quality':quality['wer']==0 and quality['identity_pass'],'outside_focus_exact':np.array_equal(x[~mask],y[~mask]),'no_clipping':quality['audio']['clipping_fraction']==0}
   rows.append({'source':i,'case':k,'source_path':str(s),'text':text,'focus':case['focus'],'source_alignment':alignment,'plan':p,'receipt':receipt,'quality':quality,'alignment':actual,'checks':checks});durable_json(d/'WAVE_RESULTS.json',rows);print(i,k,checks,quality['speaker_similarity'],flush=True)
 checks={'all_quality':all(all(x['checks'].values())for x in rows)}
 for i in range(len(sources)):
  checks[f'{i}_no_intent_exact']=sha(d/f'{i}_baseline.wav')==sha(d/f'{i}_observed_only.wav')
  checks[f'{i}_irrelevant_exact']=sha(d/f'{i}_correct.wav')==sha(d/f'{i}_irrelevant.wav')
 durable_json(d/'WAVE_ASSESSMENT.json',{'checks':checks,'all_pass':all(checks.values()),'emphasis_observer_pending':True,'UTMOS_pending':True,'full_completion':False})
if __name__=='__main__':main()
