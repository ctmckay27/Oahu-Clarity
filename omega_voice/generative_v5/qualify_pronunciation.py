"""Predeclared lexical correction generalization; no change to required words."""
import argparse,pathlib,json,re
from .native import render,anchor_reference
from .session import durable_json
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/pronunciation_qualification';d.mkdir(exist_ok=True)
 cases=[('receipt','She checked the receipt before lunch.',98500),('twice','I checked it twice.',98501),('figures','We checked the figures together.',98502)]
 durable_json(d/'PREDECLARED.json',{'cases':cases,'rule':{'checked':'chekt'},'required_target_unchanged':True,'comparison':'same text, seed, frozen profile and anchor; only pronunciation spelling differs','gates':{'every_corrected_WER':0,'identity_min':.6013473320007324,'maximum_predicted_MOS_loss':.15},'no_fitting_on_these_cases':True,'full_completion':False})
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');out=[]
 for name,target,seed in cases:
  for variant,renderer_text in [('original',target),('phonetic',re.sub(r'\bchecked\b','chekt',target))]:
   p=d/f'{name}_{variant}.wav';rec=render(r,renderer_text,p,seed=seed,reference=anchor_reference(r));q=ev.evaluate(p,target);out.append({'id':name,'variant':variant,'target':target,'renderer_text':renderer_text,'seed':seed,'native_receipt':rec,'quality':q});durable_json(d/'RESULTS.json',out);print(name,variant,q['asr_text'],q['wer'],q['speaker_similarity'],flush=True)
 durable_json(d/'CONTENT_ASSESSMENT.json',{'checks':{'all_corrected_words_exact':all(x['quality']['wer']==0 for x in out if x['variant']=='phonetic'),'all_corrected_identity_pass':all(x['quality']['identity_pass'] for x in out if x['variant']=='phonetic')},'naturalness_and_independent_content_pending':True,'full_completion':False})
if __name__=='__main__':main()
