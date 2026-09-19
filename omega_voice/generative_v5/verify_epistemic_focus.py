"""Same carrier, relevant and irrelevant knowledge, observed waveform effects."""
import argparse,json,pathlib
from .session import timeline_from_evaluation
from .physical_finality import realize as finality
from .physical import realize as body
from .acoustics import measure
from ..causal_v4.runtime import compile_scene
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root)
 source=r/'continuation/scene_session/known_other_text/turn1';receipt=json.loads((source/'receipt.json').read_text());text=receipt['delivered_plan']['text']
 timeline=timeline_from_evaluation(dict(receipt['baseline'],alignment=receipt['alignment']));carrier=source/'carrier.wav'
 if timeline['source_audio_sha256']!=sha(carrier):raise ValueError('source carrier changed')
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 focus={'id':'current-claim','kind':'assertion','at_word':0,'proposition':'package location','confidence':.3,'source':{'text':'Mari makes a provisional claim about the package location.'}}
 def knowledge(prop):return {'id':'fact','kind':'knowledge','at_word':0,'proposition':prop,'status':'known','confidence':.98,'source':{'text':'Reliable evidence establishes '+prop+'.'}}
 cases=[('base',[focus]),('unrelated',[focus,knowledge('capital of France')]),('relevant',[focus,knowledge('package location')]),
        ('realization_alone',[focus,{'id':'integration','kind':'thought','at_word':0,'mode':'realizing','source':{'text':'An observation is integrated; it does not resolve the claim.'}}])]
 d=r/'continuation/epistemic_focus';d.mkdir(exist_ok=True);rows=[]
 for name,events in cases:
  scene={'events':events};plan=compile_scene(text,scene,timeline=timeline,policy='epistemic_focus_v2')
  contour=d/(name+'_contour.wav');c=finality(carrier,contour,plan)
  rebound=dict(timeline,source_audio_sha256=sha(contour));bodyplan=compile_scene(text,scene,timeline=rebound,policy='epistemic_focus_v2')
  audio=d/(name+'.wav');b=body(contour,audio,bodyplan);quality=ev.evaluate(audio,text)
  row={'id':name,'plan':plan,'contour':c,'body':b,'audio':str(audio),'sha256':sha(audio),'quality':quality,'acoustics':measure(audio)}
  rows.append(row);(d/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,quality['wer'],quality['speaker_similarity'],flush=True)
 checks={'unrelated_fact_exact_waveform':rows[0]['sha256']==rows[1]['sha256'],
         'thought_completion_does_not_invent_commitment':rows[0]['plan']['trajectory'][0]['controls']['finality']==rows[3]['plan']['trajectory'][0]['controls']['finality'],
         'relevant_evidence_changes_waveform':rows[0]['sha256']!=rows[2]['sha256'],
         'relevant_evidence_strengthens_observed_final_contour':rows[2]['acoustics']['final_slope_semitones_per_s']<rows[0]['acoustics']['final_slope_semitones_per_s'],
         'all_engineering_quality':all(x['quality']['quality_screen_pass'] and x['quality']['wer']==0 for x in rows)}
 (d/'ASSESSMENT.json').write_text(json.dumps({'checks':checks,'complete':False,'scope':'proposition-specific knowledge-to-waveform causality; person-like performance remains unverified'},indent=2)+'\n');print(checks)
if __name__=='__main__':main()
