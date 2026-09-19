"""A belief's confidence must not overwrite the force of a question."""
import argparse,json,pathlib
from .alignment import ForcedAligner
from .physical_finality import realize
from .session import durable_json
from ..causal_v4.runtime import compile_scene
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/speech_act_scope';d.mkdir(exist_ok=True)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');rows=[]
 cases=[('question','Did you leave the gate open?','heldout_question_anchor_icl.wav','ask'),
        ('assertion','That is exactly what I meant.','heldout_short_anchor_icl.wav','assert')]
 for name,text,file,act in cases:
  p=r/'continuation/anchor_context'/file;base=ev.evaluate(p,text)
  alignment=al.align(ev.load(p),text,sha(p),base['wer']==0)
  clock={'source_audio_sha256':sha(p),'duration_s':base['audio']['duration_s'],'words':[{'start':w['start'],'end':w['end']} for w in alignment['words']]}
  for confidence in [.2,.98]:
   events=[{'id':'confidence','kind':'assertion','at_word':0,'proposition':'current proposition','confidence':confidence,
     'source':{'text':'Matched-text counterfactual: source supplies this confidence about the current proposition.'}},
     {'id':'act','kind':'speech_act','at_word':0,'until_word':len(clock['words']),'mode':act,
      'source':{'text':'Mari asks the listener to verify their action.' if act=='ask' else 'Mari asserts that the listener has understood her intention.'}}]
   plan=compile_scene(text,{'events':events},timeline=clock,policy='linguistic_scope_v4');out=d/f'{name}_{confidence}.wav'
   durable_json(out.with_suffix('.plan.json'),plan);mechanism=realize(p,out,plan);q=ev.evaluate(out,text)
   rows.append({'id':name,'confidence':confidence,'quality':q,'mechanism':mechanism,'full_completion':False})
   durable_json(d/'RESULTS.json',rows);print(name,confidence,q['wer'],q['speaker_similarity'],flush=True)
 checks={'questions_preserve_identical_native_boundary':sha(d/'question_0.2.wav')==sha(d/'question_0.98.wav'),
   'assertion_commitment_still_changes_contour':sha(d/'assertion_0.2.wav')!=sha(d/'assertion_0.98.wav'),
   'quality':all(x['quality']['quality_screen_pass'] and x['quality']['wer']==0 for x in rows)}
 durable_json(d/'ASSESSMENT.json',{'checks':checks,'full_completion':False,'scope':'scoped epistemic finality; question/request-force realization not claimed'})
if __name__=='__main__':main()
