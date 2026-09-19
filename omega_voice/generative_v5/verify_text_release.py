"""Discriminate causal lexical release from an arbitrary acoustic perturbation."""
import argparse,json,pathlib,numpy as np,soundfile as sf
from transformers import AutoTokenizer
from .native import render
from .text_release import write_release,compile_release
from .alignment import ForcedAligner
from .session import timeline_from_evaluation
from ..causal_v4.runtime import compile_scene
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/text_release';d.mkdir(exist_ok=True)
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav')
 aligner=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json')
 tokenizer=AutoTokenizer.from_pretrained(r/'models/qwen-custom');rows=[]
 texts=['I thought the package was upstairs. I can see it beside the window.',
        'I thought the package was upstairs. It is still missing from the cupboard.']
 def generate(name,text,seed,**kwargs):
  audio=d/(name+'.wav');record=render(r,text,audio,seed,**kwargs);quality=ev.evaluate(audio,text)
  row={'id':name,'audio':str(audio),'sha256':sha(audio),'quality':quality,'renderer':record,'full_completion':False};rows.append(row)
  (d/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n');print(name,quality['asr_text'],quality['speaker_similarity'],flush=True)
  return audio,row
 # Exact selected-carrier parity after a native engine change.
 parity,_=generate('carrier_parity','I thought the spare key was upstairs. I can see it beside the blue bowl.',94400)
 expected='fcf075907e6449cc26c8be0c97c8a84b87e161a6f709ce1bdefd2bd780e57e9e'
 if sha(parity)!=expected:raise ValueError('default native route changed selected carrier bytes')
 for i,text in enumerate(texts):
  audio,row=generate(str(i)+'_incremental',text,94700,incremental_text=True)
  if row['quality']['wer']!=0:continue
  alignment=aligner.align(ev.load(audio),text,sha(audio),True)
  timeline=timeline_from_evaluation(dict(row['quality'],alignment=alignment))
  scene={'events':[{'id':'new-evidence','at_word':6,'kind':'thought','mode':'realizing',
    'source':{'text':'Only at the second sentence does Mari obtain the observation determining its content.'}}]}
  plan=compile_scene(text,scene,timeline=timeline,policy='bounded_thought_recovery_v1')
  packet=d/(str(i)+'.mrl');source=compile_release(plan,tokenizer,packet)
  (d/(str(i)+'_release_plan.json')).write_text(json.dumps({'plan':plan,'release':source},indent=2)+'\n')
  # A shared external evidence clock makes the two futures comparable.
  if i==0:
   release_frame=source['gates'][0]['release_frame']
   token_ids=source['token_ids'][:-1];zero=d/'zero.mrl';write_release(zero,token_ids,[0]+list(range(len(token_ids))))
   zeroaudio,_=generate('zero_release',text,94700,incremental_text=True,text_release=zero)
   if sha(audio)!=sha(zeroaudio):raise ValueError('zero schedule changed the upstream incremental waveform')
  else:
   if 'release_frame' not in locals():continue
   token_ids=source['token_ids'][:-1];offsets=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
   frames=[0]+list(range(len(token_ids)));boundary=text.index(' It is')
   for j,(start,end) in enumerate(offsets):
    if end>boundary:frames[j]=max(frames[j],release_frame)
   for j in range(1,len(frames)):frames[j]=max(frames[j],frames[j-1])
   source=write_release(packet,token_ids,frames)
   (d/(str(i)+'_shared_clock.json')).write_text(json.dumps({'release':source,'source':'First scene measured evidence-arrival clock shared by counterfactual; content changes only after that event.'},indent=2))
  generate(str(i)+'_withheld_future',text,94700,incremental_text=True,text_release=packet)
 paths=[d/(str(i)+'_withheld_future.wav') for i in range(2)]
 if all(p.exists() for p in paths):
  aa,sr=sf.read(paths[0],dtype='int16');bb,_=sf.read(paths[1],dtype='int16');n=min(len(aa),len(bb));diff=np.flatnonzero(aa[:n]!=bb[:n]);prefix=int(diff[0]) if len(diff) else n
  (d/'PREFIX_ASSESSMENT.json').write_text(json.dumps({'identical_prefix_samples':prefix,'identical_prefix_s':prefix/sr,'evidence_release_frame':release_frame,'evidence_release_s':release_frame*.08,'all_quality_pass':all(x['quality']['quality_screen_pass'] and x['quality']['wer']==0 for x in rows),'scope':'information-access and waveform-prefix test; cognitive coherence and character not thereby established'},indent=2))
if __name__=='__main__':main()
