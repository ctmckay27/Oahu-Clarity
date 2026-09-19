"""Held-out within-utterance composition and causal clock accumulation."""
import argparse,pathlib,json,copy
import numpy as np,soundfile as sf
from .session import durable_json
from .prominence_duration import realize
from .alignment import ForcedAligner
from ..causal_v4.runtime import compile_scene
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/two_contrasts';d.mkdir(exist_ok=True)
 case=next(x for x in json.loads((r/'continuation/contrast_focus/WAVE_RESULTS.json').read_text())if x['source']==0 and x['case']=='correct');source=pathlib.Path(case['source_path']);p=case['plan'];cal=r/'continuation/matched_emphasis/SUMMARY.json'
 first=[{'id':'prior-belief-contrast','kind':'contrast','at_word':0,'expected':'I thought the spare key was downstairs.','observed':'I thought the spare key was upstairs.','perspective':'listener','source':{'kind':'authored_scene','text':'The listener misunderstood which floor Mari originally thought the key was on; she corrects that history before describing the current observation.'}}, {'id':'correct-history','kind':'action','at_word':0,'tactic':'correct','target':'correct the listener account of Mari prior belief','source':{'kind':'authored_scene','text':'Mari chooses to correct the account of her prior belief.'}}]
 variants={'first_only':{'events':first},'both':{'events':first+copy.deepcopy(p['scene']['events'])}}
 durable_json(d/'PREDECLARED.json',{'text':case['text'],'source_audio':str(source),'source_sha256':sha(source),'variants':variants,'gates':{'WER':0,'identity_min':.6013473320007324,'actual_clock_error_max_s':.08,'earlier_output_prefix_exact':True,'two_distinct_focus_words':True,'predicted_naturalness_loss_max':.15},'purpose':'test previously unseen sequential focus composition, accumulated sample offsets, and absence of backward influence from the later contrast','full_completion':False})
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');rows=[]
 for name,scene in variants.items():
  out=d/(name+'.wav');plan=compile_scene(case['text'],scene,p['initial_state'],timeline=p['realization_timeline'],policy='contrast_focus_v7');rec=realize(source,out,plan,cal);quality=ev.evaluate(out,case['text']);alignment=al.align(ev.load(out),case['text'],sha(out),True)if quality['wer']==0 else None
  error=max(abs(a[k]-b[k])for a,b in zip(rec['mapped_timeline']['words'],alignment['words'])for k in ['start','end'])if alignment else None
  row={'id':name,'audio':str(out),'plan':plan,'receipt':rec,'quality':quality,'alignment':alignment,'clock_error_s':error,'full_completion':False};rows.append(row);durable_json(d/'RESULTS.json',rows);print(name,quality['wer'],quality['speaker_similarity'],error,flush=True)
 first_audio,sr=sf.read(d/'first_only.wav',dtype='int16');both_audio,_=sf.read(d/'both.wav',dtype='int16');later=rows[1]['receipt']['windows'][1]['delivered_start_sample'];checks={'first_window_shared':rows[0]['receipt']['windows'][0]==rows[1]['receipt']['windows'][0],'earlier_prefix_exact':np.array_equal(first_audio[:later],both_audio[:later]),'two_focus_words':[w['word']for w in rows[1]['receipt']['windows']]==[6,13],'WER0':all(x['quality']['wer']==0 for x in rows),'identity':all(x['quality']['identity_pass']for x in rows),'actual_clock':all(x['clock_error_s']is not None and x['clock_error_s']<=.08 for x in rows)}
 durable_json(d/'ASSESSMENT.json',{'checks':checks,'all_pass':all(checks.values()),'later_focus_start_s':later/sr,'emphasis_and_naturalness_pending':True,'full_completion':False});print(checks,flush=True)
if __name__=='__main__':main()
