"""Explicit lexical pronunciation conditioning; target speech is never rewritten."""
import argparse,pathlib,json
from .native import render,anchor_reference
from .session import durable_json
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/pronunciation_adapter';d.mkdir(exist_ok=True);target='I checked the side door. It is open.';renderer_text='I chekt the side door. It is open.';durable_json(d/'PREDECLARED.json',{'required_spoken_text':target,'renderer_phonetic_spelling':renderer_text,'lexical_adapter':{'checked':{'target_CMU_phonemes':['CH','EH1','K','T'],'renderer_spelling':'chekt','purpose':'retain voiceless past-tense final stop, absent in independent whole/clause/greedy ASR of baseline'}},'same_seed':98401,'same_frozen_reference':True,'no_semantic_text_substitution_allowed':True,'source_failures':['banked_acknowledgment/session/checked/LEXICAL_DIAGNOSIS.json','phonetic_reference/RESULTS.json'],'gates':{'independent_target_WER':0,'identity_min':.6013473320007324,'naturalness_gate_separate':True},'full_completion':False});audio=d/'checked.wav';rec=render(r,renderer_text,audio,seed=98401,reference=anchor_reference(r));ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');q=ev.evaluate(audio,target);durable_json(d/'RESULTS.json',{'native_renderer_receipt':rec,'required_spoken_text':target,'actual_renderer_text':renderer_text,'target_quality':q,'identity_profile_unchanged':True,'full_completion':False});print(q['asr_text'],q['wer'],q['speaker_similarity'],flush=True)
if __name__=='__main__':main()
