"""Same-speaker phonetic reference discriminator for a localized lexical omission."""
import argparse,pathlib,json
from .native import render
from .session import durable_json
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.runtime import ANCHOR
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/phonetic_reference';d.mkdir(exist_ok=True);source=r/'continuation/scene_session/known/turn1';native=json.loads((source/'carrier.receipt.json').read_text());p=source/'carrier.wav';assert sha(p)==native['audio']['sha256'];ref={'audio':str(p),'text':native['text'],'sha256':sha(p),'source_carrier_sha256':ANCHOR,'role':'same selected Mari carrier, independently verified prior pronunciation of checked; no new speaker or acting direction'};text='I checked the side door. It is open.';durable_json(d/'PREDECLARED.json',{'source_reference':ref,'source_native_receipt_sha256':sha(source/'carrier.receipt.json'),'matched_text':text,'matched_seed':98401,'baseline':'banked_acknowledgment/session/checked/carrier.wav','baseline_failure':'Whisper whole and clause ASR and independent CTCgreedy report check instead of checked','changed_factor':'project-owned ICL phonetic reference only, profile and sampler unchanged','no_identity_selection':True,'full_completion':False});audio=d/'checked.wav';rec=render(r,text,audio,seed=98401,reference=ref);ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');q=ev.evaluate(audio,text);durable_json(d/'RESULTS.json',{'native':rec,'quality':q,'full_completion':False});print(q['asr_text'],q['wer'],q['speaker_similarity'],flush=True)
if __name__=='__main__':main()
