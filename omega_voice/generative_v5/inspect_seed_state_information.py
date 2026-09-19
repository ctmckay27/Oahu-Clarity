"""Measure which causal waveform distinctions survive the conversion interface.

Identical content conditioning and length, with fixed reference/noise, imply
identical deterministic output. Different conditioning is only necessary,
not sufficient, evidence of intended vocal realization.
"""
import argparse,pathlib,json
import numpy as np,librosa
from .seed_vc_diagnostic import ConversionProbe,sha
from .session import durable_json

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/seed_state_information';d.mkdir(exist_ok=True)
 pairs=[('listener_care','listener_causal/0_care_calm.wav','listener_causal/0_care_worried.wav'),('listener_clarification','listener_causal/0_clarify_calm.wav','listener_causal/0_clarify_confused.wav'),('factive_knowledge','factive_scene/0_no_result.wav','factive_scene/0_related.wav'),('irrelevant_knowledge','factive_scene/0_no_result.wav','factive_scene/0_unrelated.wav'),('physical_tenderness','physical_heldout/0_neutral.wav','physical_heldout/0_tenderness.wav'),('contained_irritation','physical_heldout/0_neutral.wav','physical_heldout/0_contained_irritation.wav')]
 durable_json(d/'PREDECLARED.json',{'pairs':pairs,'question':'does the model receive any distinguishing realization input beyond source waveform hash?','reference':'frozen selected Mari anchor','all_other_generation_inputs':'fixed length, reference mel/content/style, seed, steps and CFG','scope':'information preservation, not perceptual behavior','full_completion':False})
 p=ConversionProbe(r);m=p.model;t=p.torch;cache={};rows=[]
 def encode(rel):
  if rel in cache:return cache[rel]
  path=r/'continuation'/rel;x=librosa.load(path,sr=m.sr)[0];x16=librosa.resample(x,orig_sr=m.sr,target_sr=16000)
  with t.inference_mode():
   mel=m.mel_fn(t.from_numpy(x)[None]);_,codes,_=m.content_extractor_wide(t.from_numpy(x16)[None],[len(x16)]);mu,_=m.cfm_length_regulator(codes,ylens=t.tensor([mel.shape[-1]]))
  cache[rel]=(codes.numpy(),mu.numpy(),{'source_sha256':sha(path),'samples':len(x),'mel_frames':int(mel.shape[-1])});return cache[rel]
 for name,left,right in pairs:
  ca,ma,am=encode(left);cb,mb,bm=encode(right);compatible=ca.shape==cb.shape and ma.shape==mb.shape;same=compatible and np.array_equal(ca,cb)and np.array_equal(ma,mb)and am['mel_frames']==bm['mel_frames'];row={'id':name,'a':am,'b':bm,'code_shape_a':list(ca.shape),'code_shape_b':list(cb.shape),'length_regulated_shape_a':list(ma.shape),'length_regulated_shape_b':list(mb.shape),'identical_generation_information':bool(same),'changed_code_fraction':float(np.mean(ca!=cb))if ca.shape==cb.shape else None,'relative_conditioning_L2':float(np.linalg.norm(ma-mb)/(np.linalg.norm(ma)+1e-12))if ma.shape==mb.shape else None,'no_intended_behavior_claim':True};rows.append(row);durable_json(d/'RESULTS.json',rows);print(row,flush=True)
 durable_json(d/'PROVENANCE.json',p.provenance)
if __name__=='__main__':main()
