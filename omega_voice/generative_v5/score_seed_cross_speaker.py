"""Independent speaker/content checks for completed fixed conversion cases."""
import argparse,pathlib,json
from .session import durable_json
from ..causal_v4.evaluate import Evaluator,intelligibility_tokens,distance
from ..causal_v4.renderer import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/seed_vc_cross_speaker';cases=json.loads((d/'PREDECLARED.json').read_text())['cases'];human={x['id']:x['human_transcript']for x in json.loads((d/'HUMAN_TRANSCRIPTS.json').read_text())}
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');enroll={x['id']:ev.embedding(x['source'])for x in cases};enroll['Mari']=ev.anchor
 results=json.loads((d/'QUALITY_RESULTS.json').read_text())if(d/'QUALITY_RESULTS.json').exists()else[];done={x['id']for x in results}
 for case in cases:
  k=case['id'];out=d/(k+'.wav')
  if k in done or not out.exists():continue
  receipt=json.loads(out.with_suffix('.receipt.json').read_text())
  if sha(out)!=receipt['output_sha256']or sha(case['source'])!=case['provenance']['audio_sha256']:raise ValueError('conversion evidence changed')
  segments,_=ev.asr.transcribe(ev.load(case['source']),language='en',beam_size=5,temperature=0,condition_on_previous_text=False,vad_filter=False);source_asr=' '.join(s.text.strip()for s in segments)
  q=ev.evaluate(out,source_asr);z=ev.embedding(out);scores={name:float(z@v)for name,v in enroll.items()};source_wer=distance(intelligibility_tokens(human[k]),intelligibility_tokens(source_asr))/max(1,len(intelligibility_tokens(human[k])));output_wer=distance(intelligibility_tokens(human[k]),intelligibility_tokens(q['asr_text']))/max(1,len(intelligibility_tokens(human[k])))
  checks={'target_identity':q['identity_pass'],'target_identity_wins':max(scores,key=scores.get)=='Mari','ASR_content_retention':q['wer']<=.12,'duration':abs(receipt['output_seconds']/receipt['source_seconds']-1)<=.04,'no_clipping':q['audio']['clipping_fraction']==0}
  row={'id':k,'source_asr':source_asr,'human_transcript':human[k],'source_ASR_vs_human_WER':source_wer,'output_ASR_vs_human_WER':output_wer,'quality':q,'speaker_scores':scores,'checks':checks,'all_pass':all(checks.values()),'full_completion':False};results.append(row);durable_json(d/'QUALITY_RESULTS.json',results);print(k,checks,scores,q['asr_text'],flush=True)
 if len(results)==len(cases):durable_json(d/'ASSESSMENT.json',{'all_pass':all(x['all_pass']for x in results),'cases':len(results),'scope':'cross-speaker identity transfer and lexical retention on fixed speech; not character or short-event admission','full_completion':False})
if __name__=='__main__':main()
