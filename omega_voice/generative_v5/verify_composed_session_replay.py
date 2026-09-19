"""Exercise composed realization with an explicitly reused verified carrier.

This tests the physical/session path; it does not claim a fresh native render.
The source waveform and original native receipt remain separate from the replay.
"""
import argparse,pathlib,json,shutil
import soundfile as sf,numpy as np
from .session import PerformanceSession,durable_json
from .alignment import ForcedAligner
from ..causal_v4.evaluate import Evaluator
from ..causal_v4.renderer import sha
from ..causal_v4.runtime import digest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('root');a=ap.parse_args();r=pathlib.Path(a.root);d=r/'continuation/composed_session_replay';d.mkdir(exist_ok=True);cases=json.loads((r/'continuation/two_contrasts/RESULTS.json').read_text());source=r/'continuation/text_release/carrier_parity.wav';original=source.with_suffix('.receipt.json');native=json.loads(original.read_text())
 if sha(source)!='fcf075907e6449cc26c8be0c97c8a84b87e161a6f709ce1bdefd2bd780e57e9e' or native['audio']['sha256']!=sha(source):raise ValueError('unverified replay source')
 durable_json(d/'PREDECLARED.json',{'source_sha256':sha(source),'native_receipt_sha256':sha(original),'scope':'explicit saved-native-carrier replay through actual physical session mechanism; no fresh-render or selected backend claim','cases':[{'id':x['id'],'scene':x['plan']['scene']}for x in cases],'gates':{'raw_full_WER_max':.12,'strict_lexical_evidence':'whole WER0 or every verified clause WER0, all samples and words covered without hints','identity_min':.6013473320007324,'actual_word_clock_error_max_s':.08,'first_focus_prefix_invariance':True},'full_completion':False})
 ev=Evaluator(r/'models/whisper',r/'models/ecapa',r/'recovered/production_release/production/MARI_VOICE_V1_ANCHOR.wav');al=ForcedAligner('/tmp/mari-alignment-large',r/'continuation/FORCED_ALIGNMENT_LARGE_DEPENDENCIES.json');text=native['text'];base=ev.evaluate(source,text)
 if base['wer']!=0 or not base['identity_pass']:raise ValueError('replay source failed current observer')
 rows=[]
 for c in cases:
  folder=d/c['id'];session=PerformanceSession(r,folder,ev,mode='physical',aligner=al,temporal_policy='contrast_focus_v7',conditioning='selected_anchor',articulation=True,respiration=True,prominence=True,prominence_calibration=r/'continuation/matched_emphasis/SUMMARY.json');target=folder/'turn';target.mkdir();carrier=target/'carrier.wav';shutil.copyfile(source,carrier);shutil.copyfile(original,carrier.with_suffix('.receipt.json'));request=digest({'role':'explicit saved carrier replay','native_receipt':sha(original),'source':sha(source),'scene':c['plan']['scene']})
  try:
   rec=session._render_physical('turn',request,target,carrier,base,text,c['plan']['scene'],None,None,None);row={'id':c['id'],'admitted':rec['quality_admitted'],'receipt_sha256':sha(target/'receipt.json'),'receipt':rec,'full_completion':False}
  except Exception as error:row={'id':c['id'],'admitted':False,'error':repr(error),'receipt':json.loads((target/'receipt.json').read_text())if(target/'receipt.json').exists()else None,'full_completion':False}
  rows.append(row);durable_json(d/'RESULTS.json',rows);print(c['id'],row['admitted'],row.get('error'),flush=True)
 if all(x['admitted']for x in rows):
  a,sr=sf.read(d/'first_only/turn/performance.wav',dtype='int16');b,_=sf.read(d/'both/turn/performance.wav',dtype='int16');rec=rows[1]['receipt'];start=round(rec['trials'][-1]['observed_alignment']['words'][13]['start']*sr);same=np.array_equal(a[:start],b[:start]);durable_json(d/'ASSESSMENT.json',{'quality_pass':True,'prefix_before_later_focus_exact':bool(same),'comparison_cut_s':start/sr,'full_completion':False});print('prefix_exact',same,flush=True)
if __name__=='__main__':main()
