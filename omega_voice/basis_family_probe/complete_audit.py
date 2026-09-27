"""Independent technical audit; no synthetic listener ratings or voice promotion."""
from pathlib import Path
import hashlib,json,re,sys,traceback
import numpy as np
import soundfile as sf
import torch,torchaudio
from huggingface_hub import HfApi,snapshot_download
from faster_whisper import WhisperModel
from speechbrain.inference.speaker import EncoderClassifier
from family_probe import assets,digest,write

def words(s):
    s=s.lower().replace('’',"'")
    contractions={"it's":"it is","haven't":"have not","hasn't":"has not","don't":"do not","isn't":"is not"}
    for a,b in contractions.items():s=re.sub(r'\b'+re.escape(a)+r'\b',b,s)
    numbers={'ten':'10','thirty':'30'}
    return [numbers.get(w,w) for w in re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?",s)]

def ed(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

def main():
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    out=Path('audit');out.mkdir(exist_ok=True);anchor,_=assets(Path('package'))
    api=HfApi();versions={};errors=[]
    asr=None;encoder=None
    try:
        name='Systran/faster-whisper-small.en';rev=api.model_info(name).sha
        path=snapshot_download(name,revision=rev);versions['asr']={'model':name,'revision':rev}
        asr=WhisperModel(path,device='cpu',compute_type='int8',cpu_threads=4)
    except Exception as e:errors.append({'component':'asr','error':repr(e)})
    try:
        name='speechbrain/spkrec-ecapa-voxceleb';rev=api.model_info(name).sha
        path=snapshot_download(name,revision=rev);versions['speaker_encoder']={'model':name,'revision':rev}
        encoder=EncoderClassifier.from_hparams(source=path,savedir='ecapa_runtime',overrides={'pretrained_path':path},run_opts={'device':'cpu'})
    except Exception as e:errors.append({'component':'speaker_encoder','error':repr(e)})
    write(out/'EVALUATOR_MODELS.json',versions)
    vectors=[];vector_index=[]
    def embed(p,label):
        y,sr=sf.read(p,dtype='float32',always_2d=True);t=torch.from_numpy(y.mean(1)).unsqueeze(0)
        if sr!=16000:t=torchaudio.functional.resample(t,sr,16000)
        with torch.no_grad():e=encoder.encode_batch(t).reshape(-1).cpu().numpy()
        e=e/max(np.linalg.norm(e),1e-12);vectors.append(e);vector_index.append({'id':label,'path':str(p),'sha256':digest(p)})
        return e
    ae=embed(anchor,'identity_reference') if encoder else None
    # External examples are observational controls, never synthesis conditioning.
    external=[]
    for p in sorted(Path('external').rglob('*.wav')):
        if encoder:
            e=embed(p,p.stem);external.append({'id':p.stem,'anchor_cosine':float(ae@e),'role':'EXTERNAL_ANALYSIS_ONLY'})
    rows=[]
    for p in sorted(Path('cohorts').rglob('U*.wav')):
        row={'path':str(p),'sha256':digest(p),'human_accepted':False}
        try:
            r=json.loads(p.with_suffix('.json').read_text());y,sr=sf.read(p,always_2d=True)
            threshold=1-1/32768 if sf.info(p).subtype=='PCM_16' else 1
            row.update({'route':r['route'],'case_id':r['id'],'text':r['text'],'seed':r['seed'],'duration_s':len(y)/sr,'sample_rate':sr,'receipt_matches':digest(p)==r['waveform_sha256'],'pcm_rail_samples':int((np.abs(y)>=threshold).sum())})
            if asr:
                segs,_=asr.transcribe(str(p),language='en',beam_size=5,temperature=0,condition_on_previous_text=False,vad_filter=False,word_timestamps=True)
                segs=list(segs);hyp=' '.join(s.text.strip() for s in segs);a,b=words(r['text']),words(hyp)
                row.update({'asr_text':hyp,'normalized_wer':ed(a,b)/max(1,len(a)),
                 'asr_words':[{'word':w.word,'start_s':w.start,'end_s':w.end,'probability':w.probability} for s in segs for w in (s.words or [])],
                 'timing_scope':'ASR-derived word timing, not ground-truth phonetics or evidence of a cognitive cause'})
            if encoder:
                e=embed(p,r['route']+'-'+r['id']);row['anchor_ecapa_cosine']=float(ae@e)
        except Exception as e:row['audit_error']=repr(e);row['traceback']=traceback.format_exc()
        rows.append(row);write(out/'COMPLETE_AUDIO_AUDIT.json',{'rows':rows,'errors':errors,'perceptual_certification':False})
    if vectors:
        arr=np.stack(vectors);np.savez_compressed(out/'LEARNED_SPEAKER_EMBEDDINGS.npz',embeddings=arr,cosine_matrix=arr@arr.T)
        write(out/'EMBEDDING_INDEX.json',{'rows':vector_index,'model':versions.get('speaker_encoder'),'scope':'learned generic speaker features, not a fitted Mari identity model'})
    summary=[]
    for route in sorted({r.get('route','unknown') for r in rows}):
        rs=[r for r in rows if r.get('route')==route];wer=[r['normalized_wer'] for r in rs if 'normalized_wer' in r];sim=[r['anchor_ecapa_cosine'] for r in rs if 'anchor_ecapa_cosine' in r]
        summary.append({'route':route,'renders':len(rs),'distinct_texts':len({r.get('text') for r in rs}),'wer_zero_count':sum(x==0 for x in wer),'asr_evaluated':len(wer),'median_wer':float(np.median(wer)) if wer else None,'median_anchor_cosine':float(np.median(sim)) if sim else None,'audit_errors':sum('audit_error' in r for r in rs)})
    write(out/'COMPLETE_AUDIO_AUDIT.json',{'rows':rows,'errors':errors,'summary':summary,'external_controls':external,'interpretation':'No ranking of naturalness or Mari-ness. Eight texts with two extra seeds per route are not broad certification. Embedding similarity is uncalibrated.','perceptual_certification':False,'training':False})
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
