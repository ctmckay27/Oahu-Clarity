"""Isolated synthesis research. All voice inputs/outputs stay off the repository.
Results must be encrypted by the workflow before artifact upload.
No identity certification, publication, or CURRENT mutation is performed.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, platform, re, shutil, subprocess, time, traceback, zipfile
import numpy as np
import soundfile as sf

ROOT = pathlib.Path('private')
OUT = ROOT / 'results'
ENGINE_REV = 'e391ec5467b0218eeb175f4888ad65b259d1e7c7'
MODEL_REV = '0c0e3051f131929182e2c023b9537f8b1c68adfe'
PROFILE_SHA = '9c49146ae2bd3c5a32c686ad1705fd912391af4a39db0d63e9dc353640d304fa'
ANCHOR_SHA = '73bb894fc75e18dddf76e3327f3bf48e161b7dd8b30fbe4949c3ac7318c29567'
ANCHOR_TEXT = 'I understand the problem. Give me the facts in the order they happened, not the order they were explained later. Then we can decide what actually matters.'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def write(p, obj):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

CASES=[]
def add(i,category,text,instruct='',**kw):
    CASES.append(dict(id=i,category=category,text=text,instruct=instruct,seed=962200+len(CASES),temperature=.42,top_k=40,top_p=.95,rep_penalty=1.05,mode='profile',chunked=False,**kw))

ordinary=[
('O01','I left the spare charger in the second drawer. The blue cable is beside it.'),
('O02','The appointment is on Thursday at three. We should leave the house at half past two.'),
('O03','We need rice, apples, dish soap, and a carton of milk. There is already bread in the freezer.'),
('O04','Turn left after the bakery, then take the second right. The entrance is next to the bicycle rack.'),
('O05','The small box goes on the lower shelf. Put the empty folder beside the printer.'),
('O06','Yes, I checked both windows. The one above the sink was open, so I closed it.'),
('O07','The washing machine has finished. I will move the towels after I clear the basket.'),
('O08','No, that is the older receipt. The new one is underneath the envelope.'),
('O09','There are twelve items in the box. Three are red, four are green, and five are blue.'),
('O10','Is the back door locked, or should I check it before we leave?'),
('O11','Which shelf did you mean: the one above the desk, or the one beside the window?'),
('O12','The first switch controls the hall light. The second switch does not control anything in this room.')]
for i,t in ordinary:add(i,'ordinary',t)
phonetics=[
('F01','sibilants','Six silver scissors sat beside seven sealed jars. Susan sorted the sheets, then shut the screen.'),
('F02','plosives','Pack the blue tin, tape the top, and put it back. Do not drop the paper cup.'),
('F03','fricatives','Five thin threads fell through the vent. Heather left the other fabric over the smooth shelf.'),
('F04','affricates','George chose the orange jacket. Rachel checked each edge before attaching the badges.'),
('F05','clusters','The sixth street intersects the twelfth. Bring the scratched brass clasp and three crisp receipts.'),
('F06','vowels_diphthongs','The green room has two blue stools. Ray found five brown coins near the toy boat.'),
('F07','nasals_repetition','Mina named nine new rooms. Many mornings, Nina hums the same small tune near the window.'),
('F08','numbers_dates','The reference number is four zero seven nine two. The delivery is due on November twenty sixth at nine fifteen.'),
('F09','acronyms','The U S B cable is beside the H D M I adapter. The P D F lists three I D numbers.'),
('F10','proper_names','Amelia, Tobias, and Penelope will meet in Birmingham. Eleanor is bringing the folders from Montgomery.'),
('F11','technical_terms','The spectrogram shows several harmonics. Compare the logarithmic scale with the autocorrelation estimate and the interpolation error.'),
('F12','boundaries','An ice tray is not a nice tray. A gray tape roll can look like a great ape from a distance.'),
('F13','quotations_parentheticals','She said, "Leave the small box here," and then pointed to the shelf. The large box, not the damaged one, goes outside.'),
('F14','fragments_alternation','Not yet. After the measurements. One small adjustment, then recalibration. Done? Almost. Check the exceptionally narrow gap again.'),
('F15','foreign_names_separate','Siobhan and Nguyen are meeting Saoirse near the station. Please ask each person how they pronounce their name.')]
for i,c,t in phonetics:add(i,'phonetic_'+c,t)
shared='The spare key is in the blue box. We can use the side door, then check the lights before we leave.'
states={
'S01':('dry_amusement','Dry understated amusement. A restrained intelligent smile in the timing, never cute, breathy, or theatrical.'),
'S02':('controlled_irritation','Controlled irritation. Firmer consonants, contained force, clipped patience, no shouting and no theatrical anger.'),
'S03':('reassuring','Calm direct reassurance. Grounded warmth, measured pace, steady clarity, never sentimental or whispery.'),
'S04':('focused_urgency','Focused urgency. Purposeful faster pace, sharper emphasis, controlled intensity, no panic.'),
'S05':('quiet','Speak quietly to someone nearby. Remain supported, clearly articulated, and conversational. Do not whisper or exaggerate intimacy.'),
'S06':('assertive','Make a clear practical decision with firm ordinary projection. No shouting, aggression, or theatrical authority.'),
'S07':('neutral','')}
for i,(c,d) in states.items():add(i,'state_'+c,shared,d)
add('H01','humor','The label says this drawer is organized. That was an optimistic choice of label. The tape is under the stapler.','Notice the mild absurdity without performing a punchline; briefly amused, then return to the practical answer.')
add('T01','transition_amusement','The folders are in alphabetical order. At least, that was the plan. Someone has filed the lunch menu under urgent. All right. The invoices are in the second folder, and the blank forms are behind them. I will put the menu back in the kitchen.','Begin neutral and practical. Notice the small absurdity at the lunch menu, briefly contain an impulse to laugh, allow dry amusement, then explain the filing and return naturally to neutral. One continuing speaker, no exaggerated acting.')
add('T02','transition_urgency','The water is staying in the tray. Wait, it is reaching the edge. Move the notebook to the other table, please. I will lift the tray while you put the towel underneath. There. Nothing has spilled on the floor. We can leave the towel here until the tray is dry.','Begin calm. Concern emerges when the water reaches the edge. Give the practical directions with focused urgency, no panic. Once the tray is safe, release the tension and reassure the listener. No abrupt actor changes.')
add('T03','transition_irritation','I put the labels beside the empty boxes. No, the labels are not meant to cover the handles. Leave the handles clear so we can carry the boxes. Yes, like that. Thank you. We can finish the other two in the same way.','Start neutral. A brief irritation appears at the misplaced labels; contain it while giving a precise correction. Release the tension when the listener understands. Ordinary conversation, never a caricature of anger.')
add('L01','short_paragraph','To replace the paper, pull out the lower tray and remove any bent sheets. Align the new stack against the left guide, but do not force it beneath the clips. Slide the tray back slowly. The small indicator should stop flashing after the cover is closed.')
minute='Here is how I sorted the storage cupboard. I started with the things we use every day and put those on the middle shelf, where they are easy to reach. The cleaning cloths are together in the shallow basket. Spare bulbs and batteries are in separate boxes, with the labels facing forward. I left the top shelf for things we only need occasionally. The empty jars are at the back, and the folded bags are beside them. Nothing heavy is above shoulder height. There is one box I have not sorted yet because I cannot tell which cables still belong to something. I put that box on the floor rather than mixing its contents with the cables we actually use. Before we move anything else, we should check the list taped inside the door and add whatever is missing.'
add('L02','one_minute_single_pass',minute)
add('L03','one_minute_production_chunking',minute)
CASES[-1]['chunked']=True
longtext=' '.join([
'The purpose of this explanation is to make the packing process easy to repeat. We have three kinds of things to pack: flat paper items, small rigid objects, and objects that need padding. Begin by clearing enough space to keep the finished boxes separate from the empty ones. Put the labels, tape, scissors, and marker within reach. Check that the boxes are dry and that the corners have not been crushed. A damaged box should be set aside before anything is put into it.',
'For the paper items, make a stack that fits without bending the corners. Place the stack in a folder, close the flap, and write a brief description on the outside. Do not rely on the color of the folder to tell you what is inside. Two folders may be the same color and contain different things. If a stack is too thick for one folder, divide it into two smaller stacks and number the folders so their order is clear.',
'For the small rigid objects, first check whether any pieces can become detached. Put loose pieces into a small bag and keep that bag with the main object. Wrap anything with a sharp edge so it cannot scratch the neighboring items. Place heavier objects at the bottom of the box. Fill the empty spaces with folded paper, but leave enough room to close the lid without pressing it down. The box should feel stable when it is lifted carefully from both sides.',
'Objects that need padding should be packed separately from the paper folders. Put a layer of padding in the bottom of the box, add the object, and then fill the space around it. The aim is to prevent movement, not to squeeze the object tightly. Before sealing the box, gently check whether the contents shift. If they do, add a little more padding and check again. Keep the original label visible until you have written the new one.',
'Finally, write a label that names the contents and the room where the box belongs. Put one label on top and another on the side, so a label remains visible when the boxes are stacked. Record the number of each finished box on the list. When all the boxes are ready, compare the list with the actual boxes rather than assuming the count is correct. Leave the marker and the remaining labels out until the last check is complete. That is the whole process. The important parts are keeping the categories separate, preventing loose movement, and making the contents easy to identify later.'
])
add('L04','multi_minute_production_chunking',longtext)
CASES[-1]['chunked']=True
prosody_text='The first folder contains the old version. The second folder contains the corrected version. Please send the second one, not both.'
add('P01','prosody_baseline',prosody_text)
add('P02','prosody_selective_focus',prosody_text,'Explain the practical distinction. Keep ordinary pitch movement restrained, but make old, corrected, and second consequential through timing and selective emphasis. Do not stress every word.')
add('P03','prosody_uniform_emphasis_control',prosody_text,'Give every phrase similar emphasis and similar timing. Use a uniform, clearly articulated explanatory cadence.')
add('P04','prosody_phrase_ending_control',prosody_text,'Use a slight upward contour at the end of each statement, without changing the words. Keep the same ordinary speaking person.')
add('R01','repeat_seed_control',ordinary[0][1])
CASES[-1]['seed']=CASES[0]['seed']
add('R02','temperature_low',ordinary[0][1])
CASES[-1]['seed']=CASES[0]['seed'];CASES[-1]['temperature']=.25
add('R03','temperature_high',ordinary[0][1])
CASES[-1]['seed']=CASES[0]['seed'];CASES[-1]['temperature']=.65
add('R04','reference_audio_xvector',ordinary[0][1])
CASES[-1]['seed']=CASES[0]['seed'];CASES[-1]['mode']='reference_xvector'
add('R05','reference_audio_icl',ordinary[0][1])
CASES[-1]['seed']=CASES[0]['seed'];CASES[-1]['mode']='reference_icl'


def split_text(text,limit=230):
    out=[];cur=''
    for s in re.split(r'(?<=[.!?])\s+',text.strip()):
        if cur and len(cur)+1+len(s)>limit:out.append(cur);cur=s
        else:cur=(cur+' '+s).strip()
    if cur:out.append(cur)
    return out


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    src=ROOT/'source';src.mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT/'source.zip') as z:
        for name,expected in [('MARI_VOICE_V1_PROFILE.bin',PROFILE_SHA),('MARI_VOICE_V1_ANCHOR.wav',ANCHOR_SHA)]:
            candidates=[i for i in z.namelist() if pathlib.PurePosixPath(i).name==name]
            if len(candidates)!=1:raise RuntimeError('Ambiguous or missing input artifact')
            p=src/name;p.write_bytes(z.read(candidates[0]))
            if sha(p)!=expected:raise RuntimeError('Input hash mismatch')
            shutil.copy2(p,OUT/name)
    write(OUT/'suite.json',CASES)
    shutil.copy2(__file__,OUT/'runner.py')
    model=ROOT/'model'
    manifest=[]
    for p in sorted(model.rglob('*')):
        if p.is_file() and '.cache' not in p.parts:
            manifest.append(dict(path=str(p.relative_to(model)),size_bytes=p.stat().st_size,sha256=sha(p)))
    write(OUT/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),cpu_count=os.cpu_count(),engine_revision=ENGINE_REV,engine_sha256=sha(ROOT/'engine/qwen_tts'),model='Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',model_revision=MODEL_REV,model_files=manifest,profile_sha256=PROFILE_SHA,anchor_sha256=ANCHOR_SHA,git_commit=os.environ.get('GITHUB_SHA'),run_id=os.environ.get('GITHUB_RUN_ID'),shard=os.environ.get('SHARD'),certification='NOT_CERTIFIED',native_owners_modified=False))


def generate(shard,shards):
    prepare()
    records=[]
    for index,c in enumerate(CASES):
        if index%shards!=shard:continue
        dst=OUT/c['id'];dst.mkdir(exist_ok=True)
        record=dict(c,source_profile_sha256=PROFILE_SHA,source_anchor_sha256=ANCHOR_SHA,renderer_model_revision=MODEL_REV,engine_revision=ENGINE_REV,commands=[],status='started',human_identity_judgment=None)
        start=time.monotonic()
        try:
            chunks=split_text(c['text']) if c['chunked'] else [c['text']]
            arrays=[];rate=None
            for n,chunk in enumerate(chunks):
                p=dst/f'chunk_{n:02d}.wav'
                cmd=[str((ROOT/'engine/qwen_tts').resolve()),'-d',str((ROOT/'model').resolve()),'-l','English','--text',chunk,'--seed',str(c['seed']+n),'--temperature',str(c['temperature']),'--top-k',str(c['top_k']),'--top-p',str(c['top_p']),'--rep-penalty',str(c['rep_penalty']),'--max-duration','120','-j4','-o',str(p.resolve())]
                if c['mode']=='profile':cmd+=['--load-voice',str((ROOT/'source/MARI_VOICE_V1_PROFILE.bin').resolve()),'--xvector-only']
                elif c['mode']=='reference_xvector':cmd+=['--ref-audio',str((ROOT/'source/MARI_VOICE_V1_ANCHOR.wav').resolve()),'--xvector-only']
                elif c['mode']=='reference_icl':cmd+=['--ref-audio',str((ROOT/'source/MARI_VOICE_V1_ANCHOR.wav').resolve()),'--ref-text',ANCHOR_TEXT]
                else:raise ValueError('Unknown conditioning mode')
                if c['instruct']:cmd+=['--instruct',c['instruct']]
                record['commands'].append(dict(argv=cmd,intended_text=chunk,seed=c['seed']+n))
                with open(dst/f'chunk_{n:02d}.log','w') as log:
                    result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=480,env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                if result.returncode:raise RuntimeError(f'Native renderer exit {result.returncode}; no fallback')
                x,sr=sf.read(p,dtype='float32',always_2d=True)
                if x.shape[1]!=1 or not len(x) or not np.isfinite(x).all():raise RuntimeError('Invalid audio shape or samples')
                if rate is not None and rate!=sr:raise RuntimeError('Sample rate changed between chunks')
                rate=sr;arrays.append(x[:,0])
                if n<len(chunks)-1:arrays.append(np.zeros(round(sr*.1),dtype=np.float32))
            wave=np.concatenate(arrays);target=dst/'audio.wav'
            sf.write(target,wave,rate,subtype='PCM_16')
            record.update(status='rendered_not_perceptually_certified',file=str(target.relative_to(OUT)),sha256=sha(target),sample_rate_hz=rate,channels=1,duration_s=len(wave)/rate,peak_abs=float(np.max(np.abs(wave))),clip_fraction=float(np.mean(np.abs(wave)>=.9999)),chunks=len(chunks),join_silence_s=.1 if len(chunks)>1 else 0,continuous_neural_state_between_chunks=False if len(chunks)>1 else None)
        except Exception as e:
            record.update(status='render_failed',error=str(e),traceback=traceback.format_exc())
        record['wall_seconds']=time.monotonic()-start
        write(dst/'record.json',record);records.append(record);write(OUT/'generation_records.json',records)
    write(OUT/'generation_summary.json',dict(shard=shard,total=len(records),rendered=sum(x['status']=='rendered_not_perceptually_certified' for x in records),failures=[x['id'] for x in records if x['status']=='render_failed'],certification='NOT_CERTIFIED'))


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?",text.lower())

def wer(reference,hypothesis):
    a,b=tokens(reference),tokens(hypothesis);row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]/max(1,len(a))


def evaluate():
    records=json.loads((OUT/'generation_records.json').read_text())
    observer={};asr=None;encoder=None;anchor_embedding=None
    try:
        from faster_whisper import WhisperModel
        from huggingface_hub import HfApi,snapshot_download
        repo='Systran/faster-whisper-small.en';rev=HfApi().model_info(repo).sha
        folder=snapshot_download(repo_id=repo,revision=rev)
        asr=WhisperModel(folder,device='cpu',compute_type='int8',cpu_threads=4,num_workers=1)
        observer['asr']=dict(repository=repo,revision=rev,beam_size=5,condition_on_previous_text=False,vad_filter=False,word_timestamps=True,interpretation='Automated transcription; disagreement is a review flag, not proof of pronunciation failure.')
    except Exception as e:observer['asr_error']=str(e)
    try:
        import torch,torchaudio
        from speechbrain.inference.speaker import EncoderClassifier
        torch.set_num_threads(4)
        encoder=EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',savedir=str(ROOT/'ecapa'),run_opts={'device':'cpu'})
        observer['speaker']=dict(repository='speechbrain/spkrec-ecapa-voxceleb',files=[dict(name=str(p.relative_to(ROOT/'ecapa')),sha256=sha(p)) for p in (ROOT/'ecapa').rglob('*') if p.is_file()],interpretation='ECAPA cosine is speaker-consistency evidence only; no Mari threshold or certification.')
        def embed(p):
            x,s=sf.read(p,dtype='float32');t=torch.from_numpy(x).unsqueeze(0)
            if s!=16000:t=torchaudio.functional.resample(t,s,16000)
            with torch.inference_mode():v=encoder.encode_batch(t).reshape(-1).cpu().numpy()
            return v/np.linalg.norm(v)
        anchor_embedding=embed(OUT/'MARI_VOICE_V1_ANCHOR.wav')
        observer['anchor_embedding']=anchor_embedding.tolist()
    except Exception as e:observer['speaker_error']=str(e);encoder=None
    assessments=[]
    for r in records:
        if r['status']!='rendered_not_perceptually_certified':continue
        p=OUT/r['file'];a=dict(id=r['id'],audio_sha256=sha(p),reference_text=r['text'],human_judgment=None)
        if asr:
            try:
                segments,_=asr.transcribe(str(p),beam_size=5,condition_on_previous_text=False,vad_filter=False,word_timestamps=True)
                segments=list(segments);transcript=' '.join(s.text.strip() for s in segments)
                a.update(asr_text=transcript,raw_normalized_word_error_rate=wer(r['text'],transcript),asr_words=[dict(start=w.start,end=w.end,word=w.word,probability=w.probability) for s in segments for w in (s.words or [])])
            except Exception as e:a['asr_error']=str(e)
        if encoder:
            try:
                v=embed(p);a['ecapa_embedding']=v.tolist();a['ecapa_cosine_to_anchor']=float(v@anchor_embedding)
                x,sr=sf.read(p,dtype='float32');windows=[]
                if len(x)/sr>=20:
                    for start in np.arange(0,len(x)/sr-4,10):
                        q=ROOT/'window.wav';sf.write(q,x[int(start*sr):int(min(start+15,len(x)/sr)*sr)],sr)
                        w=embed(q);windows.append(dict(start_s=float(start),end_s=min(float(start)+15,len(x)/sr),ecapa_cosine_to_anchor=float(w@anchor_embedding)))
                a['speaker_windows']=windows
            except Exception as e:a['speaker_error']=str(e)
        assessments.append(a);write(OUT/'automatic_assessments.json',assessments)
    write(OUT/'observers.json',observer)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['generate','evaluate']);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=3);a=ap.parse_args()
    if a.action=='generate':generate(a.shard,a.shards)
    else:evaluate()
