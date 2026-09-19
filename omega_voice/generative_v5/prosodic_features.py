"""Independent word acoustics; no renderer state, word identity, or style labels."""
import re,pathlib,hashlib,importlib.metadata
import numpy as np,soundfile as sf,parselmouth,cmudict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

NAMES=['log_duration_per_phone','duration_per_phone_relative','local_duration_relative','voiced_fraction','pitch_median_relative_st','pitch_high_relative_st','pitch_range_st','pitch_movement_st','level_relative_dB','local_level_relative_dB','log_phone_count','syllable_count','primary_stress_count','vowel_phone_fraction','function_word','lexicon_fallback','position','preceding_gap','following_gap','first_word','last_word','log_duration']
_LEX=cmudict.dict()

def provenance():
 p=pathlib.Path(cmudict.__file__).parent/'data/cmudict.dict'
 return {'implementation_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'CMUdict':{'version':importlib.metadata.version('cmudict'),'data_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},'parselmouth':parselmouth.__version__,'features':NAMES,'independence':'No EmphaClass output, speaker id, token identity, causal state, renderer control, or Mari diagnostic label enters the features.'}

def extract(path,alignment):
 y,sr=sf.read(path,dtype='float64')
 if y.ndim!=1 or not np.isfinite(y).all():raise ValueError('finite mono acoustic input required')
 sound=parselmouth.Sound(y,sampling_frequency=sr);pitch=sound.to_pitch_ac(time_step=.005,pitch_floor=70,pitch_ceiling=500);hz=pitch.selected_array['frequency'];ts=pitch.xs();voiced=hz>0;ref=float(np.median(hz[voiced]))if voiced.any()else 150.;whole=max(1e-8,float(np.sqrt(np.mean(y*y))));rows=[]
 for i,w in enumerate(alignment['words']):
  start,end=w['start'],w['end'];word=w['word'].lower();phones=_LEX.get(word);fallback=phones is None;phones=phones[0]if phones else ['X']*max(1,len(word));vowels=[p for p in phones if p[-1:].isdigit()];nphone=max(1,len(phones));syll=max(1,len(vowels))if not fallback else max(1,len(re.findall(r'[aeiouy]+',word)));mask=(ts>=start)&(ts<=end)&voiced;f=hz[mask];part=y[round(start*sr):round(end*sr)];rms=max(1e-8,float(np.sqrt(np.mean(part*part))));duration=max(.001,end-start);count=len(f)
  rows.append({'index':i,'word':word,'start':start,'end':end,'duration':duration,'per_phone':duration/nphone,'phone_count':nphone,'syllables':syll,'stress':sum(p.endswith('1')for p in vowels),'vowel_fraction':len(vowels)/nphone,'fallback':fallback,'voiced_fraction':min(1.,count*.005/duration),'pitch50':float(12*np.log2(np.median(f)/ref))if count else 0.,'pitch90':float(12*np.log2(np.percentile(f,90)/ref))if count else 0.,'pitchrange':float(12*np.log2(np.percentile(f,90)/np.percentile(f,10)))if count else 0.,'movement':float(12*np.log2(np.median(f[-max(1,count//3):])/np.median(f[:max(1,count//3)])))if count else 0.,'level':20*np.log10(rms/whole)})
 median=max(.001,float(np.median([w['per_phone']for w in rows])));features=[]
 for i,w in enumerate(rows):
  neighbors=[a for j,a in enumerate(rows[max(0,i-2):min(len(rows),i+3)],max(0,i-2))if j!=i];local=max(.001,float(np.median([a['per_phone']for a in neighbors])))if neighbors else median;level=float(np.median([a['level']for a in neighbors]))if neighbors else 0.;before=w['start']-rows[i-1]['end']if i else w['start'];after=rows[i+1]['start']-w['end']if i+1<len(rows)else len(y)/sr-w['end']
  features.append([np.log(w['per_phone']),np.log(w['per_phone']/median),np.log(w['per_phone']/local),w['voiced_fraction'],w['pitch50'],w['pitch90'],w['pitchrange'],w['movement'],w['level'],w['level']-level,np.log(w['phone_count']),w['syllables'],w['stress'],w['vowel_fraction'],float(w['word']in ENGLISH_STOP_WORDS),float(w['fallback']),i/max(1,len(rows)-1),max(0,before),max(0,after),float(i==0),float(i==len(rows)-1),np.log(w['duration'])])
 x=np.asarray(features,dtype=np.float64)
 if x.shape!=(len(rows),len(NAMES))or not np.isfinite(x).all():raise ValueError('invalid acoustic feature matrix')
 return x,rows
