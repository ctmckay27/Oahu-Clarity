"""Build a small deterministic spontaneous-conversation performance bank from AMI ES2002a.

The references are performance/naturalness evidence only. They never select or
define Mari's speaker identity.
"""
from __future__ import annotations
import argparse,hashlib,json,math,pathlib,re,struct,requests
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

NITE="http://nite.sourceforge.net/"
NID="{"+NITE+"}id"
BASE="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio"
LICENSE="CC BY 4.0"
ATTRIBUTION="AMI Meeting Corpus, University of Edinburgh, https://groups.inf.ed.ac.uk/ami/corpus/"

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def _word_index(word_id: str) -> int:
    m=re.search(r"words(\d+)$",word_id)
    if not m: raise ValueError("unrecognized AMI word id: "+word_id)
    return int(m.group(1))

def _parse_header(session, url):
    res=session.get(url,headers={"Range":"bytes=0-4095"},timeout=60)
    res.raise_for_status()
    if res.status_code!=206: raise ValueError("AMI audio range request ignored")
    b=res.content
    if b[:4]!=b"RIFF" or b[8:12]!=b"WAVE": raise ValueError("AMI source is not RIFF/WAVE")
    pos=12;fmt=None;offset=None
    while pos+8<=len(b):
        kind=b[pos:pos+4];size=struct.unpack_from("<I",b,pos+4)[0]
        if kind==b"fmt ": fmt=struct.unpack_from("<HHIIHH",b,pos+8)
        if kind==b"data": offset=pos+8;break
        pos+=8+size+(size%2)
    if fmt is None or offset is None or fmt[0]!=1 or fmt[1]!=1 or fmt[5]!=16:
        raise ValueError("AMI source PCM format not admitted")
    return {"sample_rate":fmt[2],"data_offset":offset,"etag":res.headers.get("ETag"),"header_sha256":sha_bytes(b)}

def _load_words(path):
    root=ET.parse(path).getroot()
    words={}
    for w in root:
        wid=w.get(NID)
        if not wid or not w.get("starttime") or not w.get("endtime"): continue
        text=(w.text or "").strip()
        if not text: continue
        if w.tag.endswith("vocalsound") or w.tag.endswith("gap") or w.tag.endswith("disfmarker"): continue
        words[_word_index(wid)]={
            "id":wid,"text":text,"start":float(w.get("starttime")),"end":float(w.get("endtime"))
        }
    return words

def _candidates(ann, agent):
    words=_load_words(ann/f"words/ES2002a.{agent}.words.xml")
    root=ET.parse(ann/f"dialogueActs/ES2002a.{agent}.dialog-act.xml").getroot()
    rows=[]
    ns={"n":NITE}
    for act in root:
        child=act.find("n:child",ns)
        if child is None: continue
        ids=re.findall(r"id\(([^)]+)\)",child.get("href",""))
        if not ids: continue
        try:
            lo=_word_index(ids[0]);hi=_word_index(ids[-1])
        except ValueError:
            continue
        seq=[words[i] for i in range(lo,hi+1) if i in words]
        if len(seq)<4 or len(seq)>18: continue
        start=min(x["start"] for x in seq);end=max(x["end"] for x in seq);dur=end-start
        if not 1.5<=dur<=5.2 or start<60: continue
        text=" ".join(x["text"] for x in seq)
        if len(text)>220: continue
        lexical=[re.sub(r"[^a-z']","",x["text"].lower()) for x in seq]
        if sum(bool(x and x not in {"um","uh","erm","hmm","mm"}) for x in lexical)<4: continue
        # Fixed target geometry: medium-length complete conversational turns.
        score=abs(dur-2.8)+0.045*abs(len(seq)-9)
        rows.append({"agent":agent,"start":start,"end":end,"duration":dur,"text":text,
                     "word_ids":[x["id"] for x in seq],"score":score})
    if not rows: raise ValueError("no eligible AMI reference for agent "+agent)
    rows.sort(key=lambda x:(x["score"],x["start"],x["text"]))
    return rows[0]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("root");a=ap.parse_args()
    root=pathlib.Path(a.root);ann=root/"annotations";out=root/"performance_refs";out.mkdir(parents=True,exist_ok=True)
    meeting=next(x for x in ET.parse(ann/"corpusResources/meetings.xml").getroot() if x.get("observation")=="ES2002a")
    channels={x.get("nxt_agent"):int(x.get("channel")) for x in meeting}
    session=requests.Session();refs=[]
    for agent in ["A","B","D"]:
        cand=_candidates(ann,agent);ch=channels[agent]
        url=f"{BASE}/ES2002a.Headset-{ch}.wav"
        header=_parse_header(session,url);sr=header["sample_rate"]
        first=round(max(0,cand["start"]-.08)*sr);last=round((cand["end"]+.08)*sr)
        lo=header["data_offset"]+2*first;hi=header["data_offset"]+2*last-1
        res=session.get(url,headers={"Range":f"bytes={lo}-{hi}"},timeout=60);res.raise_for_status()
        if res.status_code!=206 or len(res.content)!=2*(last-first): raise ValueError("AMI excerpt range incomplete")
        if header["etag"] and res.headers.get("ETag")!=header["etag"]: raise ValueError("AMI source changed during extraction")
        y=np.frombuffer(res.content,dtype="<i2").astype(np.float32)/32768.0
        g=math.gcd(sr,24000);y24=resample_poly(y,24000//g,sr//g).astype(np.float32)
        peak=float(np.max(np.abs(y24)))
        if peak>=.999: y24*=.98/max(peak,1e-9)
        p=out/f"AMI_ES2002a_{agent}.wav";sf.write(p,y24,24000,subtype="PCM_16")
        ref={
            "audio":str(p),"text":cand["text"],"sha256":sha_file(p),
            "role":"spontaneous human conversational performance donor; performance conditioning only",
            "license":LICENSE,"attribution":ATTRIBUTION,"source_url":url,
            "donor_id":f"AMI_ES2002a_{agent}",
            "identity_authority":"PERFORMANCE_ONLY_NOT_MARI_IDENTITY",
        }
        refs.append(ref)
        (out/f"AMI_ES2002a_{agent}.provenance.json").write_text(json.dumps({
            "reference":ref,"meeting":"ES2002a","agent":agent,"channel":ch,
            "source_excerpt_s":[cand["start"],cand["end"]],"extracted_samples":[first,last],
            "source_sample_rate":sr,"source_pcm_sha256":sha_bytes(res.content),
            "source_content_range":res.headers.get("Content-Range"),"source_etag":header["etag"],
            "source_header_sha256":header["header_sha256"],"annotation_word_ids":cand["word_ids"],
            "selection_rule":"one deterministic medium-length lexical dialogue act per A/B/D; no Mari-similarity or output-quality selection",
        },indent=2)+"\n")
        print(agent,round(cand["duration"],3),cand["text"],flush=True)
    (out/"REFERENCE_BANK.json").write_text(json.dumps({
        "schema":"mari-performance-reference-bank/1.0",
        "meeting":"ES2002a","license":LICENSE,"attribution":ATTRIBUTION,
        "identity_rule":"references are performance evidence only and cannot select/define Mari speaker identity",
        "references":refs,
    },indent=2)+"\n")

if __name__=="__main__": main()
