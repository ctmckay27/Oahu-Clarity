"""Performance-reference provenance and bank validation."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any, Dict, List
import soundfile as sf

IDENTITY_AUTHORITY="PERFORMANCE_ONLY_NOT_MARI_IDENTITY"
SCHEMA="mari-performance-reference-bank/1.0"

def sha256(path: str|Path) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def validate_performance_reference(ref: Dict[str,Any]) -> Dict[str,Any]:
    required={"audio","text","sha256","role","license","attribution","source_url","donor_id","identity_authority"}
    if set(ref)!=required:
        raise ValueError("performance reference fields mismatch")
    if ref["identity_authority"]!=IDENTITY_AUTHORITY:
        raise ValueError("donor may not become Mari identity authority")
    p=Path(ref["audio"])
    if not p.is_file() or sha256(p)!=ref["sha256"]:
        raise ValueError("performance reference bytes mismatch")
    y,sr=sf.read(p,dtype="float32",always_2d=True)
    if sr!=24000 or y.shape[1]!=1 or len(y)<12000:
        raise ValueError("performance reference must be mono 24k and >=0.5 s")
    if not ref["text"].strip() or not ref["license"].strip() or not ref["attribution"].strip() or not ref["source_url"].strip():
        raise ValueError("performance reference provenance incomplete")
    return ref

def load_reference_bank(path: str|Path) -> List[Dict[str,Any]]:
    doc=json.loads(Path(path).read_text())
    if doc.get("schema")!=SCHEMA:
        raise ValueError("reference bank schema mismatch")
    refs=doc.get("references")
    if not isinstance(refs,list) or not refs:
        raise ValueError("reference bank empty")
    ids=set()
    for ref in refs:
        validate_performance_reference(ref)
        if ref["donor_id"] in ids:
            raise ValueError("duplicate donor id")
        ids.add(ref["donor_id"])
    return refs
