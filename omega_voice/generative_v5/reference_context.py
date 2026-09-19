"""Bound acoustic history to the selected anchor instead of recursive cloning."""
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from .native import anchor_reference
from .session import durable_json
from ..causal_v4.renderer import sha
from ..causal_v4.runtime import ANCHOR

def anchored_history(root, previous, out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    anchor=anchor_reference(root)
    if previous['source_carrier_sha256']!=ANCHOR or sha(previous['audio'])!=previous['sha256']:
        raise ValueError('unverified previous native carrier')
    parts=[];sources=[];cursor=0
    for ref in [anchor,previous]:
        y,sr=sf.read(ref['audio'],dtype='int16')
        if sr!=24000 or y.ndim!=1:raise ValueError('invalid context format')
        sources.append(dict(ref,start_sample=cursor,end_sample=cursor+len(y)))
        parts.append(y);cursor+=len(y)
    if cursor>30*24000:raise ValueError('anchor and complete prior unit exceed verified context budget; explicit source-unit selection required')
    sf.write(out,np.concatenate(parts),24000,subtype='PCM_16')
    reference={'audio':str(out),'text':anchor['text']+' '+previous['text'],
      'sha256':sha(out),'source_carrier_sha256':ANCHOR,
      'role':'selected fixed identity anchor followed by preceding native carrier; no future text, waveform modification, or state reset'}
    durable_json(out.with_suffix('.context.json'),{'role':'diagnostic composite conditioning, not delivered performance',
      'sources':sources,'reference':reference,'method':'exact PCM concatenation','implementation_sha256':sha(__file__)})
    return reference

