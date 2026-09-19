import json
import numpy as np
import pytest
from omega_voice.generative_v5.session import PerformanceSession,durable_json
from omega_voice.causal_v4.renderer import sha
from omega_voice.causal_v4.runtime import digest

def test_crash_between_receipt_and_state_recovers_once_and_rejects_changed_evidence(tmp_path):
    bank=tmp_path/'bank.npz';np.savez(bank,directions=np.zeros((1,29,2048)))
    s=PerformanceSession(tmp_path,tmp_path/'session',None,bank,sha(bank))
    turn=s.directory/'turn1';turn.mkdir();receipt=turn/'receipt.json';durable_json(receipt,{'verified_audio':'example'})
    saved={'state':{'turn':1},'receipt':'turn1/receipt.json','receipt_sha256':sha(receipt)}
    pending={'parent_record_hash':digest(None),'next':saved,'receipt':saved['receipt'],'receipt_sha256':sha(receipt)}
    durable_json(s.directory/'pending_commit.json',pending)
    s.recover_commit();assert json.loads(s.state_path.read_text())==saved
    s.recover_commit();assert json.loads(s.state_path.read_text())==saved
    durable_json(s.directory/'pending_commit.json',pending);receipt.write_text('{}')
    with pytest.raises(RuntimeError,match='changed'):s.recover_commit()

def test_stale_commit_cannot_overwrite_another_completed_turn(tmp_path):
    bank=tmp_path/'bank.npz';np.savez(bank,directions=np.zeros((1,29,2048)))
    s=PerformanceSession(tmp_path,tmp_path/'session',None,bank,sha(bank))
    receipt=s.directory/'receipt.json';durable_json(receipt,{'verified_audio':'example'})
    durable_json(s.state_path,{'state':{'turn':2}})
    durable_json(s.directory/'pending_commit.json',{'parent_record_hash':digest(None),'next':{'state':{'turn':1}},'receipt':'receipt.json','receipt_sha256':sha(receipt)})
    with pytest.raises(RuntimeError,match='conflicts'):s.recover_commit()
    assert json.loads(s.state_path.read_text())['state']['turn']==2
