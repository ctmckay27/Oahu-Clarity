import pytest
from omega_voice.generative_v5.text_release import write_release,verify_consumption,read_release

def test_future_token_cannot_be_consumed_early_or_omitted(tmp_path):
 packet=tmp_path/'source.mrl';write_release(packet,[10,20,30],[0,0,18,18])
 audit=tmp_path/'audit.csv';audit.write_text('token_index,feed_frame\n0,-1\n1,0\n2,18\n3,19\n')
 assert verify_consumption(packet,audit)['no_early_consumption']
 audit.write_text('token_index,feed_frame\n0,-1\n1,0\n2,17\n3,19\n')
 with pytest.raises(ValueError,match='future text'):verify_consumption(packet,audit)
 audit.write_text('token_index,feed_frame\n0,-1\n1,0\n')
 with pytest.raises(ValueError,match='every scheduled'):verify_consumption(packet,audit)

def test_release_malformed_and_reverse_chronology_fail_closed(tmp_path):
 packet=tmp_path/'source.mrl'
 with pytest.raises(ValueError,match='nonmonotonic'):write_release(packet,[10,20],[0,18,2])
 write_release(packet,[10,20],[0,18,18]);packet.write_bytes(packet.read_bytes()[:-1])
 with pytest.raises(ValueError,match='invalid release'):read_release(packet)
