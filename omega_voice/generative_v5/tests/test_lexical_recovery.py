"""Reject repairs that would change correctly spoken or unrelated content."""
from omega_voice.generative_v5.lexical_recovery import proposal

def test_only_confirmed_past_stop_omission_is_eligible():
 target='I checked the door. It is open.'
 assert proposal(target,'I check the door. It is open.')['renderer_text']=='I chekt the door. It is open.'
 assert proposal(target,target)is None
 assert proposal(target,'I checked a door. It is open.')is None
 assert proposal(target,'I check a door. It is open.')is None
 assert proposal(target,'I checked the door.')is None
 assert proposal('I check the door.','I checked the door.')is None

def test_only_the_observed_occurrence_is_reencoded():
 target='She checked it. I checked it too.'
 assert proposal(target,'She checked it. I check it too.')['renderer_text']=='She checked it. I chekt it too.'
