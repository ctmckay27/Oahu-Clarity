import json,pytest
from omega_voice.generative_v5.scene_language import parse_events

def test_scene_quote_is_required_and_renderer_never_receives_it():
 scene='She conceals her fear from her listener.'
 raw=json.dumps({'events':[{'kind':'mask','at_word':0,'source_quote':scene,'dimension':'fear','internal':.8,'display':.1,'effort':.8}]})
 x=parse_events(raw,scene,'We can handle this.')
 assert x['plan']['renderer_instruction'] is None
 assert x['plan']['final_state']['mask']['active']
 assert x['plan']['final_state']['listener_model']['monitoring']>=.8

def test_untraceable_changes_and_extra_renderer_fields_fail():
 base={'kind':'thought','at_word':0,'mode':'searching','source_quote':'She searches for the name.'}
 with pytest.raises(ValueError,match='ungrounded'):parse_events(json.dumps({'events':[base]}),'She knows the name.','It is here.')
 base['voice']='breathy'
 with pytest.raises(ValueError,match='uncontrolled'):parse_events(json.dumps({'events':[base]}),'She searches for the name.','It is here.')
