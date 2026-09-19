import copy,unittest
from omega_voice.causal_v4.runtime import compile_scene,verify_plan

TEXT='The package is beside the window. I will bring it to you.'
def e(kind,**kw):return dict(id=kind,kind=kind,at_word=0,source={'text':'explicit causal test scene'},**kw)
def mismatch(**kw):return e('contrast',expected='The package is beside the door.',observed='The package is beside the window.',perspective='listener',**kw)
def plan(events,prior=None,policy='contrast_focus_v7'):return compile_scene(TEXT,{'events':events},prior,policy=policy)
class ContrastFocus(unittest.TestCase):
 def test_scoped_correction(self):
  p=plan([e('action',tactic='correct',target='listener misconception'),mismatch()]);self.assertTrue(verify_plan(p))
  self.assertEqual([k['at_word']for k in p['trajectory']if k['controls']['emphasis']>0],[5])
  self.assertNotIn('contrast_focus',p['final_state']['speech_behavior'])
  self.assertEqual(p['final_state']['character']['shared_history'][-1]['expected'],'The package is beside the door.')
 def test_amusement_alone_not_wallpaper(self):
  p=plan([e('set',values={'emotional_state.amusement':.8,'relationship.playfulness':.8})]);self.assertTrue(all(k['controls']['emphasis']==0 for k in p['trajectory']))
 def test_shared_teasing_requires_chosen_tactic(self):
  c=mismatch();c['perspective']='shared';base=[e('set',values={'emotional_state.amusement':.8,'relationship.playfulness':.8,'relationship.trust':.8}),c]
  a=plan(base);b=plan(base+[e('action',tactic='tease',target='shared discrepancy')]);self.assertEqual(a['trajectory'][5]['controls']['emphasis'],0);self.assertGreater(b['trajectory'][5]['controls']['emphasis'],0)
 def test_irrelevant_relationship_cannot_select_new_word(self):
  base=[e('action',tactic='correct',target='listener misconception'),mismatch()];a=plan(base);b=plan(base+[e('set',values={'relationship.familiarity':.8})]);self.assertEqual([k['controls']['emphasis']for k in a['trajectory']],[k['controls']['emphasis']for k in b['trajectory']])
 def test_no_mismatch_no_emphasis(self):
  c=mismatch();c['expected']=c['observed'];p=plan([e('action',tactic='correct',target='listener'),c]);self.assertTrue(all(k['controls']['emphasis']==0 for k in p['trajectory']))
 def test_history_persists_focus_does_not_recur(self):
  a=plan([e('action',tactic='correct',target='listener'),mismatch()]);b=plan([],a['final_state']);self.assertEqual(len(b['final_state']['character']['shared_history']),1);self.assertTrue(all(k['controls']['emphasis']==0 for k in b['trajectory']))
 def test_fail_closed_ambiguity_late_event_and_old_policy(self):
  c=mismatch();c['observed']='the window'
  with self.assertRaises(ValueError):plan([c])
  c=mismatch();c['at_word']=6
  with self.assertRaises(ValueError):plan([c])
  with self.assertRaises(ValueError):plan([mismatch()],policy='listener_causal_v6')
 def test_no_event_predecessor_parity(self):
  a=plan([]);b=plan([],policy='listener_causal_v6');self.assertEqual(a['final_state'],b['final_state']);self.assertEqual(a['trajectory'],b['trajectory'])
if __name__=='__main__':unittest.main()
