import copy,unittest
from omega_voice.causal_v4.runtime import compile_scene,new_state,verify_plan

def event(kind,**fields):return dict(id=kind,kind=kind,at_word=0,source={'text':'explicit test scene'},**fields)
def plan(events,prior=None):return compile_scene('The package is ready.',{'events':events},prior,policy='listener_causal_v6')
class ListenerCausality(unittest.TestCase):
 def test_listener_worry_is_not_mari_fear(self):
  a=plan([]);b=plan([event('listener_condition',values={'worry':.8})])
  self.assertEqual(b['final_state']['emotional_state']['fear'],0)
  self.assertEqual([k['controls']for k in a['trajectory']],[k['controls']for k in b['trajectory']])
 def test_response_requires_chosen_tactic_and_concern(self):
  base=[event('listener_condition',values={'worry':.8}),event('action',tactic='reassure',target='current listener')]
  a=plan(base);b=plan(base+[event('set',values={'relationship.concern':.8})]);c=plan([e for e in base if e['kind']!='action']+[event('set',values={'relationship.concern':.8})])
  self.assertGreater(b['trajectory'][0]['controls']['attack_softness'],c['trajectory'][0]['controls']['attack_softness'])
  self.assertEqual(a['trajectory'][0]['controls']['attack_softness'],.25)
  self.assertLess(b['trajectory'][0]['controls']['gain_db'],c['trajectory'][0]['controls']['gain_db'])
 def test_no_anticipation(self):
  e=event('listener_condition',values={'confusion':.8});e['at_word']=2
  base=[event('action',tactic='clarify',target='current listener')];a=plan(base);b=plan(base+[e])
  self.assertEqual(a['trajectory'][1]['controls'],b['trajectory'][1]['controls'])
  self.assertGreater(b['trajectory'][2]['controls']['precision'],a['trajectory'][2]['controls']['precision'])
 def test_listener_history_is_specific_and_recoverable(self):
  p=plan([event('listener_condition',values={'worry':.8})]);prior=p['final_state']
  a=plan([event('listener',listener_id='stranger')],prior)
  self.assertNotIn('condition',a['final_state']['listener_model'])
  b=plan([event('listener',listener_id='unspecified')],a['final_state'])
  self.assertEqual(b['final_state']['listener_model']['condition'],{'worry':.8});self.assertTrue(verify_plan(b))
 def test_fail_closed_wrong_listener_or_policy(self):
  e=event('listener_condition',values={'worry':.8},listener_id='other')
  with self.assertRaises(ValueError):plan([e])
  e.pop('listener_id')
  with self.assertRaises(ValueError):compile_scene('The package is ready.',{'events':[e]},policy='respiratory_budget_v5')
  with self.assertRaises(ValueError):compile_scene('The package is ready.',{},plan([e])['final_state'],policy='respiratory_budget_v5')
 def test_no_event_matches_predecessor_controls(self):
  a=plan([]);b=compile_scene('The package is ready.',{},policy='respiratory_budget_v5')
  self.assertEqual(a['final_state'],b['final_state']);self.assertEqual(a['trajectory'],b['trajectory'])
if __name__=='__main__':unittest.main()
