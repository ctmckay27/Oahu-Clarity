"""Semantic counterfactual checks independent of model syntax success.

This small qualification corpus does not certify arbitrary scene interpretation.
It detects the concrete timing, negation, and internal/display failures at issue.
"""
import argparse,json,pathlib

def assess(row):
 if 'error' in row:return {'id':row['case']['id'],'passed':False,'checks':{'valid_result':False},'error':row['error']}
 events=row['result']['scene']['events'];name=row['case']['id'];checks={}
 def values():return {k:v for e in events if e['kind']=='set' for k,v in e['values'].items()}
 v=values()
 if name=='concealed_fear':
  masks=[e for e in events if e['kind']=='mask' and e['dimension']=='fear']
  checks={'internal_display_separated':len(masks)==1 and masks[0]['internal']>=.5 and masks[0]['display']<=.25 and masks[0]['effort']>=.5,
          'no_invented_discovery':not any(e['kind']=='knowledge' or (e['kind']=='thought' and e['mode'] in ['realizing','discovering','correcting']) for e in events),
          'no_invented_later_beats':all(e['at_word']==0 for e in events)}
 elif name=='discovery_inside_turn':
  checks={'search_before_resolution':any(e['at_word']==0 and e['kind']=='thought' and e['mode'] in ['searching','remembering'] for e in events),
          'resolution_at_observation':any(e['at_word']==5 and ((e['kind']=='thought' and e['mode'] in ['realizing','discovering','known']) or (e['kind']=='knowledge' and e['status']=='known' and e['confidence']>=.8)) for e in events),
          'no_premature_knowledge':not any(e['at_word']<5 and ((e['kind']=='knowledge' and e['status']=='known') or (e['kind']=='thought' and e['mode'] in ['realizing','known']) or (e['kind']=='set' and e['values'].get('knowledge_state.certainty',0)>.7)) for e in events),
          'no_unstated_boundaries':all(e['at_word'] in [0,5] for e in events)}
 elif name=='trusted_reassurance':
  checks={'trust':v.get('relationship.trust',0)>=.7,'familiarity':v.get('relationship.familiarity',0)>=.7,
          'action':any(e['kind']=='action' and e['tactic']=='reassure' for e in events),
          'no_invented_later_beats':all(e['at_word']==0 for e in events),
          'listener_worry_not_mari_fear':v.get('emotional_state.fear',0)==0 and not any(e['kind']=='mask' and e['dimension']=='fear' for e in events)}
 elif name=='physical_exertion':
  checks={'embodied_consequence':v.get('body_state.exertion',0)>=.5 or v.get('body_state.breath_reserve',1)<=.5,
          'negation_preserved':v.get('emotional_state.fear',0)==0 and not any(e['kind']=='mask' and e['dimension']=='fear' and e['internal']>0 for e in events),
          'no_invented_later_beats':all(e['at_word']==0 for e in events)}
 elif name=='irrelevant_room_fact':checks={'irrelevant_fact_causes_no_event':events==[]}
 else:raise ValueError('no semantic qualification defined for '+name)
 return {'id':name,'passed':all(checks.values()),'checks':checks,'scope':'tested semantic obligations only; not general scene-language certification'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('results');ap.add_argument('output');a=ap.parse_args()
 rows=[assess(r) for r in json.loads(pathlib.Path(a.results).read_text())]
 pathlib.Path(a.output).write_text(json.dumps({'cases':rows,'all_tested_cases_passed':bool(rows) and all(x['passed'] for x in rows),'full_scene_compiler_qualified':False},indent=2)+'\n')
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
