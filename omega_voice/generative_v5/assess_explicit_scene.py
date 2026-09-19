"""Independent semantic obligations for scene compilation qualification."""
import argparse,json,pathlib
def signatures(result):
 values=[]
 for e in result['scene']['events']:
  k=e['kind'];at='@'+str(e['at_word'])
  if k=='set':values.extend(p+'='+str(v)+at for p,v in e['values'].items())
  elif k=='thought':values.append('thought:'+e['mode']+at)
  elif k=='action':values.append('action:'+e['tactic']+at)
  elif k=='mask':values.append('mask:'+e['dimension']+at)
  elif k=='knowledge':values.append('knowledge:'+e['status']+at)
  else:values.append(k+at)
 return values
def matches(want,actual):
 if '@' in want:
  stem,at=want.rsplit('@',1)
  if not actual.endswith('@'+at):return False
 else:stem=want
 return actual.split('@')[0].startswith(stem)
def assess(row):
 expected=row['case']['expected'];result=row.get('result');checks={}
 if result is None:return {'id':row['case']['id'],'passed':False,'error':row.get('error')}
 actual=signatures(result)
 checks['admission_matches']=result['admitted']==expected['admitted']
 for want in expected['required']:checks['requires '+want]=any(matches(want,s) for s in actual)
 for forbid in expected['forbidden']:checks['excludes '+forbid]=not any(matches(forbid,s) for s in actual)
 if not expected['required'] and expected['admitted']:checks['no_unrequested_state']=actual==[]
 return {'id':row['case']['id'],'passed':all(checks.values()),'checks':checks,'observed':actual,'unresolved':result['unresolved']}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('results');ap.add_argument('output');a=ap.parse_args()
 rows=[assess(r) for r in json.loads(pathlib.Path(a.results).read_text())]
 record={'cases':rows,'passed':sum(x['passed'] for x in rows),'total':len(rows),'scope':'only these semantic obligations, not unrestricted scene-language certification'}
 pathlib.Path(a.output).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
