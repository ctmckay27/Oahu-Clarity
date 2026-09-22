"""Remove cross-runner build confounding from matched-seed comparisons."""
import argparse,pathlib,shutil,hashlib
source=pathlib.Path('voice_lab_pairing/paired_runner.py').read_text()
marker='ap=argparse.ArgumentParser()'
if source.count(marker)!=1:raise RuntimeError('Unexpected paired-runner layout; do not guess execution boundary')
ns={'__name__':'paired_specification_only'}
exec(compile(source.split(marker)[0],'voice_lab_pairing/paired_runner.py','exec'),ns)
b=ns['b']
for c in b.CASES:
    c['replicated_case_id']=c['id'];c['id']='D'+c['id'];c['comparison_execution']='all matched cases in one job with one compiled binary'
p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','evaluate']);a=p.parse_args()
if a.action=='generate':
    b.generate(0,1)
    shutil.copy2(__file__,b.OUT/'replicate.py')
    shutil.copy2('voice_lab_pairing/paired_runner.py',b.OUT/'paired_runner.py')
    b.write(b.OUT/'replication_design.json',dict(original_run=35730773797,control='all 21 comparisons run sequentially using one exact executable and model installation',scope='Measured acoustic and ASR comparisons; no human identity adjudication',paired_specification_sha256=hashlib.sha256(source.encode()).hexdigest(),certification='NOT_CERTIFIED'))
else:b.evaluate()
