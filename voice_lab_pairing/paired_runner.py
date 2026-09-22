"""Supplementary causal comparisons; no identity or coordinate canonization."""
import argparse, copy, importlib.util, pathlib, shutil
p=pathlib.Path('voice_lab_sealed/runner.py')
spec=importlib.util.spec_from_file_location('base_runner',p)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
original={c['id']:c for c in b.CASES};cases=[]
for i in ['P01','P02','P03','P04']:
    c=copy.deepcopy(original[i]);c['id']='Q'+i;c['seed']=966700;c['comparison_group']='prosody_fixed_seed_A';cases.append(c)
for j,seed in [('B',966701),('C',966702)]:
    for i in ['P01','P02']:
        c=copy.deepcopy(original[i]);c['id']='Q'+i+j;c['seed']=seed;c['comparison_group']='prosody_fixed_seed_'+j;cases.append(c)
for i in ['S01','S02','S03','S04','S05','S06','S07']:
    c=copy.deepcopy(original[i]);c['id']='Q'+i;c['seed']=966600;c['comparison_group']='state_fixed_seed';cases.append(c)
coords=[
('C01',.2,.2,.2,'Low expressive energy and low warmth, but remain attentive and naturally conversational. Calm, neutral, clear; not cold or robotic.'),
('C02',.2,.8,.2,'Low expressive energy and warm practical attentiveness. Calm and grounded, with a little warmth in timing, not breathiness or sentimentality.'),
('C03',.8,.2,.2,'Higher purposeful expressive energy with neutral warmth. Brisk, alert, and precise; not angry, loud, or theatrical.'),
('C04',.8,.8,.2,'Higher purposeful expressive energy with warm practical attentiveness. Brisk, engaged, and clear; not cute, loud, or theatrical.'),
('C05',.5,.5,.2,'Moderate ordinary energy and moderate warmth. Balanced practical conversation, with restrained pitch movement and responsive timing.'),
('C06',.5,.5,.7,'The same moderate energy and warmth, but with contained underlying tension. Keep control and clarity without an exaggerated anger performance.')]
for i,a,w,t,d in coords:
    c=copy.deepcopy(original['S07']);c.update(id='Q'+i,category='hypothetical_state_coordinates',seed=966600,instruct=d,comparison_group='exploratory_energy_warmth_tension',intended_coordinates=dict(expressive_energy=a,warmth=w,tension=t),coordinate_status='prompt-target hypotheses; achieved coordinates unmeasured');cases.append(c)
b.CASES=cases
ap=argparse.ArgumentParser();ap.add_argument('action',choices=['generate','evaluate']);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=2);args=ap.parse_args()
if args.action=='generate':
    b.generate(args.shard,args.shards)
    shutil.copy2(__file__,b.OUT/'paired_runner.py')
    b.write(b.OUT/'comparison_design.json',dict(status='unjudged_research',fixed_seed_comparisons=True,source_exploratory_run=35729356566,confounds_not_removed=['prompt interpretation is renderer-dependent','intended warmth and energy are not calibrated physical coordinates'],certification='NOT_CERTIFIED'))
else:b.evaluate()
