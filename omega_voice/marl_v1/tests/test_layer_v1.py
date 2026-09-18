import copy, json, sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from mari_acoustic_realization.bootstrap import build_current_layer
from mari_acoustic_realization.layer import AcousticRealizationLayer
from mari_acoustic_realization.models import (
    CandidateStatus, ExperimentPurpose, ExperimentRecord, HumanVerdict, IdentityJudgment,
    ListeningFeedback, PerformanceState, QualityJudgment, RendererReceipt,
)
from mari_acoustic_realization.store import atomic_write_json, load_json


def q_good():
    return QualityJudgment(.92,.93,.86,.86,.84,.95,.88)

def i_good(accepted=False):
    return IdentityJudgment(.90,.88,.97,.89,.82,accepted_as_mari=accepted)

def receipt(tag='a'):
    return RendererReceipt('qwen3-tts-voicedesign-1.7b','native-c','Qwen VoiceDesign','rev1',404,output_sha256=(tag*64)[:64],no_fallback_verified=True)

def exp(layer,eid,purpose=ExperimentPurpose.RENDERER_CALIBRATION,quality=None,identity=None,controls=None):
    return ExperimentRecord(eid,'2026-09-18T00:00:00Z',layer.state_digest(),'Give me the problem as it is.',PerformanceState(),controls or {'brightness':-.2,'vocal_weight':.3},'test',receipt(eid),purpose=purpose,quality=quality,identity=identity)

def open_gate(layer):
    return layer.set_population_route_state(population_basis_ingested=True,project_owned_continuous_generator_fit=True,source_refs=['basis:verified','generator:verified'])


def test_bootstrap_binds_current_sources_and_gate():
    layer=build_current_layer()
    assert layer.state['current_sources']['voice_resolution']['target']=='MARI_VOICE_RESOLUTION_STATE_OBJECT_v1.5_2026-09-17'
    assert layer.state['current_sources']['generation_gate']['target']=='MARI_VOICE_GENERATION_GATE_v3.1_2026-09-17'
    assert layer.state['current_sources']['voice_machine']['target']=='MARI_VOICE_MACHINE_STATE_v0.3_2026-09-18'
    assert layer.state['admission']['promotable_candidate_allowed'] is False


def test_default_plan_is_calibration_while_gate_closed():
    layer=build_current_layer(); plan=layer.next_search_plan('qwen3-tts-voicedesign-1.7b')
    assert plan.purpose==ExperimentPurpose.RENDERER_CALIBRATION


def test_promotable_plan_refuses_while_gate_closed():
    layer=build_current_layer()
    with pytest.raises(PermissionError):
        layer.next_search_plan('qwen3-tts-voicedesign-1.7b',purpose=ExperimentPurpose.PROMOTABLE_CANDIDATE)


def test_calibration_experiment_cannot_become_identity_evidence():
    layer=build_current_layer(); status=layer.record_experiment(exp(layer,'CAL1',quality=q_good(),identity=i_good(True)))
    assert status==CandidateStatus.CALIBRATION_ONLY
    assert layer.state['plausible_exemplars']==[]
    assert layer.state['accepted_anchor_id'] is None


def test_quality_failure_discards_identity():
    layer=build_current_layer(); bad=QualityJudgment(.3,.3,.3,.3,.3,.3,.3)
    status=layer.record_experiment(exp(layer,'BAD',quality=bad,identity=i_good(True)))
    assert status==CandidateStatus.QUALITY_REJECTED
    assert layer.state['experiments'][-1]['identity'] is None


def test_vague_rejection_does_not_move_coordinates():
    layer=build_current_layer(); layer.record_experiment(exp(layer,'CAL2'))
    before=copy.deepcopy(layer.state['coordinates'])
    layer.record_feedback(ListeningFeedback('CAL2','2026-09-18T00:00:01Z',quality_verdict=HumanVerdict.REJECT,observations=['not there']))
    assert layer.state['coordinates']==before


def test_directional_feedback_moves_only_named_coordinate():
    layer=build_current_layer(); layer.record_experiment(exp(layer,'CAL3'))
    before_b=layer.state['coordinates']['brightness']['center']; before_w=layer.state['coordinates']['warmth']['center']
    result=layer.record_feedback(ListeningFeedback('CAL3','2026-09-18T00:00:01Z',desired_movements={'brightness':-.5},observations=['darker']))
    assert layer.state['coordinates']['brightness']['center'] < before_b
    assert layer.state['coordinates']['warmth']['center']==before_w
    assert result['promotable_identity_evidence'] is False


def test_non_carl_feedback_refused():
    layer=build_current_layer(); layer.record_experiment(exp(layer,'CAL4'))
    with pytest.raises(PermissionError): layer.record_feedback(ListeningFeedback('CAL4','x',actor='system'))


def test_gate_change_requires_source_refs():
    layer=build_current_layer()
    with pytest.raises(ValueError): layer.set_population_route_state(population_basis_ingested=True,project_owned_continuous_generator_fit=True,source_refs=[])


def test_gate_opens_only_when_both_conditions_true():
    layer=build_current_layer()
    assert not layer.set_population_route_state(population_basis_ingested=True,project_owned_continuous_generator_fit=False,source_refs=['basis'])
    assert open_gate(layer)
    assert layer.state['admission']['promotion_gate_status']=='OPEN'


def test_promotable_plan_allowed_after_gate_open():
    layer=build_current_layer(); open_gate(layer)
    plan=layer.next_search_plan('qwen3-tts-voicedesign-1.7b',purpose=ExperimentPurpose.PROMOTABLE_CANDIDATE)
    assert plan.purpose==ExperimentPurpose.PROMOTABLE_CANDIDATE


def test_promotable_candidate_blocked_if_gate_closed():
    layer=build_current_layer(); status=layer.record_experiment(exp(layer,'P0',ExperimentPurpose.PROMOTABLE_CANDIDATE,q_good(),i_good(True)))
    assert status==CandidateStatus.PROMOTION_BLOCKED


def test_anchor_requires_open_gate_and_human_verdicts():
    layer=build_current_layer(); open_gate(layer)
    rec=exp(layer,'P1',ExperimentPurpose.PROMOTABLE_CANDIDATE,q_good(),i_good(True)); layer.record_experiment(rec)
    with pytest.raises(ValueError): layer.accept_anchor('P1','Carl')
    layer.record_feedback(ListeningFeedback('P1','x',quality_verdict=HumanVerdict.PASS,identity_verdict=HumanVerdict.ACCEPT,observations=['yes']))
    a=layer.accept_anchor('P1','Carl')
    assert a.anchor_id==layer.state['accepted_anchor_id']


def test_calibration_probe_cannot_be_promoted_even_after_gate_opens():
    layer=build_current_layer(); layer.record_experiment(exp(layer,'CAL5',quality=q_good(),identity=i_good(True))); open_gate(layer)
    layer.record_feedback(ListeningFeedback('CAL5','x',quality_verdict=HumanVerdict.PASS,identity_verdict=HumanVerdict.ACCEPT))
    with pytest.raises(ValueError): layer.accept_anchor('CAL5','Carl')


def test_non_carl_anchor_refused():
    layer=build_current_layer(); open_gate(layer); layer.record_experiment(exp(layer,'P2',ExperimentPurpose.PROMOTABLE_CANDIDATE,q_good(),i_good(True)))
    layer.record_feedback(ListeningFeedback('P2','x',quality_verdict=HumanVerdict.PASS,identity_verdict=HumanVerdict.ACCEPT))
    with pytest.raises(PermissionError): layer.accept_anchor('P2','system')


def test_persistence_roundtrip(tmp_path):
    layer=build_current_layer(); path=tmp_path/'state.json'; atomic_write_json(path,layer.snapshot())
    restored=AcousticRealizationLayer(load_json(path)); assert restored.state_digest()==layer.state_digest()


def test_engineering_readiness_requires_execution_and_guards():
    layer=build_current_layer(); r=layer.engineering_readiness_report(); assert not r.ready
    for k in ['layer_persistence_verified','judgment_interface_verified','promotion_guard_verified','round_execution_verified']:
        layer.record_verification(k,True)
    assert layer.engineering_readiness_report().ready


def test_voice_readiness_stays_false_when_promotion_gate_closed():
    layer=build_current_layer(); r=layer.readiness_report(); assert not r.ready
    assert any(g.gate=='promotion_route_eligible' and not g.passed for g in r.gates)


def test_completion_separates_engineering_from_voice():
    layer=build_current_layer()
    for k in ['layer_persistence_verified','judgment_interface_verified','promotion_guard_verified','round_execution_verified']:
        layer.record_verification(k,True)
    c=layer.completion_report(); assert c['engineering_complete'] is True; assert c['voice_v1_ready'] is False


def test_renderer_proof_sets_no_fallback():
    layer=build_current_layer(); assert layer.state['verification']['no_fallback_verified'] is True


def test_prompt_compiler_includes_hard_constraints_and_axes():
    layer=build_current_layer(); p=layer.compile_voice_design_prompt('qwen3-tts-voicedesign-1.7b',{'brightness':-.4,'vocal_weight':.4})
    assert 'adult feminine' in p and 'darker tonal color' in p and 'heavier grounded vocal weight' in p


def test_migrate_v01_state_to_v1_preserves_experiments():
    legacy=build_current_layer().snapshot(); legacy['schema']='mari-acoustic-realization-layer/0.1'; legacy['object']['version']='v0.2_2026-09-18'; legacy.pop('admission',None); legacy.pop('current_sources',None); legacy.pop('feedback_events',None); legacy.pop('completion',None)
    layer=AcousticRealizationLayer(legacy)
    assert layer.state['schema']=='mari-acoustic-realization-layer/1.0'
    assert layer.state['admission']['promotable_candidate_allowed'] is False


def test_feedback_parser_fields_roundtrip():
    fb=ListeningFeedback('X','t',quality_verdict=HumanVerdict.PASS,identity_verdict=HumanVerdict.OPEN,desired_movements={'brightness':-.2})
    assert fb.desired_movements['brightness']==-.2