from __future__ import annotations

from .layer import AcousticRealizationLayer
from .models import RendererCalibration, RendererReceipt


def register_current_renderers(layer: AcousticRealizationLayer) -> AcousticRealizationLayer:
    layer.register_renderer(RendererCalibration(
        renderer_id="qwen3-tts-voicedesign-1.7b",
        implementation_id="gabriele-mastrapasqua/qwen3-tts@e391ec5467b0218eeb175f4888ad65b259d1e7c7",
        model_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        model_revision="NOT_CAPTURED_IN_ROUND1_RECEIPT",
        role="renderer_calibration_and_identity_authoring_surface",
        hard_prompt_constraints=[
            "An unmistakably adult feminine human voice.",
            "Natural conversational speech with coherent whole-utterance timing and prosody.",
            "Self-possessed, agentic, and socially directed rather than decorative or generic.",
        ],
        axis_language={
            "pitch_center": {"negative":"lower pitch center", "neutral":"centered adult pitch", "positive":"somewhat higher pitch center without youthfulness"},
            "vocal_weight": {"negative":"lighter vocal weight", "neutral":"balanced vocal weight", "positive":"heavier grounded vocal weight"},
            "resonance_size": {"negative":"more compact resonance", "neutral":"balanced resonance size", "positive":"larger more spacious resonance"},
            "brightness": {"negative":"darker tonal color", "neutral":"balanced brightness and depth", "positive":"brighter tonal color without girlishness"},
            "warmth": {"negative":"cooler tonal color", "neutral":"moderate natural warmth", "positive":"warmer tonal color without softness becoming the identity"},
            "breathiness": {"negative":"dry supported phonation", "neutral":"natural low breathiness", "positive":"audibly breathier phonation while remaining adult and controlled"},
            "roughness": {"negative":"clean smooth phonation", "neutral":"subtle organic texture", "positive":"slightly rough lived-in texture without vocal fry affectation"},
            "articulation_precision": {"negative":"relaxed articulation", "neutral":"clear natural diction", "positive":"precise fast consonants without over-enunciation"},
            "individuality": {"negative":"conventional unobtrusive timbre", "neutral":"recognizable individual vocal character", "positive":"distinctive uncommon but fully human timbre"},
            "polish": {"negative":"more lived-in and unpolished", "neutral":"natural studio-quality speech", "positive":"highly polished controlled delivery without synthetic smoothness"},
        },
        notes=[
            "Search coordinates are renderer controls/hypotheses, not physical acoustic truths.",
            "While MARI_VOICE_GENERATION_GATE v3.1 is closed, outputs are calibration probes only.",
        ],
    ))
    layer.register_renderer(RendererCalibration(
        renderer_id="qwen3-tts-base-0.6b",
        implementation_id="official-qwen-python-qwen-tts-0.1.1",
        model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        model_revision="5d83992436eae1d760afd27aff78a71d676296fc",
        role="reference_conditioned_clone_renderer",
        hard_prompt_constraints=[], axis_language={},
        notes=["Use after a lawful accepted anchor/reference exists; not a free-form identity authoring surface."],
    ))
    return layer


def import_current_renderer_proofs(layer: AcousticRealizationLayer) -> AcousticRealizationLayer:
    if not layer.state.get("renderer_proofs"):
        layer.import_renderer_proof(RendererReceipt(
            renderer_id="qwen3-tts-base-0.6b",
            implementation_id="official-qwen-python-qwen-tts-0.1.1",
            model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            model_revision="5d83992436eae1d760afd27aff78a71d676296fc",
            output_sha256="b108d9d7c8e53a2f768ca526ab8286f060c15aed5e42082c25d53e95baa8fdb9",
            config={"conditioning":"ICL reference audio + transcript", "sample_rate":24000},
            no_fallback_verified=True,
        ), "Official Qwen reference-conditioned clone proof, workflow run 35390986107.")
        layer.import_renderer_proof(RendererReceipt(
            renderer_id="qwen3-tts-base-0.6b-native-c",
            implementation_id="gabriele-mastrapasqua/qwen3-tts@e391ec5467b0218eeb175f4888ad65b259d1e7c7",
            model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            model_revision="MODEL_BYTES_ACQUIRED_DURING_RUN",
            config={"conditioning":"reference WAV -> ECAPA-TDNN x-vector", "profile_bytes":4096},
            no_fallback_verified=True,
        ), "Independent pinned native-C clone proof, workflow run 35391726537.")
    return layer


def build_current_layer() -> AcousticRealizationLayer:
    layer = AcousticRealizationLayer()
    register_current_renderers(layer)
    import_current_renderer_proofs(layer)
    layer.state["bootstrap_evidence"] = {
        "mari_native_design_trio": {
            "workflow_run": 35391726537,
            "role": "quality/progress evidence only; no candidate selected as Mari",
            "user_observation": "noticeably more coherent than prior attempts; progress; not done; no selection",
            "candidates": [
                {"id":"MVD-A","sha256":"f1be90f825ad5b5586b852ff6d4570cf5e530d93334e9f805e7b702b3c086aa8"},
                {"id":"MVD-B","sha256":"32890e310ff17eec4d92b024fbb477347c6680807686854b19039fa59e3430ae"},
                {"id":"MVD-F","sha256":"a7f2382cfd7423ad6827e764d74d9d286b74788bae5c85d26577a704f1d87263"},
            ],
        }
    }
    return layer