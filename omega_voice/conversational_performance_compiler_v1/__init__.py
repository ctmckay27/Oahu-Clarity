"""MARI_CONVERSATIONAL_PERFORMANCE_COMPILER_v1."""

from .compiler import (
    SCHEMA,
    STATE_ID,
    compile_performance_plan,
    verify_performance_plan,
)
from .adapter import (
    SCHEMA as NATIVE_SCHEMA,
    STATE_ID as NATIVE_STATE_ID,
    compile_native_performance,
    verify_native_performance_plan,
)
from .renderer import RECEIPT_SCHEMA, render_native_performance
from .learning import capture_reaction, new_learning_state, append_reaction
from .respiration import qualification_candidate
from .manifold import compile_manifold, verify_manifold
from .runtime import plan_native_turn, render_planned_turn, commit_actual_delivery, record_ordinary_reaction

__all__ = [
    "SCHEMA",
    "STATE_ID",
    "NATIVE_SCHEMA",
    "NATIVE_STATE_ID",
    "RECEIPT_SCHEMA",
    "compile_performance_plan",
    "verify_performance_plan",
    "compile_native_performance",
    "verify_native_performance_plan",
    "render_native_performance",
    "capture_reaction",
    "new_learning_state",
    "append_reaction",
    "qualification_candidate",
    "compile_manifold",
    "verify_manifold",
    "plan_native_turn",
    "render_planned_turn",
    "commit_actual_delivery",
    "record_ordinary_reaction",
]
