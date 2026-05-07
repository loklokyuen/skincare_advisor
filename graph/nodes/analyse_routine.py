from __future__ import annotations

from graph.tracing import traceable
from graph.context_status import input_status
from schemas import GraphState
from tools.analysis_tools import analyse_routine_basics, flag_common_conflicts
from tools.ingredient_tools import extract_ingredient_terms


@traceable(name="analyse_routine")
def analyse_routine(state: GraphState) -> GraphState:
    """Create routine observations with analysis tools before drafting advice."""
    routine = state.get("user_routine") or {}
    routine_items = routine.get("items") or routine.get("routine") or []
    profile = state.get("user_profile") or {}
    tool_outputs = list(state.get("tool_outputs") or [])
    status = state.get("input_status") or input_status(
        state.get("mode"),
        profile,
        routine,
        state.get("message", ""),
    )

    if not status["ready"]:
        analysis_results = {
            "conflicts": [],
            "gaps": [],
            "suitability_notes": [],
            "missing_inputs": status["missing"],
        }
        return {
            **state,
            "input_status": status,
            "analysis_results": analysis_results,
            "tool_outputs": tool_outputs,
        }

    basics = analyse_routine_basics.invoke({"profile": profile, "routine_items": routine_items})
    ingredient_terms = state.get("ingredient_terms")
    if ingredient_terms is None:
        ingredient_terms = extract_ingredient_terms.invoke({"text": state.get("message", "")})
    conflicts = flag_common_conflicts.invoke(
        {"ingredient_terms": ingredient_terms, "routine_items": routine_items}
    )

    tool_outputs.extend(
        [
            {"tool": "analyse_routine_basics", "output": basics},
            {"tool": "flag_common_conflicts", "output": conflicts},
        ]
    )

    analysis_results = {
        "conflicts": conflicts,
        "gaps": basics.get("gaps", []),
        "suitability_notes": basics.get("suitability_notes", []),
    }

    return {**state, "analysis_results": analysis_results, "tool_outputs": tool_outputs}
