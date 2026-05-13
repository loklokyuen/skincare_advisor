from __future__ import annotations

from graph.tracing import traceable
from graph.context_status import input_status
from schemas import GraphState
from tools.advisor_tools import analyse_ingredient_candidates, analyse_product_candidates
from tools.analysis_tools import analyse_routine_basics, flag_common_conflicts
from tools.ingredient_tools import extract_ingredient_terms


@traceable(name="analyse_basics")
def analyse_basics(state: GraphState) -> GraphState:
    """Routine basics + conflict detection."""
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
        return {
            "input_status": status,
            "analysis_results": {
                "conflicts": [],
                "gaps": [],
                "suitability_notes": [],
                "missing_inputs": status["missing"],
            },
            "tool_outputs": tool_outputs,
        }

    basics = analyse_routine_basics.invoke({"profile": profile, "routine_items": routine_items})
    ingredient_terms = state.get("ingredient_terms")
    if ingredient_terms is None:
        ingredient_terms = extract_ingredient_terms.invoke({"text": state.get("message", "")})
    conflicts = flag_common_conflicts.invoke(
        {"ingredient_terms": ingredient_terms, "routine_items": routine_items}
    )

    tool_outputs.extend([
        {"tool": "analyse_routine_basics", "output": basics},
        {"tool": "flag_common_conflicts", "output": conflicts},
    ])

    return {
        "analysis_results": {
            "conflicts": conflicts,
            "gaps": basics.get("gaps", []),
            "suitability_notes": basics.get("suitability_notes", []),
        },
        "tool_outputs": tool_outputs,
    }


@traceable(name="analyse_products")
def analyse_products(state: GraphState) -> GraphState:
    """Rank product candidates for recommend/build modes."""
    if state.get("mode") not in {"recommend", "build"}:
        return {"product_recommendation_analysis": []}

    from graph.nodes.generate_response import _previous_assistant_recommendations

    candidates = (state.get("matched_products") or []) + (state.get("retrieved_products") or [])
    return {
        "product_recommendation_analysis": analyse_product_candidates(
            candidate_products=candidates,
            user_profile=state.get("user_profile") or {},
            user_routine=state.get("user_routine") or {},
            message=state.get("message") or "",
            previously_recommended=_previous_assistant_recommendations(state),
            limit=4,
        )
    }


@traceable(name="analyse_ingredients")
def analyse_ingredients(state: GraphState) -> GraphState:
    """Rank ingredient candidates for recommend/build/learn modes."""
    if state.get("mode") not in {"recommend", "build", "learn"}:
        return {"ingredient_recommendation_analysis": []}

    return {
        "ingredient_recommendation_analysis": analyse_ingredient_candidates(
            candidate_ingredients=state.get("retrieved_ingredients") or [],
            user_profile=state.get("user_profile") or {},
            user_routine=state.get("user_routine") or {},
            limit=3,
        )
    }
