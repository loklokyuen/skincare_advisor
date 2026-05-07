from __future__ import annotations

import logging
from functools import lru_cache

from config.openai import load_openai_config, openai_chat_kwargs
from graph.nodes.generate_response import _build_context_block, _previous_assistant_recommendations
from graph.tracing import traceable
from schemas import GraphState
from tools.advisor_tools import analyse_ingredient_candidates, analyse_product_candidates

log = logging.getLogger(__name__)


AGENT_SYSTEM_PROMPT = """
You are the orchestration agent for a skincare advisor app.

Your job is to decide what supporting tools are needed before a separate response
writer drafts the final user-facing answer. Do not write the final answer.

Responsibilities:
- Analyse product candidates before any product recommendation answer.
- Analyse ingredient candidates before any ingredient advice answer.
- Select 1-2 exact product names that the response writer should recommend.
- Flag community or literature follow-up only when the user explicitly asks for it.

Rules:
- Treat the "User profile" section in available context as already saved memory.
  The user manages their own profile via the Profile page; do not try to update it.
- The retrieve_context node has already fetched product and ingredient candidates;
  rely on those candidates and on candidate analysis. Do not search the catalog.
- In recommend/build mode, when you choose products, call select_product_cards
  once with 1-2 exact product names from the candidate analysis output.
- Do not select products already in the user's routine or previously recommended, unless specified by the user.
- Recommend at most 1-2 products or ingredients. Prefer the single strongest fit
  unless there are two clearly different gaps.
- Prefer routine adjustments or ingredient categories when no product fit is strong.
- Return concise internal advisor notes only. No user-facing prose.
- If the user asks something unrelated to topical skincare products, politely decline.
"""


@lru_cache(maxsize=8)
def _get_agent_llm(model: str, api_key: str, base_url: str | None):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        **openai_chat_kwargs(
            model,
            api_key,
            base_url,
            temperature=0.0,
            timeout=25,
            max_completion_tokens=700,
            reasoning_effort="minimal",
            verbosity="low",
        )
    )


def _fallback_agent_notes(state: GraphState) -> dict:
    products = (state.get("matched_products") or []) + (state.get("retrieved_products") or [])
    ingredients = state.get("retrieved_ingredients") or []
    previous = _previous_assistant_recommendations(state)
    product_analysis = analyse_product_candidates(
        candidate_products=products,
        user_profile=state.get("user_profile") or {},
        user_routine=state.get("user_routine") or {},
        message=state.get("message") or "",
        previously_recommended=previous,
        limit=2,
    )
    ingredient_analysis = analyse_ingredient_candidates(
        candidate_ingredients=ingredients,
        user_profile=state.get("user_profile") or {},
        user_routine=state.get("user_routine") or {},
        limit=2,
    )
    selected = [
        item["product_name"]
        for item in product_analysis[:2]
        if item.get("score", 0) >= 0 and item.get("product_name")
    ]
    notes = []
    if product_analysis:
        notes.append("Product candidate analysis is available; use the highest-ranked fits first.")
    if ingredient_analysis:
        notes.append("Ingredient candidate analysis is available; do not suggest ingredients already in routine.")
    return {
        "advisor_notes": "\n".join(notes),
        "product_recommendation_analysis": product_analysis,
        "ingredient_recommendation_analysis": ingredient_analysis,
        "agent_selected_product_names": selected,
        "community_search_request": None,
        "literature_search_request": None,
    }


@traceable(name="skincare_agent")
def skincare_agent(state: GraphState) -> GraphState:
    """Orchestrate tool use and prepare internal notes for the response writer."""
    api_key, model, base_url = load_openai_config("SKINIQ_AGENT_MODEL")

    products = (state.get("matched_products") or []) + (state.get("retrieved_products") or [])
    ingredients = state.get("retrieved_ingredients") or []
    profile = state.get("user_profile") or {}
    routine = state.get("user_routine") or {}
    previous = _previous_assistant_recommendations(state)

    if not api_key:
        return {**state, **_fallback_agent_notes(state)}

    # Analyse and learn modes do not need the agent's product picker or
    # tool-flagging. Use the deterministic fallback (pure ranking, no LLM)
    # to skip the main agent LLM + tool loop on those turns.
    if state.get("mode") in {"analyse", "learn"}:
        return {**state, **_fallback_agent_notes(state)}

    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool

    selected_product_names: list[str] = []
    community_requests: list[dict] = []
    literature_requests: list[dict] = []
    product_analysis_outputs: list[dict] = []
    ingredient_analysis_outputs: list[dict] = []

    @tool
    def analyse_product_recommendation_fit(limit: int = 2) -> list[dict]:
        """Rank available product candidates for the current user and request."""
        result = analyse_product_candidates(
            candidate_products=products,
            user_profile=profile,
            user_routine=routine,
            message=state.get("message") or "",
            previously_recommended=previous,
            limit=min(max(limit, 1), 2),
        )
        product_analysis_outputs.clear()
        product_analysis_outputs.extend(result)
        return result

    @tool
    def analyse_ingredient_recommendation_fit(limit: int = 2) -> list[dict]:
        """Rank retrieved ingredient candidates and flag any already present in the routine."""
        result = analyse_ingredient_candidates(
            candidate_ingredients=ingredients,
            user_profile=profile,
            user_routine=routine,
            limit=min(max(limit, 1), 2),
        )
        ingredient_analysis_outputs.clear()
        ingredient_analysis_outputs.extend(result)
        return result

    @tool
    def select_product_cards(product_names: list[str]) -> dict:
        """Select 1-2 exact product names the response writer should recommend and the UI should render."""
        selected_product_names.clear()
        selected_product_names.extend(product_names[:2])
        return {"selected": selected_product_names}

    @tool
    def flag_community_search(query: str, reason: str) -> dict:
        """Flag community/review evidence only when the user explicitly asked for it."""
        community_requests.append({"query": query, "reason": reason})
        return {"flagged": True, "query": query}

    @tool
    def flag_literature_search(query: str, reason: str) -> dict:
        """Flag scientific literature only for an explicit named-ingredient evidence question."""
        literature_requests.append({"query": query, "reason": reason})
        return {"flagged": True, "query": query}

    tools = [
        analyse_product_recommendation_fit,
        analyse_ingredient_recommendation_fit,
        select_product_cards,
        flag_community_search,
        flag_literature_search,
    ]

    try:
        context = state.get("context_block") or _build_context_block(state)
        messages = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"User message: {state.get('message') or ''}\n\n"
                    f"Mode: {state.get('mode') or 'analyse'}\n\n"
                    f"Available context:\n{context}\n\n"
                    "The User profile context is already saved. Do not resave those values.\n\n"
                    "Prepare internal advisor notes and call tools as needed."
                )
            ),
        ]
        llm = _get_agent_llm(model, api_key, base_url)
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(messages)
        tool_calls = getattr(response, "tool_calls", None) or []

        for _iteration in range(3):
            if not tool_calls:
                break
            messages.append(response)
            for call in tool_calls:
                name = call.get("name")
                args = call.get("args") or {}
                call_id = call.get("id", name or "tool_call")
                if name == "analyse_product_recommendation_fit":
                    tool_result = analyse_product_recommendation_fit.invoke(args)
                elif name == "analyse_ingredient_recommendation_fit":
                    tool_result = analyse_ingredient_recommendation_fit.invoke(args)
                elif name == "select_product_cards":
                    tool_result = select_product_cards.invoke(args)
                elif name == "flag_community_search":
                    tool_result = flag_community_search.invoke(args)
                elif name == "flag_literature_search":
                    tool_result = flag_literature_search.invoke(args)
                else:
                    continue
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=call_id))
            response = llm_with_tools.invoke(messages)
            tool_calls = getattr(response, "tool_calls", None) or []

        advisor_notes = response.content if isinstance(response.content, str) else ""
    except Exception as exc:
        log.warning("skincare_agent failed, using deterministic notes: %s", exc, exc_info=True)
        return {**state, **_fallback_agent_notes(state)}

    tool_outputs = list(state.get("tool_outputs") or [])
    if product_analysis_outputs:
        tool_outputs.append({"tool": "analyse_product_recommendation_fit", "output": product_analysis_outputs})
    if ingredient_analysis_outputs:
        tool_outputs.append({"tool": "analyse_ingredient_recommendation_fit", "output": ingredient_analysis_outputs})

    return {
        **state,
        "advisor_notes": advisor_notes,
        "tool_outputs": tool_outputs,
        "community_search_request": community_requests[0] if community_requests else None,
        "literature_search_request": literature_requests[0] if literature_requests else None,
        "product_recommendation_analysis": product_analysis_outputs,
        "ingredient_recommendation_analysis": ingredient_analysis_outputs,
        "agent_selected_product_names": selected_product_names,
        "context_block": context,
    }
