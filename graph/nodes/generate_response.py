from __future__ import annotations

import logging
import re
from functools import lru_cache

from config.openai import load_openai_config, openai_chat_kwargs
from graph.tracing import traceable
from schemas import GraphState
from utils.ingredient_format import format_ingredient_name, format_ingredient_names

log = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _get_llm(model: str, api_key: str, base_url: str | None):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        **openai_chat_kwargs(
            model,
            api_key,
            base_url,
            temperature=0.3,
            timeout=30,
            max_completion_tokens=900,
            reasoning_effort="minimal",
            verbosity="low",
        )
    )

SYSTEM_PROMPT = """
You are a practical, friendly, evidence-aware skincare advisor, focused on topical skincare solutions only.

Draft the final user-facing answer using the provided context: saved profile,
routine, advisor notes, product/ingredient analysis, catalog data, retrieved
ingredient details, and conservative general skincare knowledge for uncatalogued
terms. Do not call tools or mention internal actions.

Core rules:
- Be concise, warm, conservative, and specific.
- Do not invent products, ingredients, studies, reviews, formulation details, or user facts.
- Start from the user's existing routine and profile.
- Do not answer anything besides topical skincare products, such as injection, laser, or exercise/diet.
- If there is a new term, assume it is about topical skincare products/ingredients, if it is not, politely decline to answer.
- For a named skincare ingredient or product that is not in the provided catalog
  context, still answer the user's high-level educational question from general
  skincare knowledge. Be conservative, say when evidence or formulation details
  are limited, and do not claim local catalog confirmation.
- Do not ask the user for permission to give a high-level explanation of an
  uncatalogued skincare term. Ask a follow-up only when they want a specific
  product recommendation, routine placement, or compatibility check that needs
  profile, routine, or product-label details.
- If required data is missing, ask for that data instead of guessing.
- Do not diagnose medical conditions; for severe, persistent, prescription, pregnancy,
  infection, swelling, or burning concerns, advise speaking to a clinician.
- Do not mention tools, searches, the catalog, database, Reddit, reviews, or evidence
  follow-up unless actual evidence text is provided in context.
- If a profile fact was saved this turn, acknowledge it naturally as part of the
  answer, then continue answering the user's request. Do not use "Saved:" or
  other system-style status labels.

Routine/product rules:
- Products already in the user's routine are not new recommendations.
- If agent-selected product names are provided, use those exact names for specific
  recommendations.
- Recommend at most 1-2 products or ingredients.
- If recommending two, present them as separate suitable options without ranking labels.
- Do not call products "primary", "secondary", "optional", "alternate", or "alternative".
- When agent-selected product names or candidate products are present, recommend
  those specific products by exact name. Do not substitute a generic "a hyaluronic
  acid serum" or "any peptide moisturiser" line when a real candidate exists.
- Only fall back to a product category, ingredient type, or routine adjustment
  when no candidate product fits — say so explicitly ("no strong product match
  for this; try ...") instead of mixing a real recommendation with a generic one.

Analyse-mode rules:
- Do not restate the user's saved profile or routine back to them. They already
  know what they use and what their goals are. Refer to a product or goal only
  when it is load-bearing for the point you are making.
- Skip any section that has nothing to report. Do not write "Redundancies: none"
  or "No conflicts detected" — just omit the section.
- Treat cleanser and sunscreen as baseline staples that the user almost certainly
  already has. Do not list them as "missing steps" or "gaps" unless the user has
  explicitly asked about them or has stated they do not use one.
- A "missing step" is only a real gap when (a) the user's stated goal needs a
  specific active or product type they do not have, or (b) their analysis_results
  flagged it as a gap.

Response style:
- Use Markdown headings when the answer has more than one topic.
- Prefer H4 headings (`####`) for sections, products, and ingredients.
- Use short bullets instead of dense paragraphs.
- Each bullet should usually be 6-16 words.
- Each bullet should contain one clear idea only.
- Avoid semicolons and long chained clauses.
- Use **bold** for important ingredients, skin concerns, goals, risks, or routine context.
- Use *italic* for gentle cautions, uncertainty, or practical notes.
- Do not use label-heavy lines such as "Why it fits:", "Gap filled:", "Usage/caution:", or "Key actives:".
- Do not describe the user with blunt identity-style wording such as "you're oily", "you're acne-prone", or "you're dry".
- Use profile-aware phrasing instead, such as "with oily, acne-prone skin" or "for a dry skin profile".
- Keep the answer concise by default.
- End with at most one natural next step, and only when useful.

Example output:You already use **Retinol** in PM and **Niacinamide** regularly, so I’d keep any new acne step separate from your active nights.

#### CeraVe Blemish Control Gel Moisturiser with 2% Salicylic Acid & Niacinamide

- Fits **oily, acne-prone skin** because it adds a leave-on **BHA** step for clogged pores.
- **Salicylic Acid** can help with congestion, breakouts, and the look of large pores.
- 🕒 Use on **Rest PM** nights instead of your retinol nights.
- *Start 2-3 times weekly to reduce the chance of dryness or irritation.*

#### Skin + Me Breakouts + Visible Pores Serum, with Azelaic Acid

- Fits if you want a gentler-feeling option for **breakouts**, **oiliness**, and visible pores.
- **Azelaic Acid** can support blemish-prone skin without adding another retinoid.
- 🕒 Use on **Rest PM** nights or in the morning.
- Avoid applying it at the same time as retinol until you know your skin tolerates it.*
"""

def _dict_items(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _text_items(value: object) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def _previous_assistant_recommendations(state: GraphState) -> list[str]:
    from langchain_core.messages import AIMessage
    from tools.product_tools import _response_product_suggestion_queries

    names = []
    seen = set()

    for message in state.get("messages") or []:
        if not isinstance(message, AIMessage):
            continue
        content = message.content if isinstance(message.content, str) else ""
        if not content:
            continue
        for query in _response_product_suggestion_queries(content):
            name = re.sub(r"\s+", " ", query).strip(" -:;.")
            key = name.lower()
            if name and key not in seen:
                names.append(name)
                seen.add(key)
    return names


def _build_context_block(state: GraphState) -> str:
    profile = state.get("user_profile") or {}
    routine = state.get("user_routine") or {}
    if not isinstance(profile, dict):
        profile = {}
    if not isinstance(routine, dict):
        routine = {}
    products = _dict_items(state.get("retrieved_products") or [])
    ingredients = _dict_items(state.get("retrieved_ingredients") or [])
    matched_products = _dict_items(state.get("matched_products") or [])
    analysis = state.get("analysis_results") or {}
    input_status = state.get("input_status") or {}
    if not isinstance(analysis, dict):
        analysis = {}
    if not isinstance(input_status, dict):
        input_status = {}
    mode = state.get("mode", "analyse")

    lines = [f"## Conversation mode: {mode}"]
    missing_inputs = _text_items(input_status.get("missing") or analysis.get("missing_inputs") or [])
    if missing_inputs:
        lines.append(
            "\n## Missing user data\n"
            + "\n".join(f"- {item}" for item in missing_inputs)
            + "\nInstruction: acknowledge this immediately. Do not analyse, recommend, "
            "or infer routine details that are not present. Ask for the missing data needed "
            "to answer the user's request."
        )

    # Profile
    skin_type = profile.get("skin_type") or "unknown"
    concerns = _text_items(profile.get("concerns") or [])
    goals = _text_items(profile.get("goals") or [])
    allergens = _text_items(profile.get("allergens") or [])
    notes = str(profile.get("notes") or "").strip()
    avoid = [
        k.replace("avoid_", "").replace("_", " ")
        for k in ("avoid_fragrance", "avoid_alcohol", "avoid_parabens", "avoid_silicones")
        if profile.get(k)
    ]
    lines.append(
        f"\n## User profile\n"
        f"- Skin type: {skin_type}\n"
        f"- Concerns: {', '.join(concerns) or 'none stated'}\n"
        f"- Goals: {', '.join(goals) or 'none stated'}\n"
        f"- User notes: {notes or 'none stated'}\n"
        f"- Ingredients to avoid: {', '.join(avoid) or 'none'}\n"
        f"- Known allergens: {', '.join(allergens) or 'none'}"
    )

    previous_recommendations = _previous_assistant_recommendations(state)
    if previous_recommendations:
        lines.append(
            "\n## Previously recommended products\n"
            + "\n".join(f"- {name}" for name in previous_recommendations[:12])
            + "\nInstruction: Do not recommend these again, including different-size variants or near-duplicates."
        )

    advisor_notes = str(state.get("advisor_notes") or "").strip()
    if advisor_notes:
        lines.append(
            "\n## Advisor agent notes\n"
            + advisor_notes
            + "\nInstruction: Treat these as internal planning notes, not user-facing text."
        )

    product_analysis = _dict_items(state.get("product_recommendation_analysis") or [])
    selected_names = _text_items(state.get("agent_selected_product_names") or [])
    if product_analysis:
        lines.append("\n## Product recommendation analysis")
        for item in product_analysis[:2]:
            name = item.get("product_name") or ""
            brand = item.get("brand") or ""
            score = item.get("score", "")
            reasons = _text_items(item.get("reasons") or [])
            cautions = _text_items(item.get("cautions") or [])
            parts = [f"- {name}{' · ' + brand if brand else ''}"]
            if score != "":
                parts.append(f"fit score: {score}")
            if reasons:
                parts.append(f"reasons: {'; '.join(reasons[:3])}")
            if cautions:
                parts.append(f"cautions: {'; '.join(cautions[:3])}")
            lines.append(" | ".join(parts))
    if selected_names:
        lines.append(
            "\n## Agent-selected product recommendations\n"
            + "\n".join(f"- {name}" for name in selected_names[:2])
            + "\nInstruction: If recommending specific products, use these exact names."
        )

    ingredient_analysis = _dict_items(state.get("ingredient_recommendation_analysis") or [])
    if ingredient_analysis:
        lines.append("\n## Ingredient recommendation analysis")
        for item in ingredient_analysis[:2]:
            name = item.get("ingredient") or ""
            reasons = _text_items(item.get("reasons") or [])
            cautions = _text_items(item.get("cautions") or [])
            already = "yes" if item.get("already_in_routine") else "no"
            parts = [f"- {name}", f"already in routine: {already}"]
            if reasons:
                parts.append(f"reasons: {'; '.join(reasons[:3])}")
            if cautions:
                parts.append(f"cautions: {'; '.join(cautions[:3])}")
            lines.append(" | ".join(parts))

    # Routine products
    routine_items = _dict_items(routine.get("items") or routine.get("routine") or [])
    if routine_items:
        routine_ingredients = []
        seen_routine_ingredients = set()
        mode_label = routine.get("mode") or "unknown"
        ar_days = routine.get("ar_days") or {}
        lines.append(f"\n## Routine schedule\n- Mode: {mode_label}")
        if mode_label == "active_rest":
            active_days = ", ".join(ar_days.get("Active") or []) or "not specified"
            rest_days = ", ".join(ar_days.get("Rest") or []) or "not specified"
            lines.append(f"- Active days: {active_days}")
            lines.append(f"- Rest days: {rest_days}")
            lines.append(
                "- Instruction: Active PM and Rest PM are different nights. Analyse "
                "them separately and never treat products from those groups as layered together."
            )

        lines.append("\n## Current routine products")
        for item in routine_items:
            name = item.get("product_name", "Unknown")
            brand = item.get("brand") or ""
            time = item.get("time") or ""
            group = item.get("group") or ""
            scope = item.get("scope") or ""
            if scope == "ar" and group and time:
                slot = f"{group} {time}"
            else:
                slot = f"{group} {time}".strip() if (group or time) else ""
            key_ings = format_ingredient_names(_text_items(item.get("key_ingredients") or []))
            item_ingredients = format_ingredient_names(_text_items(item.get("ingredients") or []))
            for ingredient in key_ings + item_ingredients:
                key = ingredient.lower().strip()
                if key and key not in seen_routine_ingredients:
                    routine_ingredients.append(ingredient)
                    seen_routine_ingredients.add(key)
            ing_str = f" [{', '.join(key_ings[:4])}]" if key_ings else ""
            if not ing_str and item_ingredients:
                ing_str = f" [{', '.join(item_ingredients[:4])}]"
            lines.append(f"- {name}{' · ' + brand if brand else ''}{' (' + slot + ')' if slot else ''}{ing_str}")
        if routine_ingredients:
            lines.append(
                "\n## Already in routine ingredients\n"
                + "\n".join(f"- {ingredient}" for ingredient in routine_ingredients[:30])
                + "\nInstruction: Do not present these as new ingredients to try or add."
            )

    # Retrieved ingredient details
    if ingredients:
        lines.append("\n## Ingredient details from catalog")
        for ing in ingredients[:8]:
            name = format_ingredient_name(ing.get("inci_name") or "")
            aliases = format_ingredient_names(_text_items(ing.get("common_names") or []))
            functions = _text_items(ing.get("functions") or [])
            cautions = _text_items(ing.get("cautions") or [])
            suitable = _text_items(ing.get("suitable_for") or [])
            avoid_for = _text_items(ing.get("avoid_for") or [])
            guidance = ing.get("usage_guidance") or ""
            alias_str = f" (aka {', '.join(aliases[:2])})" if aliases else ""
            parts = [f"- **{name}**{alias_str}"]
            if functions:
                parts.append(f"functions: {', '.join(functions)}")
            if suitable:
                parts.append(f"suitable for: {', '.join(suitable)}")
            if avoid_for:
                parts.append(f"avoid for: {', '.join(avoid_for)}")
            if cautions:
                parts.append(f"cautions: {'; '.join(cautions)}")
            if guidance:
                parts.append(f"guidance: {guidance}")
            lines.append(" | ".join(parts))

    # Catalog-backed product candidates for recommendations and product analysis.
    if matched_products:
        if mode in {"recommend", "build"}:
            lines.append(
                "\n## Catalog recommendation candidates\n"
                "Instruction: Use these as available candidates, prioritising the agent's product analysis "
                "and exact selected product names. If none fit well, recommend an ingredient type, product "
                "category, or routine adjustment instead."
            )
        else:
            lines.append("\n## Products mentioned by user (analyse each in depth)")
        for idx, p in enumerate(matched_products[:10]):
            name = p.get("product_name", "")
            brand = p.get("brand") or ""
            cats = _text_items(p.get("categories") or [])
            key_ings = format_ingredient_names(_text_items(p.get("key_ingredients") or []))
            active_ings = format_ingredient_names(_text_items(p.get("active_ingredients") or []))
            ingredients = format_ingredient_names(_text_items(p.get("ingredients") or []))
            quantity = p.get("quantity") or ""
            source = p.get("source") or ""
            index_prefix = f"[{idx}] " if mode in {"recommend", "build"} else ""
            header = f"### {index_prefix}{name}" + (f" · {brand}" if brand else "")
            lines.append(header)
            if quantity:
                lines.append(f"- Size: {quantity}")
            if source:
                lines.append(f"- Catalog source: {source}")
            if cats:
                lines.append(f"- Categories: {', '.join(cats[:6])}")
            if key_ings:
                lines.append(f"- Key active ingredients: {', '.join(key_ings)}")
            elif active_ings:
                lines.append(f"- Key active ingredients: {', '.join(active_ings)}")
            else:
                lines.append("- Key active ingredients: not identified in catalog")
            if ingredients:
                lines.append(f"- Ingredient list excerpt: {', '.join(ingredients[:18])}")

    # Additional searched products (recommend mode)
    catalog_products = [p for p in products if p.get("source") == "catalog_search"]
    if catalog_products:
        lines.append("\n## Catalog search results")
        for p in catalog_products[:15]:
            name = p.get("product_name", "")
            brand = p.get("brand") or ""
            ings = format_ingredient_names(_text_items(
                p.get("key_ingredients") or p.get("active_ingredients") or p.get("ingredients") or []
            ))
            quantity = p.get("quantity") or ""
            size = f" ({quantity})" if quantity else ""
            lines.append(f"- {name}{' · ' + brand if brand else ''}{size}" + (f" [{', '.join(ings[:4])}]" if ings else ""))

    # Analysis
    conflicts = _dict_items(analysis.get("conflicts") or [])
    gaps = _text_items(analysis.get("gaps") or [])
    notes = _text_items(analysis.get("suitability_notes") or [])
    if conflicts:
        lines.append("\n## Detected conflicts")
        for c in conflicts:
            risk_level = str(c.get("risk_level") or "unknown")
            ingredients_involved = format_ingredient_names(_text_items(c.get("ingredients_involved") or []))
            products_involved = _text_items(c.get("products_involved") or [])
            lines.append(
                f"- [{risk_level.upper()}] {' + '.join(ingredients_involved)}"
                f" (products: {', '.join(products_involved)}): {c.get('reason') or ''}"
            )
    if gaps:
        lines.append("\n## Routine gaps\n" + "\n".join(f"- {g}" for g in gaps))
    if notes:
        lines.append("\n## Suitability notes\n" + "\n".join(f"- {n}" for n in notes))

    return "\n".join(lines)


def _fallback_response(state: GraphState) -> str:
    profile = state.get("user_profile") or {}
    skin_type = profile.get("skin_type") or "unknown"
    concerns = profile.get("concerns") or []
    analysis = state.get("analysis_results") or {}
    missing_inputs = (
        (state.get("input_status") or {}).get("missing")
        or analysis.get("missing_inputs")
        or []
    )
    gaps = analysis.get("gaps") or []

    if missing_inputs:
        if "profile" in missing_inputs and "routine" in missing_inputs:
            return (
                "I need your skin profile and saved routine before I can analyse this properly. "
                "Create or load a profile, then add the products you use in your routine."
            )
        if "profile" in missing_inputs:
            return (
                "I need your skin profile before I can tailor this properly. "
                "Create or load a profile with your skin type, concerns, and goals first."
            )
        return (
            "I need your saved routine before I can analyse what you use. "
            "Add the products in your AM and PM routine, then I can review conflicts, gaps, and priorities."
        )

    parts = [f"I can see your skin type is {skin_type}."]
    if concerns:
        parts.append(f"Your main concerns are: {', '.join(concerns[:3])}.")
    if gaps:
        parts.append("Worth noting: " + " ".join(gaps))
    parts.append(
        "Tell me the exact products you use and when, and I can give you a more specific analysis."
    )
    return " ".join(parts)


def _key_hint(k: str) -> str:
    if not k:
        return "missing"
    return f"{k[:7]}…{k[-4:]}" if len(k) > 12 else "***"


@traceable(name="generate_response")
def generate_response(state: GraphState) -> GraphState:
    """Draft the assistant answer from prepared context. No orchestration side effects."""
    api_key, model, base_url = load_openai_config("SKINIQ_GENERATE_RESPONSE_MODEL")

    if not api_key:
        log.warning("No API key found — using fallback response")
        return {**state, "draft_response": _fallback_response(state)}

    from langchain_core.messages import SystemMessage

    try:
        context_block = state.get("context_block") or _build_context_block(state)
        system_content = f"{SYSTEM_PROMPT}\n\n{context_block}"
    except Exception as exc:
        log.error("_build_context_block failed: %s", exc, exc_info=True)
        return {
            **state,
            "draft_response": f"**Debug — context build failed**\n{type(exc).__name__}: {exc}",
        }

    try:
        conversation = list(state.get("messages") or [])
        llm_messages = [SystemMessage(content=system_content)] + conversation
    except Exception as exc:
        log.error("Message assembly failed: %s", exc, exc_info=True)
        return {
            **state,
            "draft_response": f"**Debug — message assembly failed**\n{type(exc).__name__}: {exc}",
        }

    try:
        llm = _get_llm(model, api_key, base_url)
        response = llm.invoke(llm_messages)
        draft = response.content or ""
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        body = ""
        try:
            body = getattr(exc.response, "text", "")[:300]
        except Exception:
            pass

        detail = (
            f"**LLM error** — {type(exc).__name__}: {exc}"
            + (f"\nHTTP {status}" if status else "")
            + (f"\nResponse body: {body}" if body else "")
            + f"\n\nDebug: key={_key_hint(api_key)}, model={model}, "
            f"base_url={base_url or 'default (api.openai.com)'}"
        )
        log.error(
            "LLM call failed: %s | key=%s | model=%s | status=%s",
            exc, _key_hint(api_key), model, status,
        )
        return {**state, "draft_response": detail}

    return {**state, "draft_response": draft}
