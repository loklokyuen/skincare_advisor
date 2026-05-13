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
            max_completion_tokens=600,
            reasoning_effort="minimal",
            verbosity="low",
            streaming=True,
        )
    )

_PROMPT_INTRO = """You are a practical, friendly, evidence-aware skincare advisor, focused on topical skincare solutions only.

Draft the final user-facing answer using the provided context: saved profile,
routine, catalog data, retrieved ingredient details, any tool results in this
turn, and conservative general skincare knowledge for uncatalogued terms.
Do not mention any internal actions or local catalog."""

_PROMPT_TOOL_USE = """\
Tool use:
- Call select_product_cards EXACTLY ONCE with the 1-2 exact product names you
  will recommend in the final answer, then write the final answer in the same
  response. The names you pass MUST match what you write verbatim.
- Do not call any tool more than once. Do not split tool calls across turns.
- Product/ingredient ranking is already provided in context; do not request it."""

_PROMPT_CORE = """\
Core rules:
- Be concise, warm, conservative, and specific.
- Do not invent products, ingredients, studies, reviews, formulation details, or user facts.
- Start from the user's existing routine and profile.
- Do not answer anything besides topical skincare products, such as injection, laser, or exercise/diet.
- If there is a new term, assume it is about topical skincare products/ingredients, if it is not, politely decline to answer.
- For a named skincare ingredient or product that is not in the provided catalog
  context, answer the user's high-level educational question from general
  skincare knowledge.
- If required data from user is missing, ask for that data instead of guessing.
- Do not diagnose medical conditions; for severe, persistent, prescription, pregnancy,
  infection, swelling, or burning concerns, advise speaking to a clinician.
- Do not mention tools, searches, the catalog, database, Reddit, reviews, or evidence
  follow-up unless actual evidence text is provided in context."""

_PROMPT_ROUTINE_PRODUCT = """\
Routine/product rules:
- The only products the user currently uses are listed under "Current routine products".
- Catalog candidates, search results, product analysis, and selected product names are
  possible recommendations only. Never describe them as already in the user's routine.
- Products already in the user's routine are not new recommendations.
- Recommend at most 1-2 products or ingredients.
- If recommending two, present them as separate suitable options without ranking labels.
- Do not call products "primary", "secondary", "optional", "alternate", or "alternative".
- When candidate products are present, recommend those specific products by exact
  name. Do not substitute a generic "a hyaluronic acid serum" or "any peptide
  moisturiser" line when a real candidate exists."""

_PROMPT_ANALYSE = """\
Analyse-mode rules:
- Base "what works", conflicts, overlap, and improvement ideas only on
  "Current routine products", "Already in routine ingredients", and detected
  routine analysis. Do not use catalog candidates as evidence of what the user uses.
- Do not restate the user's saved profile or routine back to them. They already
  know what they use and what their goals are. Refer to a product or goal only
  when it is load-bearing for the point you are making.
- Skip any section that has nothing to report. Do not write "Redundancies: none"
  or "No conflicts detected" — just omit the section.
- Do not call minor omissions "gaps". Phrase them as optional improvements or
  next-step tweaks.
- Do not discuss redundancies or overlap unless they are likely to worsen
  irritation, dryness, acne, or make the routine harder to follow.
- Treat cleanser and sunscreen as baseline staples that the user almost certainly
  already has. Do not list them as "missing steps" or "gaps" unless the user has
  explicitly asked about them or has stated they do not use one.
- Focus on improving their routine instead of filling gaps.
- For routine analysis, keep the answer to 2-3 short sections and about 120-180
  words unless the user asks for detail.
- Prioritise practical changes over diagnostic labels. Prefer "Try..." or
  "Move..." phrasing over "Gap:" or "Redundancy:"."""

_PROMPT_STYLE = """\
Response style:
- Use Markdown headings when the answer has more than one topic.
- Prefer H4 headings (`####`) for sections, products, and ingredients.
- An H4 heading must name a specific real product (brand + product) or a single
  ingredient. Never use abstract labels like "Option —", "Choice —", "Idea —",
  "Alternative —", "Pick —", or "Plan —" as a heading.
- Do not put category, format, or concentration descriptors in parentheses
  after a heading (e.g. "(serum, 10-15%)", "(leave-on BHA gel)"). Put that
  detail in a bullet underneath instead.
- Use short bullets instead of dense paragraphs.
- Each bullet should usually be 6-16 words.
- Each bullet should contain one clear idea only.
- Avoid semicolons and long chained clauses.
- Use **bold** for important ingredients, skin concerns, goals, risks, or routine context.
- Use *italic* for gentle cautions, uncertainty, or practical notes.
- Use profile-aware phrasing, such as "with oily, acne-prone skin" or "for a dry skin profile".
- Keep the answer concise by default.
- End with at most one natural next step, and only when useful."""

_PROMPT_LEARN = """\
Learn-mode rules:
- Default to teaching about the term. Treat any unknown ingredient name as a
  cosmetic / topical skincare ingredient unless it is clearly non-skincare
  (food, exercise, oral medication, surgery, injection-only with no topical
  form). When in doubt, answer.
- Mesotherapy or injectable ingredients (e.g. PDRN, polynucleotides, certain
  peptides) often have topical formulations too — explain the topical use
  case from general skincare knowledge instead of declining.
- If catalog/RAG context is empty for the term, rely on general skincare
  knowledge: what it is, how it works on skin, typical use, cautions.
- Keep the answer educational and concise. Do not push products unless asked.
- Only decline when the term is plainly off-topic (e.g. "what is squat depth",
  "what is amoxicillin dose")."""


_PROMPT_EXAMPLE_RECOMMEND = """\
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
- Avoid applying it at the same time as retinol until you know your skin tolerates it.*"""


def _build_system_prompt(mode: str) -> str:
    sections: list[str] = [_PROMPT_INTRO, _PROMPT_CORE]
    if mode in {"recommend", "build"}:
        sections.extend([_PROMPT_TOOL_USE, _PROMPT_ROUTINE_PRODUCT, _PROMPT_STYLE, _PROMPT_EXAMPLE_RECOMMEND])
    elif mode == "analyse":
        sections.extend([_PROMPT_ROUTINE_PRODUCT, _PROMPT_ANALYSE, _PROMPT_STYLE])
    elif mode == "learn":
        sections.extend([_PROMPT_LEARN, _PROMPT_STYLE])
    else:
        sections.extend([_PROMPT_ROUTINE_PRODUCT, _PROMPT_STYLE])
    return "\n\n".join(sections)


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


_PREV_REC_STOP = {
    "for", "the", "and", "with", "a", "an", "to", "of", "in", "by", "on", "or",
}


_ROUTINE_ACTIVE_RE = re.compile(
    "|".join(re.escape(token) for token in (
        "retinol", "retinal", "retinoid", "retinaldehyde", "retinyl", "retinoate",
        "tretinoin", "adapalene", "bakuchiol",
        "niacinamide", "azelaic", "salicylic", "glycolic", "lactic", "mandelic",
        "kojic", "arbutin", "tranexamic", "ferulic",
        "ascorbate", "ascorbic", "ascorbyl", "vitamin c", "vitamin a", "vitamin b",
        "tocopherol", "panthenol", "allantoin",
        "peptide", "matrixyl", "hexapeptide", "tripeptide", "tetrapeptide",
        "ceramide", "squalane", "shea", "centella", "cica", "madecassoside",
        "hyaluronic", "sodium hyaluronate", "polyglutamic", "snail",
        "benzoyl peroxide", "sulfur", "tea tree", "zinc",
        "resveratrol", "alpha lipoic", "coenzyme q10", "ubiquinone",
        "glycerin", "urea", "lactobionic", "gluconolactone",
    ))
)


def _is_routine_active(name: str) -> bool:
    return bool(_ROUTINE_ACTIVE_RE.search(name.lower()))


def _previous_assistant_recommendations(state: GraphState) -> list[str]:
    """Product names recommended in earlier assistant turns, skipping ingredient-style headings."""
    from langchain_core.messages import AIMessage
    from tools.product_tools import _response_product_suggestion_queries
    from utils.product_match import product_family_name

    ingredient_families: set[str] = set()
    for ingredient in state.get("retrieved_ingredients") or []:
        if not isinstance(ingredient, dict):
            continue
        for key in ("inci_name", "display_name"):
            value = ingredient.get(key)
            if not value:
                continue
            family = product_family_name(str(value))
            if family:
                ingredient_families.add(family)
            for token in re.split(r"[,/]", str(value)):
                family = product_family_name(token)
                if family:
                    ingredient_families.add(family)

    names: list[str] = []
    seen: set[str] = set()

    for message in state.get("messages") or []:
        if not isinstance(message, AIMessage):
            continue
        content = message.content if isinstance(message.content, str) else ""
        if not content:
            continue
        for query in _response_product_suggestion_queries(content):
            name = re.sub(r"\s+", " ", query).strip(" -:;.")
            if not name:
                continue
            family = product_family_name(name)
            distinctive = {
                t for t in family.split()
                if t not in _PREV_REC_STOP and len(t) > 1
            }
            if len(distinctive) < 3:
                continue
            if any(
                ing_family and (ing_family == family or ing_family in family)
                for ing_family in ingredient_families
            ):
                continue
            key = name.lower()
            if key in seen:
                continue
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
    is_learn = mode == "learn"

    if is_learn:
        message_lower = (state.get("message") or "").lower()
        if message_lower:
            relevant: list[dict] = []
            for ing in ingredients:
                names = [str(ing.get("inci_name") or ""), str(ing.get("display_name") or "")]
                names.extend(str(n) for n in (ing.get("common_names") or []) if n)
                if any(n and n.lower() in message_lower for n in names):
                    relevant.append(ing)
            ingredients = relevant

    lines = [f"## Conversation mode: {mode}"]
    if not is_learn:
        lines.append(
            "\n## Context boundary\n"
            "- The only products the user currently uses are listed under \"Current routine products\".\n"
            "- Catalog candidates, search results, product analysis, and selected product names are possible recommendations only."
        )
    missing_inputs = _text_items(input_status.get("missing") or analysis.get("missing_inputs") or [])
    if missing_inputs:
        lines.append(
            "\n## Missing user data\n"
            + "\n".join(f"- {item}" for item in missing_inputs)
            + "\nInstruction: acknowledge this immediately. Do not analyse, recommend, "
            "or infer routine details that are not present. Ask for the missing data needed "
            "to answer the user's request."
        )

    skin_type = profile.get("skin_type") or "unknown"
    sensitive_skin = "yes" if profile.get("sensitive_skin") else "no"
    if is_learn:
        lines.append(
            f"\n## User profile\n"
            f"- Skin type: {skin_type}\n"
            f"- Sensitive skin: {sensitive_skin}"
        )
    else:
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
            f"- Sensitive skin: {sensitive_skin}\n"
            f"- Concerns: {', '.join(concerns) or 'none stated'}\n"
            f"- Goals: {', '.join(goals) or 'none stated'}\n"
            f"- User notes: {notes or 'none stated'}\n"
            f"- Ingredients to avoid: {', '.join(avoid) or 'none'}\n"
            f"- Known allergens: {', '.join(allergens) or 'none'}"
        )

    if not is_learn:
        previous_recommendations = _previous_assistant_recommendations(state)
        if previous_recommendations:
            lines.append(
                "\n## Previously recommended products\n"
                + "\n".join(f"- {name}" for name in previous_recommendations[:12])
                + "\nInstruction: Do not recommend these again, including different-size variants or near-duplicates."
            )

    product_analysis = _dict_items(state.get("product_recommendation_analysis") or [])
    if product_analysis:
        lines.append("\n## Product recommendation analysis")
        for item in product_analysis[:4]:
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
        lines.append(
            "Instruction: pick 1-2 candidates that best match the user's specific "
            "phrasing, concerns, and goals — not strictly the top score. Use exact "
            "product names from this list. Pass them to select_product_cards."
        )

    ingredient_analysis = (
        _dict_items(state.get("ingredient_recommendation_analysis") or [])
        if mode not in {"recommend", "build"}
        else []
    )
    if is_learn and ingredient_analysis:
        message_lower = (state.get("message") or "").lower()
        if message_lower:
            ingredient_analysis = [
                item for item in ingredient_analysis
                if str(item.get("ingredient") or "").lower() in message_lower
            ]
    if ingredient_analysis:
        lines.append("\n## Ingredient recommendation analysis")
        for item in ingredient_analysis[:3]:
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

    routine_items = _dict_items(routine.get("items") or routine.get("routine") or [])
    if routine_items and not is_learn:
        routine_ingredients = []
        seen_routine_ingredients = set()
        mode_label = routine.get("mode") or "unknown"
        ar_days = routine.get("ar_days") or {}
        if mode_label == "active_rest":
            lines.append(f"\n## Routine schedule\n- Mode: {mode_label}")
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
        active_routine_ingredients = [
            ingredient for ingredient in routine_ingredients
            if _is_routine_active(ingredient)
        ]
        if active_routine_ingredients:
            lines.append(
                "\n## Already in routine ingredients\n"
                + "\n".join(f"- {ingredient}" for ingredient in active_routine_ingredients[:20])
                + "\nInstruction: Do not present these as new ingredients to try or add."
            )

    drop_guidance = mode in {"recommend", "build"}
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
            if guidance and not drop_guidance:
                parts.append(f"guidance: {guidance}")
            lines.append(" | ".join(parts))

    if matched_products and not is_learn:
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
            index_prefix = f"[{idx}] " if mode in {"recommend", "build"} else ""
            header = f"### {index_prefix}{name}" + (f" · {brand}" if brand else "")
            lines.append(header)
            if quantity:
                lines.append(f"- Size: {quantity}")
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

    catalog_products = [p for p in products if p.get("source") == "catalog_search"]
    if catalog_products and not is_learn:
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

    conflicts = _dict_items(analysis.get("conflicts") or []) if not is_learn else []
    gaps = _text_items(analysis.get("gaps") or []) if not is_learn else []
    notes = _text_items(analysis.get("suitability_notes") or []) if not is_learn else []
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
        lines.append(
            "\n## Improvement ideas\n"
            + "\n".join(f"- {g}" for g in gaps)
            + "\nInstruction: Present these as optional ways to improve the routine, "
            "not as major gaps unless clearly serious."
        )
    if notes:
        lines.append(
            "\n## Practical notes\n"
            + "\n".join(f"- {n}" for n in notes)
            + "\nInstruction: Include only if it changes how the user should use the routine."
        )

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
            "Add the products in your AM and PM routine, then I can suggest the most useful improvements."
        )

    parts = [f"I can see your skin type is {skin_type}."]
    if profile.get("sensitive_skin"):
        parts.append("I will also treat your skin as sensitive.")
    if concerns:
        parts.append(f"Your main concerns are: {', '.join(concerns[:3])}.")
    if gaps:
        parts.append("One useful improvement: " + " ".join(gaps[:1]))
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
    """Draft the assistant answer from prepared context."""
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    api_key, model, base_url = load_openai_config("SKINIQ_GENERATE_RESPONSE_MODEL")

    if not api_key:
        log.warning("No API key found — using fallback response")
        return {**state, "draft_response": _fallback_response(state)}

    mode = state.get("mode") or "analyse"

    try:
        context_block = state.get("context_block") or _build_context_block(state)
        system_content = f"{_build_system_prompt(mode)}\n\n{context_block}"
    except Exception as exc:
        log.error("_build_context_block failed: %s", exc, exc_info=True)
        return {
            **state,
            "draft_response": f"**Debug — context build failed**\n{type(exc).__name__}: {exc}",
        }

    conversation = list(state.get("messages") or [])

    selected_names: list[str] = []
    tools = []
    if mode in {"recommend", "build"}:
        @tool
        def select_product_cards(product_names: list[str]) -> dict:
            """Commit to the 1-2 exact product names you will recommend. The UI renders these as product cards. Call exactly once."""
            selected_names[:] = [str(n).strip() for n in product_names[:2] if str(n).strip()]
            return {"selected": list(selected_names)}

        tools = [select_product_cards]

    try:
        llm = _get_llm(model, api_key, base_url)
        writer = create_agent(llm, tools=tools, system_prompt=system_content, name="skincare_writer")
        result = writer.invoke({"messages": conversation}, config={"recursion_limit": 8})
        messages = result.get("messages") or []
        draft = ""
        if messages:
            content = getattr(messages[-1], "content", "")
            draft = content if isinstance(content, str) else ""
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

    return {
        **state,
        "draft_response": draft,
        "agent_selected_product_names": selected_names,
    }
