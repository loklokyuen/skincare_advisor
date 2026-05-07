import sys
import uuid
from html import escape
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from graph.context_status import has_profile, has_routine
from utils.helpers import format_ingredient_names

# ── Session init ───────────────────────────────────────────────────────────────

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "chat_thread_id" not in st.session_state:
    st.session_state.chat_thread_id = str(uuid.uuid4())

if "chat_shortcuts_hidden" not in st.session_state:
    st.session_state.chat_shortcuts_hidden = False

if "pending_shortcut" not in st.session_state:
    st.session_state.pending_shortcut = None

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "interested_products" not in st.session_state:
    st.session_state.interested_products = []

if "chat_adding_product" not in st.session_state:
    st.session_state.chat_adding_product = None

if "advisor_prewarm_signature" not in st.session_state:
    st.session_state.advisor_prewarm_signature = None

# ── Helpers ────────────────────────────────────────────────────────────────────

STARTER_SHORTCUTS = [
    {
        "label": "Can you analyse my skincare routine?",
        "mode": "analyse",
        "prompt": (
            "Analyse my AM and PM routine. Identify what works, conflicts, redundancies, "
            "missing steps, and the one change I should prioritise."
        ),
    },
    {
        "label": "Can you build a routine for me?",
        "mode": "build",
        "prompt": (
            "Build a simple AM and PM skincare routine for my skin profile and goals. "
            "Keep it minimal and explain each step briefly."
        ),
    },
    {
        "label": "What ingredients should I try for my skin?",
        "mode": "learn",
        "prompt": (
            "Suggest the top 1-2 skincare ingredients that suit my skin profile, concerns, "
            "goals, and are not already active ingredients in my current routine. Keep it "
            "concise. For each ingredient, explain why it fits and give one usage or caution note."
        ),
    },
    {
        "label": "What should I add to my routine?",
        "mode": "recommend",
        "prompt": (
            "Suggest 1-2 targeted additions to my routine. Explain why they fit, what gap they fill, "
            "and where they belong (AM or PM). Prefer one if there is one clear best fit."
        ),
    },
]


def _interested_ids() -> set[str]:
    return {p["product_name"] for p in st.session_state.interested_products}


def _add_to_interested(product: dict):
    if product["product_name"] not in _interested_ids():
        st.session_state.interested_products.append(product)


def _routine_product_keys() -> set[tuple[str, str]]:
    keys = set()
    for item in st.session_state.get("routine", []) or []:
        name = (item.get("product_name") or "").strip().lower()
        brand = (item.get("brand") or "").strip().lower()
        if name:
            keys.add((name, brand))
    return keys


def _product_key(product: dict) -> tuple[str, str]:
    return (
        (product.get("product_name") or "").strip().lower(),
        (product.get("brand") or "").strip().lower(),
    )


def _product_is_in_routine(product: dict) -> bool:
    name, brand = _product_key(product)
    if not name:
        return False
    routine_keys = _routine_product_keys()
    return (name, brand) in routine_keys or any(
        name == item_name for item_name, _ in routine_keys
    )


def _routine_product_payload(product: dict) -> dict:
    return {
        "product_name": product.get("product_name") or "",
        "brand": product.get("brand") or "",
        "key_ingredients": product.get("key_ingredients") or [],
        "active_ingredients": product.get("active_ingredients") or [],
        "ingredients": product.get("ingredients") or [],
        "quantity": product.get("quantity") or "",
        "image_url": product.get("image_url") or "",
        "categories": product.get("categories") or [],
    }


def _render_add_to_routine_selector(product: dict, key_prefix: str):
    mode = st.session_state.get("routine_mode")
    if mode not in {"daily", "active_rest"}:
        st.warning("Choose a routine structure on the Routine page before adding products.")
        return

    p = _routine_product_payload(product)
    brand_str = f" · {p['brand']}" if p.get("brand") else ""
    with st.container(border=True):
        st.markdown(f"**Add to routine:** {p['product_name']}{brand_str}")
        new_item = None

        if mode == "daily":
            time_slot = st.radio(
                "Slot",
                ["AM", "PM"],
                horizontal=True,
                key=f"chat_slot_time_{key_prefix}",
            )
            new_item = {
                "id": str(uuid.uuid4()),
                "scope": "daily",
                "time": time_slot,
                **p,
            }
        elif mode == "active_rest":
            col_grp, col_time = st.columns(2)
            with col_grp:
                group = st.radio(
                    "Routine",
                    ["Active", "Rest"],
                    horizontal=True,
                    key=f"chat_slot_group_{key_prefix}",
                )
            with col_time:
                time_slot = st.radio(
                    "Time",
                    ["AM", "PM"],
                    horizontal=True,
                    key=f"chat_slot_time_{key_prefix}",
                )
            new_item = {
                "id": str(uuid.uuid4()),
                "scope": "ar",
                "time": time_slot,
                "group": group,
                **p,
            }

        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button(
                "✓ Add to Routine",
                type="primary",
                use_container_width=True,
                key=f"chat_confirm_add_{key_prefix}",
            ):
                if new_item and not _product_is_in_routine(new_item):
                    st.session_state.routine.append(new_item)
                st.session_state.chat_adding_product = None
                st.rerun()
        with col_cancel:
            if st.button(
                "✗ Cancel",
                use_container_width=True,
                key=f"chat_cancel_add_{key_prefix}",
            ):
                st.session_state.chat_adding_product = None
                st.rerun()


def _queue_shortcut(shortcut: dict):
    st.session_state.pending_shortcut = {
        "label": shortcut["label"],
        "mode": shortcut["mode"],
        "prompt": shortcut["prompt"],
    }
    st.session_state.chat_shortcuts_hidden = True


def _available_shortcuts(profile: dict, routine_context: dict) -> list[dict]:
    profile_ready = has_profile(profile)
    routine_ready = has_routine(routine_context)

    return [
        shortcut
        for shortcut in STARTER_SHORTCUTS
        if shortcut["mode"] != "analyse" or (profile_ready and routine_ready)
    ]


def _render_reddit_posts(posts: list[dict], query: str):
    if not posts:
        st.caption("No relevant Reddit discussions found.")
        return
    from services.reddit_service import summarize as summarize_reddit

    summary = summarize_reddit(query, posts)
    st.markdown(f"**Takeaway:** {summary['takeaway']}")
    if summary.get("liked"):
        st.markdown("**What people liked**")
        for quote in summary["liked"]:
            st.caption(f'"{quote}"')
    if summary.get("cautions"):
        st.markdown("**Common cautions**")
        for quote in summary["cautions"]:
            st.caption(f'"{quote}"')
    st.divider()
    for i, post in enumerate(posts):
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"**[{post['title']}]({post['url']})**")
            pos = post.get("positive_quote") or ""
            neg = post.get("negative_quote") or ""
            if pos:
                st.markdown(f'**Liked:** "{pos}"')
            if neg:
                st.markdown(f'**Cautious:** "{neg}"')
            if not pos and not neg and post.get("quote"):
                st.markdown(f'"{post["quote"]}"')
        with col2:
            st.caption(f"↑{post['score']}  💬{post['num_comments']}")
            if post.get("subreddit"):
                st.caption(f"r/{post['subreddit']}")
        if i < len(posts) - 1:
            st.divider()


def _render_community_button(community_request: dict, key_prefix: str):
    """Show a lazy-load button; only fetches Reddit when the user clicks."""
    query = (community_request or {}).get("query") or ""
    if not query:
        return

    posts_key = f"community_posts_{key_prefix}"
    btn_key = f"community_btn_{key_prefix}"

    if posts_key in st.session_state:
        with st.expander(f"💬 Community opinions on **{query}**", expanded=True):
            _render_reddit_posts(st.session_state[posts_key], query)
    else:
        if st.button(f"💬 Community opinions on **{query}**", key=btn_key):
            from services.reddit_service import search as reddit_search
            with st.spinner("Searching Reddit…"):
                posts = reddit_search(query, limit=3)
            st.session_state[posts_key] = posts
            st.rerun()


def _render_literature_section(literature_request: dict, key_prefix: str):
    query = literature_request.get("query") or ""
    if not query:
        return
    from services.literature_service import search as literature_search
    from services.literature_service import summarize as summarize_literature

    with st.spinner("Finding PubMed literature..."):
        papers = literature_search(query)
    if not papers:
        return

    with st.expander("Scientific literature", expanded=False):
        summary = summarize_literature(query, papers)
        st.markdown(f"**Takeaway:** {summary['takeaway']}")
        for point in summary.get("points") or []:
            st.caption(f"- {point}")
        st.divider()
        st.caption(f"PubMed results related to **{query}** and topical skin use")
        for i, paper in enumerate(papers):
            with st.container():
                st.markdown(f"**[{paper['title']}]({paper['url']})**")
                meta = " · ".join(
                    part for part in [paper.get("source") or "", paper.get("year") or ""] if part
                )
                if meta:
                    st.caption(meta)
                summary = paper.get("summary") or paper.get("snippet") or ""
                if summary:
                    st.markdown(f"**Key point:** {summary}")
                if i < len(papers) - 1:
                    st.divider()


def _render_evidence_summary(evidence_summary: dict | None):
    paragraph = (evidence_summary or {}).get("paragraph") or ""
    if not paragraph:
        return
    with st.expander("Evidence signals", expanded=True):
        st.markdown(paragraph)


def _render_product_cards(products: list[dict], key_prefix: str):
    """Render product cards beneath an assistant message."""
    loaded_products = [
        product
        for product in products or []
        if isinstance(product, dict)
        and product.get("product_name")
    ]
    if not loaded_products:
        return
    st.markdown("---")
    st.caption("Product suggestions")
    for idx, product in enumerate(loaded_products):
        with st.container(border=True):
            image_col, detail_col = st.columns([1, 2])
            with image_col:
                image_url = product.get("image_url") or ""
                if image_url:
                    st.image(image_url, use_container_width=True)
                else:
                    st.markdown(
                        '<div style="height:120px;background:#f5f5f5;border-radius:8px;'
                        'display:flex;align-items:center;justify-content:center;'
                        'color:#bbb;font-size:0.8rem">No image</div>',
                        unsafe_allow_html=True,
                    )

            with detail_col:
                name = product.get("product_name", "")
                brand = product.get("brand") or ""
                quantity = product.get("quantity") or ""
                cats = product.get("categories") or []
                st.markdown(f"**{name}**")
                meta = " · ".join(part for part in [brand, quantity] if part)
                if meta:
                    st.caption(meta)

                key_ings = product.get("key_ingredients") or product.get("active_ingredients") or []
                key_ings = format_ingredient_names(key_ings)
                if key_ings:
                    tags = " ".join(
                        f'<span style="background:#f0f4ff;color:#3f51b5;padding:2px 8px;'
                        f'border-radius:10px;font-size:0.72rem">{i}</span>'
                        for i in key_ings[:5]
                    )
                    st.markdown(tags, unsafe_allow_html=True)

                with st.expander("Details"):
                    if cats:
                        st.caption(", ".join(cats[:4]))
                    ingredients = product.get("ingredients") or []
                    ingredients = format_ingredient_names(ingredients)
                    if ingredients:
                        st.markdown("**Full ingredient list**")
                        st.caption(", ".join(ingredients[:80]))
                    if not cats and not ingredients:
                        st.caption("No additional catalog details available.")

                already_saved = name in _interested_ids()
                already_in_routine = _product_is_in_routine(product)
                interested_col, routine_col = st.columns(2)
                with interested_col:
                    btn_label = "✓ Saved" if already_saved else "♡ Interested"
                    btn_type = "secondary" if already_saved else "primary"
                    if st.button(
                        btn_label,
                        key=f"interested_{key_prefix}_{name}_{idx}",
                        use_container_width=True,
                        type=btn_type,
                        disabled=already_saved,
                    ):
                        _add_to_interested(product)
                        st.rerun()
                with routine_col:
                    routine_label = "✓ In routine" if already_in_routine else "Add to routine"
                    if st.button(
                        routine_label,
                        key=f"routine_{key_prefix}_{name}_{idx}",
                        use_container_width=True,
                        disabled=already_in_routine,
                    ):
                        st.session_state.chat_adding_product = {
                            "key": f"{key_prefix}_{idx}",
                            "product": _routine_product_payload(product),
                        }
                        st.rerun()

                pending_add = st.session_state.get("chat_adding_product") or {}
                if pending_add.get("key") == f"{key_prefix}_{idx}":
                    _render_add_to_routine_selector(
                        pending_add.get("product") or product,
                        f"{key_prefix}_{idx}",
                    )


def _render_assistant_markdown(content: str):
    """Render assistant Markdown, forcing H4 product headings to display as headings."""
    if not content:
        return

    buffer = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#### "):
            if buffer:
                st.markdown("\n".join(buffer))
                buffer = []
            heading = stripped.removeprefix("#### ").strip()
            st.markdown(f"<h4>{escape(heading)}</h4>", unsafe_allow_html=True)
        else:
            buffer.append(line)

    if buffer:
        st.markdown("\n".join(buffer))


# ── Page ───────────────────────────────────────────────────────────────────────

title_col, action_col = st.columns([4, 1])
with title_col:
    st.title("💬 Chat")
with action_col:
    if st.button("New chat", type="primary", use_container_width=True):
        st.session_state.chat_messages = []
        st.session_state.chat_thread_id = str(uuid.uuid4())
        st.session_state.chat_shortcuts_hidden = False
        st.session_state.pending_shortcut = None
        st.session_state.pending_prompt = None
        st.rerun()

profile = st.session_state.get("user_profile", {})
routine_context = {
    "mode": st.session_state.get("routine_mode"),
    "items": st.session_state.get("routine", []),
    "ar_days": st.session_state.get("ar_days", {}),
}

profile_ready = has_profile(profile)
routine_ready = has_routine(routine_context)

if not profile_ready:
    st.warning("Create or load a profile before using chat.")
    if st.button("Go to Profile", type="primary"):
        st.switch_page("pages/1_Profile.py")
    st.stop()

if not routine_ready:
    st.warning("Add your routine before using chat.")
    if st.button("Go to Routine", type="primary"):
        st.switch_page("pages/2_Routine.py")
    st.stop()

if profile.get("name"):
    st.caption(f"Talking to SkinIQ as **{profile['name']}**")

prewarm_signature = (
    profile.get("user_id"),
    profile.get("skin_type"),
    tuple(profile.get("concerns") or []),
    tuple(profile.get("goals") or []),
    profile.get("notes", ""),
    tuple(
        (
            item.get("product_name", ""),
            item.get("brand") or "",
            item.get("time") or "",
            item.get("scope") or "",
        )
        for item in routine_context.get("items") or []
        if isinstance(item, dict)
    ),
)
if st.session_state.advisor_prewarm_signature != prewarm_signature:
    from graph.recommendation_cache import prepare_advisor_answers

    prepare_advisor_answers(
        prompts=[
            shortcut
            for shortcut in STARTER_SHORTCUTS
            if shortcut["mode"] in {"recommend", "learn"}
        ],
        user_profile=profile,
        user_routine=routine_context,
        thread_id=st.session_state.chat_thread_id,
    )
    st.session_state.advisor_prewarm_signature = prewarm_signature

# Replay stored messages
for msg_idx, message in enumerate(st.session_state.chat_messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            _render_assistant_markdown(message["content"])
        else:
            st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_product_cards(message.get("matched_products") or [], f"history_{msg_idx}")
            if message.get("evidence_summary"):
                _render_evidence_summary(message.get("evidence_summary"))
            else:
                if message.get("literature_search_request"):
                    _render_literature_section(message["literature_search_request"], f"history_{msg_idx}")
                if message.get("community_search_request"):
                    _render_community_button(message["community_search_request"], f"history_{msg_idx}")

prompt = st.chat_input("Ask about your routine, ingredients, or product choices")
if prompt:
    st.session_state.chat_shortcuts_hidden = True
    st.session_state.pending_prompt = {
        "content": prompt,
    }
    st.rerun()

shortcuts_slot = st.empty()
if not st.session_state.chat_messages and not st.session_state.chat_shortcuts_hidden:
    shortcuts = _available_shortcuts(profile, routine_context)
    if shortcuts:
        with shortcuts_slot.container():
            with st.chat_message("user"):
                _, shortcut_col, _ = st.columns([1, 6, 1])
                with shortcut_col:
                    for shortcut in shortcuts:
                        st.button(
                            shortcut["label"],
                            key=f"starter_{shortcut['mode']}",
                            use_container_width=True,
                            on_click=_queue_shortcut,
                            args=(shortcut,),
                        )
else:
    shortcuts_slot.empty()

llm_prompt = None
user_content = None
selected_mode = None
if st.session_state.pending_prompt:
    pending_prompt = st.session_state.pending_prompt
    llm_prompt = pending_prompt["content"]
    user_content = pending_prompt["content"]
    st.session_state.pending_prompt = None

if st.session_state.pending_shortcut:
    shortcut = st.session_state.pending_shortcut
    llm_prompt = shortcut["prompt"]
    user_content = shortcut["label"]
    selected_mode = shortcut["mode"]
    st.session_state.pending_shortcut = None

if llm_prompt:
    st.session_state.chat_messages.append({
        "role": "user",
        "content": user_content,
    })
    with st.chat_message("user"):
        st.markdown(user_content)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            from graph.graph import run_advisor_with_progress

            sub_status = st.empty()

            def show_progress(message: str) -> None:
                sub_status.caption(message)

            result = run_advisor_with_progress(
                llm_prompt,
                user_profile=profile,
                user_routine=routine_context,
                thread_id=st.session_state.chat_thread_id,
                on_progress=show_progress,
                explicit_mode=selected_mode,
            )
            sub_status.empty()

        response_text = result["response"]
        matched_products = result["matched_products"]
        community_request = result.get("community_search_request")
        literature_request = result.get("literature_search_request")
        evidence_summary = result.get("evidence_summary")

        _render_assistant_markdown(response_text)
        _render_product_cards(matched_products, f"live_{len(st.session_state.chat_messages)}")
        if evidence_summary:
            _render_evidence_summary(evidence_summary)
        else:
            if literature_request:
                _render_literature_section(literature_request, f"live_{len(st.session_state.chat_messages)}")
            if community_request:
                _render_community_button(community_request, f"live_{len(st.session_state.chat_messages)}")

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": response_text,
        "matched_products": matched_products,
        "community_search_request": community_request,
        "literature_search_request": literature_request,
        "evidence_summary": evidence_summary,
    })
    st.rerun()
