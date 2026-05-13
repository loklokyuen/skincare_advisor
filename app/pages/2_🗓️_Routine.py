import sys
import uuid
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path so services/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.product_service import search_products, search_products_for_routine
from services.profile_service import normalize_profile, normalize_user_id, save_user_profile
from graph.context_status import has_profile
from utils.helpers import format_ingredient_names

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIMES = ["AM", "PM"]
MODE_LABELS = {"daily": "📅 Daily", "active_rest": "⚡ Active / Rest"}

DEFAULT_ACTIVE = ["Monday", "Thursday"]
DAY_PATTERNS = [
    ("2× / week", ["Monday", "Thursday"]),
    ("3× / week", ["Monday", "Wednesday", "Friday"]),
    ("Weekdays", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]),
    ("Weekends", ["Saturday", "Sunday"]),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_items(scope=None, time=None, group=None) -> list[dict]:
    result = st.session_state.routine
    if scope:
        result = [i for i in result if i.get("scope") == scope]
    if time:
        result = [i for i in result if i.get("time") == time]
    if group:
        result = [i for i in result if i.get("group") == group]
    return result


def remove_item(item_id: str):
    st.session_state.routine = [r for r in st.session_state.routine if r["id"] != item_id]


def clear_product_search():
    st.session_state.routine_search = ""
    st.session_state.routine_search_results = []
    st.session_state.routine_last_search = ""


def sync_routine_to_profile():
    profile = normalize_profile(st.session_state.get("user_profile") or {})
    user_id = normalize_user_id(profile.get("user_id") or st.session_state.get("profile_user_id"))
    if not profile.get("skin_type") or user_id == "default":
        st.session_state.routine_sync_status = "missing_profile"
        return

    profile.update({
        "user_id": user_id,
        "routine_mode": st.session_state.get("routine_mode"),
        "routine": st.session_state.get("routine", []),
        "ar_days": st.session_state.get("ar_days", {}),
    })
    if save_user_profile(user_id, profile):
        st.session_state.user_profile = profile
        st.session_state.profile_user_id = user_id
        st.session_state.routine_sync_status = "saved"
    else:
        st.session_state.routine_sync_status = "failed"


def init_ar_days():
    if "ar_days" not in st.session_state:
        st.session_state.ar_days = {
            "Active": DEFAULT_ACTIVE,
            "Rest": [d for d in DAYS if d not in DEFAULT_ACTIVE],
        }


def do_migrate(target: str):
    src = st.session_state.routine_mode
    items = st.session_state.routine

    if src == "daily" and target == "active_rest":
        for i in items:
            if i.get("scope") == "daily":
                i["scope"] = "ar"
                i["group"] = "Active"
        init_ar_days()
        st.session_state.routine_mode = "active_rest"

    elif src == "active_rest" and target == "daily":
        new_items = []
        for i in items:
            if i.get("scope") == "ar" and i.get("group") == "Active":
                ni = {k: v for k, v in i.items() if k != "group"}
                ni["scope"] = "daily"
                new_items.append(ni)
            elif i.get("scope") != "ar":
                new_items.append(i)
        st.session_state.routine = new_items
        st.session_state.routine_mode = "daily"

    st.session_state.adding_product = None
    st.rerun()


def product_html(
    name: str,
    brand: str = "",
    key_ingredients: list[str] | None = None,
    quantity: str = "",
) -> str:
    """Single HTML block for name + brand + ingredient tags — avoids Streamlit's inter-element spacing."""
    parts = [f"<strong>{name}</strong>"]
    meta = " · ".join(part for part in [brand, quantity] if part)
    if meta:
        parts.append(f'<span style="color:#999;font-size:0.82rem">{meta}</span>')
    if key_ingredients:
        key_ingredients = format_ingredient_names(key_ingredients)
        tags = " ".join(
            f'<span style="background:#f0f4ff;color:#3f51b5;padding:1px 8px;'
            f'border-radius:10px;font-size:0.72rem">{i}</span>'
            for i in key_ingredients[:4]
        )
        parts.append(f'<div style="margin-top:2px">{tags}</div>')
    return "<br>".join(parts)


@st.cache_data(ttl=600, show_spinner=False)
def _enrich_lookup(name: str, brand: str) -> list[dict]:
    if not name:
        return []
    try:
        return search_products(name, limit=3, use_obf_fallback=False)
    except Exception:
        return []


def enrich_product_details(item: dict) -> dict:
    if item.get("ingredients") or item.get("active_ingredients") or item.get("quantity"):
        return item
    name = (item.get("product_name") or "").lower().strip()
    brand = (item.get("brand") or "").lower().strip()
    matches = _enrich_lookup(item.get("product_name", ""), item.get("brand") or "")
    for match in matches:
        match_name = (match.get("product_name") or "").lower().strip()
        match_brand = (match.get("brand") or "").lower().strip()
        if match_name == name and (not brand or match_brand == brand):
            return {**match, **{k: v for k, v in item.items() if v}}
    return {**matches[0], **{k: v for k, v in item.items() if v}} if matches else item


@st.dialog("Product Details")
def product_details_dialog(item: dict):
    item = enrich_product_details(item)
    st.markdown(f"### {item['product_name']}")
    meta = " · ".join(part for part in [item.get("brand"), item.get("quantity")] if part)
    if meta:
        st.caption(meta)
    if item.get("image_url"):
        st.image(item["image_url"], width=180)
    cats = item.get("categories") or []
    if cats and isinstance(cats, list):
        st.markdown(" ".join(f"`{c}`" for c in cats[:4]))
    st.divider()
    ings = item.get("key_ingredients") or item.get("active_ingredients") or []
    ings = format_ingredient_names(ings)
    if ings:
        st.markdown("**Key Ingredients**")
        for ing in ings:
            st.markdown(f"- {ing}")
    full_ings = item.get("ingredients") or []
    full_ings = format_ingredient_names(full_ings)
    if full_ings:
        st.markdown("**Full ingredient list**")
        st.caption(", ".join(full_ings[:80]))
    if not ings and not full_ings:
        st.caption("No ingredient data available yet.")


def product_card(item: dict):
    with st.container(border=True):
        col_info, col_btns = st.columns([3, 1])
        with col_info:
            st.markdown(
                product_html(
                    item["product_name"],
                    item.get("brand", ""),
                    item.get("key_ingredients") or item.get("active_ingredients"),
                    item.get("quantity", ""),
                ),
                unsafe_allow_html=True,
            )
        with col_btns:
            if st.button("More", key=f"info_{item['id']}", type="primary"):
                st.session_state.show_product_details = item
                st.rerun()
            if st.button("✕", key=f"rm_{item['id']}", help="Remove"):
                remove_item(item["id"])
                st.rerun()


# ── Init session state ─────────────────────────────────────────────────────────
for k, v in {
    "routine_mode": None,
    "routine": [],
    "adding_product": None,
    "show_product_details": None,
    "routine_search_results": [],
    "routine_last_search": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Normalize / retire legacy mode values
_legacy = {"Daily": "daily", "Weekly": None, "Active / Rest": "active_rest"}
if st.session_state.routine_mode in _legacy:
    new_mode = _legacy[st.session_state.routine_mode]
    st.session_state.routine_mode = new_mode
    if new_mode is None:  # weekly retired — start fresh
        st.session_state.routine = []
    st.rerun()

# Open product details dialog if triggered
if st.session_state.show_product_details:
    item_to_show = st.session_state.show_product_details
    st.session_state.show_product_details = None
    product_details_dialog(item_to_show)

st.title("🗓️ My Routine")
mode = st.session_state.routine_mode
profile = st.session_state.get("user_profile", {})

if not has_profile(profile):
    st.warning("Create or load a profile before building your routine.")
    if st.button("Go to Profile", type="primary"):
        st.switch_page("pages/1_👤_Profile.py")
    st.stop()

sync_status = st.session_state.pop("routine_sync_status", None)
if sync_status == "saved":
    st.success("Routine synced to your profile.")
elif sync_status == "missing_profile":
    st.warning("Create or load a profile before syncing your routine.")
elif sync_status == "failed":
    st.error("Could not sync your routine to Cloud SQL. Check the database connection and try again.")

if mode is not None:
    sync_label = "Sync routine to profile"
    if profile.get("user_id"):
        st.caption(f"Current profile: `{profile['user_id']}`")
    st.button(sync_label, type="primary", on_click=sync_routine_to_profile)

# ── First visit: choose routine type ──────────────────────────────────────────
if mode is None:
    st.markdown("### How do you want to structure your routine?")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### 📅 Daily")
            st.caption("Same products every morning and evening. Simple and consistent.")
            if st.button("Choose", key="choose_daily", type="primary", width='stretch'):
                st.session_state.routine_mode = "daily"
                st.rerun()
    with col2:
        with st.container(border=True):
            st.markdown("#### ⚡ Active / Rest")
            st.caption(
                "Two routines: one for active days (e.g. Mon + Thu), one for the rest. "
                "Great if you use strong actives that need rest days."
            )
            if st.button("Choose", key="choose_ar", type="primary", width='stretch'):
                st.session_state.routine_mode = "active_rest"
                init_ar_days()
                st.rerun()
    st.stop()

scope = "ar" if mode == "active_rest" else mode
current_items = [i for i in st.session_state.routine if i.get("scope") == scope]

# ── Mode header + switch ───────────────────────────────────────────────────────
other_mode = "active_rest" if mode == "daily" else "daily"
col_label, col_switch = st.columns([3, 1])
with col_label:
    st.markdown(f"**{MODE_LABELS[mode]}**")
with col_switch:
    if st.button(f"Switch to {MODE_LABELS[other_mode]}", width='stretch'):
        do_migrate(other_mode)



# ── Add product ────────────────────────────────────────────────────────────────
search_col, search_btn, cancel_btn = st.columns([4, 1, 1])
with search_col:
    query = st.text_input(
        "Search products",
        placeholder="Product name, brand, or need...",
        key="routine_search",
        label_visibility="collapsed",
    )
with search_btn:
    search_clicked = st.button("Search", type="primary", width='stretch')
with cancel_btn:
    st.button("Cancel", width='stretch', on_click=clear_product_search)

if search_clicked:
    cleaned_query = query.strip()
    if len(cleaned_query) < 2:
        st.session_state.routine_search_results = []
        st.session_state.routine_last_search = ""
    else:
        with st.spinner("Searching products..."):
            st.session_state.routine_search_results = search_products_for_routine(cleaned_query)
            st.session_state.routine_last_search = cleaned_query

if st.session_state.routine_last_search:
    results = st.session_state.routine_search_results
    if not results:
        st.info("No products found.")
    else:
        with st.container(height=300):
            for row in results:
                r_key = f"{row['product_name']}|{row.get('brand', '')}"
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(
                        product_html(
                            row["product_name"],
                            row.get("brand", ""),
                            row.get("key_ingredients") or row.get("active_ingredients"),
                            row.get("quantity", ""),
                        ),
                        unsafe_allow_html=True,
                    )
                with c2:
                    if st.button("Add", key=f"add_{r_key}", type="primary", width='stretch'):
                        st.session_state.adding_product = {
                            "product_name": row["product_name"],
                            "brand": row.get("brand") or "",
                            "key_ingredients": row.get("key_ingredients") or [],
                            "active_ingredients": row.get("active_ingredients") or [],
                            "ingredients": row.get("ingredients") or [],
                            "quantity": row.get("quantity") or "",
                            "image_url": row.get("image_url") or "",
                            "categories": row.get("categories") or [],
                        }
                        st.rerun()

# ── Slot selector ──────────────────────────────────────────────────────────────
if st.session_state.adding_product:
    p = st.session_state.adding_product
    brand_str = f" · {p['brand']}" if p.get("brand") else ""
    slot_col, _ = st.columns([2, 1])
    with slot_col:
        with st.container(border=True):
            st.markdown(f"**{p['product_name']}**{brand_str}")

            time_slots: list[str] = []
            group = None
            if mode == "daily":
                col_am, col_pm = st.columns(2)
                with col_am:
                    if st.checkbox("AM", value=True, key="slot_am"):
                        time_slots.append("AM")
                with col_pm:
                    if st.checkbox("PM", key="slot_pm"):
                        time_slots.append("PM")
            elif mode == "active_rest":
                col_grp, col_am, col_pm = st.columns(3)
                with col_grp:
                    group = st.radio("Routine", ["Active", "Rest"], horizontal=True, key="slot_group")
                with col_am:
                    if st.checkbox("AM", value=True, key="slot_am"):
                        time_slots.append("AM")
                with col_pm:
                    if st.checkbox("PM", key="slot_pm"):
                        time_slots.append("PM")

            if not time_slots:
                st.caption("Pick at least one slot (AM or PM).")

            product_fields = {k: p[k] for k in (
                "product_name", "brand", "key_ingredients", "active_ingredients",
                "ingredients", "quantity", "image_url", "categories",
            )}

            def _new_item(slot: str) -> dict:
                base = {"id": str(uuid.uuid4()), "time": slot, **product_fields}
                if mode == "active_rest":
                    base["scope"] = "ar"
                    base["group"] = group
                else:
                    base["scope"] = "daily"
                return base

            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button(
                    "Add to Routine",
                    type="primary",
                    width='stretch',
                    disabled=not time_slots,
                ):
                    for slot in time_slots:
                        st.session_state.routine.append(_new_item(slot))
                    st.session_state.adding_product = None
                    st.rerun()
            with col_cancel:
                if st.button("Cancel", key="cancel_slot", width='stretch'):
                    st.session_state.adding_product = None
                    st.rerun()

# st.divider()

# ── Dashboard ──────────────────────────────────────────────────────────────────
st.subheader("My Routine")

if not current_items:
    st.info("No products yet. Search above to add your first product.")
    st.stop()

# Daily ────────────────────────────────────────────────────────────────────────
if mode == "daily":
    col_am, col_pm = st.columns(2)
    with col_am:
        st.markdown("#### ☀️ Morning (AM)")
        am_items = get_items(scope="daily", time="AM")
        for item in am_items:
            product_card(item)
        if not am_items:
            st.caption("Nothing here yet.")
    with col_pm:
        st.markdown("#### 🌙 Evening (PM)")
        pm_items = get_items(scope="daily", time="PM")
        for item in pm_items:
            product_card(item)
        if not pm_items:
            st.caption("Nothing here yet.")

# Active / Rest ────────────────────────────────────────────────────────────────
elif mode == "active_rest":
    ar_days = st.session_state.get("ar_days", {
        "Active": DEFAULT_ACTIVE,
        "Rest": [d for d in DAYS if d not in DEFAULT_ACTIVE],
    })

    with st.expander("⚙️ Day assignments"):
        pat_cols = st.columns(len(DAY_PATTERNS))
        for i, (label, active_days) in enumerate(DAY_PATTERNS):
            with pat_cols[i]:
                is_current = set(ar_days.get("Active", [])) == set(active_days)
                if st.button(
                    label, key=f"pat_{i}", width='stretch',
                    type="primary" if is_current else "secondary",
                ):
                    st.session_state.ar_days = {
                        "Active": active_days,
                        "Rest": [d for d in DAYS if d not in active_days],
                    }
                    st.rerun()
        st.write("")
        new_active = st.multiselect(
            "Custom active days", DAYS,
            default=ar_days.get("Active", []),
            key="ar_custom",
        )
        if new_active and new_active != ar_days.get("Active"):
            st.session_state.ar_days = {
                "Active": new_active,
                "Rest": [d for d in DAYS if d not in new_active],
            }
            st.rerun()

    for group, icon, label in [("Active", "⚡", "Active Days"), ("Rest", "🛋️", "Rest Days")]:
        g_days = ar_days.get(group, [])
        day_str = ", ".join(d[:3] for d in g_days) if g_days else "no days assigned"
        st.markdown(f"#### {icon} {label}")
        st.caption(day_str)
        c_am, c_pm = st.columns(2)
        with c_am:
            st.markdown("**☀️ AM**")
            am_items = get_items(scope="ar", time="AM", group=group)
            for item in am_items:
                product_card(item)
            if not am_items:
                st.caption("Nothing here yet.")
        with c_pm:
            st.markdown("**🌙 PM**")
            pm_items = get_items(scope="ar", time="PM", group=group)
            for item in pm_items:
                product_card(item)
            if not pm_items:
                st.caption("Nothing here yet.")
        st.write("")
