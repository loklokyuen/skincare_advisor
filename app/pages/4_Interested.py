import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.helpers import format_ingredient_names

# ── Session init ───────────────────────────────────────────────────────────────

if "interested_products" not in st.session_state:
    st.session_state.interested_products = []

if "interested_adding_product" not in st.session_state:
    st.session_state.interested_adding_product = None

# ── Helpers ────────────────────────────────────────────────────────────────────

def _remove_product(product_name: str):
    st.session_state.interested_products = [
        p for p in st.session_state.interested_products
        if p["product_name"] != product_name
    ]


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
                key=f"interested_slot_time_{key_prefix}",
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
                    key=f"interested_slot_group_{key_prefix}",
                )
            with col_time:
                time_slot = st.radio(
                    "Time",
                    ["AM", "PM"],
                    horizontal=True,
                    key=f"interested_slot_time_{key_prefix}",
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
                key=f"interested_confirm_add_{key_prefix}",
            ):
                if new_item and not _product_is_in_routine(new_item):
                    st.session_state.routine.append(new_item)
                st.session_state.interested_adding_product = None
                st.rerun()
        with col_cancel:
            if st.button(
                "✗ Cancel",
                use_container_width=True,
                key=f"interested_cancel_add_{key_prefix}",
            ):
                st.session_state.interested_adding_product = None
                st.rerun()


def _render_card(product: dict, idx: int):
    with st.container(border=True):
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

        name = product.get("product_name", "")
        brand = product.get("brand") or ""
        cats = product.get("categories") or []

        st.markdown(f"**{name}**")
        if brand:
            st.caption(brand)

        key_ings = product.get("key_ingredients") or []
        key_ings = format_ingredient_names(key_ings)
        if key_ings:
            tags = " ".join(
                f'<span style="background:#f0f4ff;color:#3f51b5;padding:2px 8px;'
                f'border-radius:10px;font-size:0.72rem">{i}</span>'
                for i in key_ings[:5]
            )
            st.markdown(tags, unsafe_allow_html=True)

        if cats:
            st.caption(", ".join(cats[:3]))

        already_in_routine = _product_is_in_routine(product)
        add_col, remove_col = st.columns(2)
        with add_col:
            routine_label = "✓ In routine" if already_in_routine else "Add to routine"
            if st.button(
                routine_label,
                key=f"routine_{name}_{idx}",
                use_container_width=True,
                disabled=already_in_routine,
                type="primary"
            ):
                st.session_state.interested_adding_product = {
                    "key": f"interested_{idx}",
                    "product": _routine_product_payload(product),
                }
                st.rerun()
        with remove_col:
            if st.button("Remove", key=f"remove_{name}_{idx}", use_container_width=True):
                _remove_product(name)
                st.session_state.interested_adding_product = None
                st.rerun()

        pending_add = st.session_state.get("interested_adding_product") or {}
        if pending_add.get("key") == f"interested_{idx}":
            _render_add_to_routine_selector(
                pending_add.get("product") or product,
                f"interested_{idx}",
            )


# ── Page ───────────────────────────────────────────────────────────────────────

st.title("♡ Interested")

products = st.session_state.interested_products

if not products:
    st.info(
        "No saved products yet. Chat with SkinIQ and tap **♡ Interested** on any product card to save it here."
    )
    st.stop()

st.caption(f"{len(products)} product{'s' if len(products) != 1 else ''} saved")

cols = st.columns(3)
for idx, product in enumerate(products):
    with cols[idx % 3]:
        _render_card(product, idx)
