import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import APP_CONFIG

st.set_page_config(
    page_title="SkinIQ — Skincare Intelligence",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

st.markdown("""
# 🧴 Welcome to SkinIQ

**Your personalised skincare intelligence platform.**

| Page | What you can do |
|------|----------------|
| 👤 **Profile** | Build your skin profile — type, concerns, sensitivities |
| 🗓️ **Routine** | Build and manage your AM/PM skincare routine |
| 💬 **Chat** | Get personalised skincare advice |

---
> Get started by setting up your **skin profile** — it personalises your entire experience.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Products", "400+", "in database")
with col2:
    st.metric("Ingredients", "66", "tracked")
with col3:
    st.metric("Skin Concerns", "12", "categories")
