import logging
import sys
import warnings
from pathlib import Path

# Suppress transformers import noise when torchvision is absent.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r"Accessing `__path__` from .*")

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import APP_CONFIG
from graph.context_status import has_profile

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
| 💭 **Interested** | Review saved products and add them to your routine |

---
Get started by setting up your **skin profile**.
""")

profile_ready = has_profile(st.session_state.get("user_profile", {}))
cta_label = "Continue to Chat" if profile_ready else "Create or Load Profile"
cta_page = "pages/3_💬_Chat.py" if profile_ready else "pages/1_👤_Profile.py"

if st.button(cta_label, type="primary"):
    st.switch_page(cta_page)
