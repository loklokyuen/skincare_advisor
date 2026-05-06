from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(ENV_PATH, override=True)

if os.getenv("LANGSMITH_TRACING_V2") and not os.getenv("LANGSMITH_TRACING"):
    os.environ["LANGSMITH_TRACING"] = os.environ["LANGSMITH_TRACING_V2"]
if os.getenv("LANGSMITH_TRACING_V2") and not os.getenv("LANGCHAIN_TRACING_V2"):
    os.environ["LANGCHAIN_TRACING_V2"] = os.environ["LANGSMITH_TRACING_V2"]
if os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]

try:
    from langsmith import traceable
except ImportError:
    def traceable(func=None, **_kwargs):
        if func is None:
            return lambda wrapped: wrapped
        return func
