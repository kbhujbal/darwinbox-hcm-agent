"""Central place for model names, pricing, and tunable thresholds.

Pricing is illustrative (public Gemini list pricing at time of writing, USD
per 1M tokens) and is only used to produce comparable, order-of-magnitude
cost numbers for the naive-vs-optimized benchmark. Update PRICING if your
account's actual rates differ.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TRACES_DIR = ROOT_DIR / "traces"
CHROMA_DIR = ROOT_DIR / "chroma_db"
CHECKPOINT_DB = ROOT_DIR / "conversation_state.sqlite"

# --- Models -----------------------------------------------------------
# Flash: cheap/fast model used for the optimized pipeline (routing fallback,
# policy answer generation, action slot-filling).
#
# NOTE: model names below were picked by probing which models this account's
# API key actually has usable quota for (see README) — "gemini-2.0-flash"
# and every genuine Pro-tier model returned 429 RESOURCE_EXHAUSTED (0 free
# quota) on this key, so PRO_MODEL is the best available stand-in for "a
# larger, non-cost-optimized model" rather than true Gemini Pro. Swap these
# for your account's actual available models.
FLASH_MODEL = "gemini-flash-lite-latest"
# Pro: larger model used only by the *naive* baseline in the cost benchmark,
# to represent an "everything through the big model" pipeline.
PRO_MODEL = "gemini-flash-latest"
EMBEDDING_MODEL = "gemini-embedding-001"

# USD per 1,000,000 tokens.
PRICING = {
    FLASH_MODEL: {"input": 0.10, "output": 0.40},
    PRO_MODEL: {"input": 1.25, "output": 10.00},
    EMBEDDING_MODEL: {"input": 0.15, "output": 0.0},
}

# --- RAG ----------------------------------------------------------------
RAG_TOP_K = 3
RAG_CHUNK_MAX_TOKENS = 400
RAG_CHUNK_OVERLAP_TOKENS = 40
RAG_COLLECTION_NAME = "hr_policy"
# Cosine distance (0 = identical, 2 = opposite) above which a retrieved chunk
# is considered "not a confident match" and excluded from grounding context.
RAG_DISTANCE_FLOOR = 0.55

# --- Routing --------------------------------------------------------------
# Below this confidence, the LLM fallback router asks a clarifying question
# instead of guessing which sub-agent should handle the request.
ROUTING_CONFIDENCE_THRESHOLD = 0.55

# --- Tools ------------------------------------------------------------
TOOL_MAX_RETRIES = 3
TOOL_BACKOFF_BASE_SECONDS = 0.4
MOCK_API_FAILURE_RATE = 0.15
