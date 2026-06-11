# SkinIQ — Skincare Intelligence Platform

A personalised skincare advisor built with LangGraph, OpenAI, and Streamlit. It maintains a user profile and routine, retrieves relevant products and ingredient evidence, and answers questions in one of four modes depending on what the user is asking.

[Live Demo](https://skincare-advisor-405192745511.europe-west2.run.app)

## What it does

You set up a skin profile (type, concerns, goals, sensitivities) and log your current AM/PM routine. From there the chat can:

- **Analyse** your existing routine — conflicts, redundancies, what to prioritise
- **Recommend** specific products from the catalog that fit your profile
- **Build** a routine from scratch based on your goals
- **Teach** you what an ingredient does, how to use it, what to watch out for

Product recommendations come with ingredient breakdowns and can be saved to an "Interested" list, from which you can add them directly to your routine.

## Stack

| Layer        | Tech                                                                 |
| ------------ | -------------------------------------------------------------------- |
| Frontend     | Streamlit                                                            |
| LLM / graph  | LangGraph + LangChain OpenAI (`gpt-4o-mini`)                         |
| Embeddings   | OpenAI `text-embedding-ada-002` via pgvector                         |
| Database     | Supabase (PostgreSQL + pgvector) — originally on GCP Cloud SQL       |
| Evidence     | PubMed + Reddit                                                      |
| Product data | Scraped from Boots using ScraperAPI                                  |

Conversation state is checkpointed to Postgres so sessions persist across restarts.

## Project layout

```
app/
  Home.py                   landing page with profile-aware CTA
  pages/
    1_👤_Profile.py         skin profile setup and persistence
    2_🗓️_Routine.py         AM/PM routine builder
    3_💬_Chat.py            main chat interface
    4_💭_Interested.py      saved products, add to routine

graph/
  graph.py                  LangGraph workflow definition
  nodes/
    classify_intent.py      mode detection (analyse / build / learn / recommend)
    retrieve_context.py     product + ingredient lookup, profile-aware ranking
    analyse_routine.py      parallel analysis nodes (basics, products, ingredients)
    generate_response.py    final answer writer with tool use
    validate_response.py    post-generation cleanup

services/
  product_service.py        catalog search and vector similarity
  ingredient_service.py     ingredient lookup by name / function
  profile_service.py        profile save/load with backward-compat normalisation
  evidence_summary_service.py  PubMed + community evidence summarisation
  rag_service.py            retrieval-augmented context assembly
  reddit_service.py         community source search
  literature_service.py     PubMed search and relevance filtering

tools/
  product_tools.py          card extraction and product name matching
  advisor_tools.py          candidate ranking for products and ingredients
  analysis_tools.py         routine conflict and gap detection
  ingredient_tools.py       ingredient term extraction

data/
  products/scrape_boots.py  Boots catalog scraper (ScraperAPI)
  ingredients/              ingredient seed scripts and DB import
  db/                       embedding backfill utilities
```

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL with pgvector extension
- OpenAI API key
- ScraperAPI key (for product scraping, optional if you seed the DB directly)

### Environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=...
DB_HOST=localhost
DB_PORT=5432
DB_NAME=skincare_advisor
DB_USER=postgres
DB_PASSWORD=...

# Optional: separate models for chat vs embeddings vs evidence summaries
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_SUMMARY_MODEL=gpt-4o-mini

# LangSmith tracing (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=skincare-advisor

# ScraperAPI (for product scraping)
SCRAPER_API_KEY=...
```

### Install and run

```bash
pip install -e .

# seed ingredients (run once)
python data/ingredients/seed_ingredients.py

# scrape products into the DB (run once, takes a while)
python data/products/scrape_boots.py

# backfill embeddings after scraping
python data/db/backfill_embeddings.py

# start the app
streamlit run app/Home.py
```

## Running tests

```bash
pytest tests/
```

The test suite covers product card extraction, profile migration, candidate ranking, and evidence summary formatting. No external API calls — everything is monkeypatched.

## Notes

- The graph checkpointer will fall back to in-memory if Postgres is unreachable, so you can prototype without a DB (profiles and routine won't persist between sessions).
- Sensitive skin is stored as a `sensitive_skin` boolean separate from skin type, so it stacks with any of the four skin type options rather than replacing them.
- The advisor only covers topical skincare. It will politely decline questions about injectables, diet, oral medication, or anything medical.
