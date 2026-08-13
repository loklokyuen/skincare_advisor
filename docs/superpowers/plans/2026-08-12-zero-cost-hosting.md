# Skincare Advisor Zero-Cost Hosting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Skincare Advisor GCP service with a free Streamlit Community Cloud deployment while preserving the current Supabase database and application behavior.

**Architecture:** Streamlit Community Cloud deploys `app/Home.py` from `codex/zero-cost-hosting` under Python 3.11. Existing Supabase, OpenAI, and LangSmith connections remain unchanged. GCP remains live until the replacement passes page, database, and representative chat checks.

**Tech Stack:** Python 3.11, Streamlit Community Cloud, Supabase PostgreSQL, OpenAI API, LangSmith, GitHub CLI, GCP CLI

---

### Task 1: Verify deployment compatibility

**Files:**
- Create: `tests/test_streamlit_deployment.py`
- Test: `tests/test_streamlit_deployment.py`

- [ ] **Step 1: Write the deployment contract test**

```python
from pathlib import Path
import unittest


class StreamlitDeploymentTests(unittest.TestCase):
    def test_entrypoint_and_dependencies_exist(self):
        self.assertTrue(Path("app/Home.py").is_file())
        requirements = Path("requirements.txt").read_text()
        self.assertIn("streamlit", requirements)
        self.assertIn("psycopg2-binary", requirements)
        self.assertIn("langgraph-checkpoint-postgres", requirements)

    def test_secrets_are_not_tracked(self):
        self.assertFalse(Path(".env").exists())
        self.assertFalse(Path(".streamlit/secrets.toml").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the deployment contract**

Run: `python3.11 -m unittest -v tests/test_streamlit_deployment.py`

Expected: two tests pass.

- [ ] **Step 3: Create an isolated Python 3.11 environment and install dependencies**

Run:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt pytest
```

Expected: installation completes without dependency errors.

- [ ] **Step 4: Run the existing tests**

Run: `.venv/bin/python -m pytest -q`

Expected: all repository tests pass.

- [ ] **Step 5: Commit the deployment contract**

```bash
git add tests/test_streamlit_deployment.py
git commit -m "test: cover Streamlit deployment contract"
```

### Task 2: Publish the hosting branch

**Files:**
- No new repository file changes.

- [ ] **Step 1: Check the complete diff**

Run: `git diff --check advisor-followups-and-card-matching..HEAD`

Expected: no whitespace errors.

- [ ] **Step 2: Push the branch**

Run: `git push -u origin codex/zero-cost-hosting`

Expected: GitHub accepts the branch.

- [ ] **Step 3: Create a pull request**

Create a pull request from `codex/zero-cost-hosting` to `main`. Document that the deployment initially uses the branch because it includes the current follow-up and product-card work.

- [ ] **Step 4: Review the pull request**

Confirm no secret values, local environments, caches, or unrelated files appear.

### Task 3: Deploy Streamlit Community Cloud

**Files:**
- No repository file changes.

- [ ] **Step 1: Create the app**

In Streamlit Community Cloud, choose repository `loklokyuen/skincare_advisor`, branch `codex/zero-cost-hosting`, entry point `app/Home.py`, and Python 3.11.

- [ ] **Step 2: Add the current service configuration as root-level secrets**

Copy these existing Cloud Run values without printing them: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_TRACING_V2`, `SKINIQ_GENERATE_RESPONSE_MODEL`, `SKINIQ_PRODUCT_CARD_SELECTION_MODEL`, `SKINIQ_INTENT_CLASSIFIER_MODEL`, and `SKINIQ_ONLINE_PRODUCT_AGENT_MODEL`.

- [ ] **Step 3: Wait for deployment**

Expected: Community Cloud reports the app running and provides a `streamlit.app` URL.

- [ ] **Step 4: Test the public app**

Verify Home, Profile, Routine, Chat, and Interested pages render. Confirm the database connection succeeds and one representative chat prompt returns a response with product-card behavior intact.

### Task 4: Remove Skincare Advisor from GCP

**Files:**
- No repository file changes.

- [ ] **Step 1: Delete the Cloud Run service**

Delete `skincare-advisor` from `vigilant-host-491314-g7` in `europe-west2` only after the Community Cloud checks pass.

- [ ] **Step 2: Inventory retained project resources**

List Artifact Registry repositories, Cloud Build buckets, Cloud Run source buckets, functions, jobs, schedulers, Cloud SQL instances, and VMs.

- [ ] **Step 3: Remove obsolete artifacts**

Delete container images, artifact repositories, and build/source buckets that exist only for the deleted service.

- [ ] **Step 4: Verify the project is empty**

Confirm no Cloud Run services, jobs, functions, schedulers, Cloud SQL instances, VMs, application buckets, or Artifact Registry repositories remain.

- [ ] **Step 5: Detach billing**

Disable billing for `vigilant-host-491314-g7` and verify billing is disabled.

### Task 5: Final verification

**Files:**
- No repository file changes.

- [ ] **Step 1: Run tests again**

Run:

```bash
python3.11 -m unittest -v tests/test_streamlit_deployment.py
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Verify the public site**

Confirm the Community Cloud URL responds and all five pages still render after GCP deletion.

- [ ] **Step 3: Verify Supabase remains external**

Confirm a live database-backed page loads and no GCP database resource exists.
