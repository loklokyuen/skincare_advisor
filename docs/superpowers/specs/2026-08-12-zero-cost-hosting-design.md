# Skincare Advisor Zero-Cost Hosting Design

## Goal

Move Skincare Advisor from GCP Cloud Run to Streamlit Community Cloud without adding a monthly hosting charge. Preserve the current application behavior and external database.

## Current State

- Cloud Run serves the Streamlit application from `app/Home.py`.
- The service uses PostgreSQL hosted outside GCP; the repository records the earlier move to Supabase.
- The service uses OpenAI and LangSmith configuration supplied through environment variables.
- The current GCP deployment remains the rollback target until the replacement passes verification.
- The latest application work lives on `advisor-followups-and-card-matching`, which tracks its GitHub branch.

## Target Architecture

Streamlit Community Cloud will deploy `app/Home.py` from a public GitHub branch based on `advisor-followups-and-card-matching`. Community Cloud will install the root `requirements.txt` under Python 3.11.

The external PostgreSQL database, OpenAI services, LangSmith tracing, and all application flows will remain unchanged. Community Cloud may hibernate the app after inactivity; the user accepts its wake-up delay in exchange for zero monthly hosting cost.

## Configuration

Community Cloud will receive the current Cloud Run variables as root-level secrets:

- `DB_HOST`
- `DB_PORT`
- `DB_NAME`
- `DB_USER`
- `DB_PASSWORD`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `LANGSMITH_TRACING_V2`
- `SKINIQ_GENERATE_RESPONSE_MODEL`
- `SKINIQ_PRODUCT_CARD_SELECTION_MODEL`
- `SKINIQ_INTENT_CLASSIFIER_MODEL`
- `SKINIQ_ONLINE_PRODUCT_AGENT_MODEL`

`DEPLOY_TIMESTAMP` is operational metadata and will not move. The migration will not rotate credentials because the user excluded rotation from this work. No secret value will enter Git history or documentation.

## Source Control

The isolated `codex/zero-cost-hosting` branch starts at the current `advisor-followups-and-card-matching` commit. It will contain only hosting compatibility changes, tests, and migration documentation. The branch will be pushed and used for the initial Community Cloud deployment. A pull request will preserve a reviewable path without changing the user's existing checkout.

## Validation

Before cutover, the migration will verify:

- The app starts under Python 3.11 with the declared dependencies.
- Home, Profile, Routine, Chat, and Interested pages render.
- The external database accepts a connection.
- A representative chat request completes and returns a response.
- Product cards and existing tests still behave as expected.
- No secret appears in logs or the repository diff.

## Cutover and Rollback

GCP will remain live until the Community Cloud URL passes validation. After validation, the Cloud Run service will be deleted. Until the final artifact cleanup, its existing tagged image can recreate the service if Community Cloud fails.

## GCP Cleanup

After the replacement remains healthy:

1. Delete the Skincare Advisor Cloud Run service.
2. Delete its obsolete Artifact Registry images and repository.
3. Delete project build buckets that contain no unique data.
4. Verify the Digital Futures project has no Cloud Run services, jobs, functions, Cloud SQL instances, or retained application data.
5. Detach the billing account from the empty project.

## Success Criteria

- The Community Cloud URL passes all validation checks.
- The hosting branch is pushed and reviewable.
- The existing external database and integrations remain unchanged.
- The Digital Futures GCP project has no billable application workloads or retained application data.

## Out of Scope

- Credential rotation.
- Application feature, prompt, model, or interface changes.
- Database schema or provider changes.
