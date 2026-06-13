# Design: `LOKAL=true` — self-contained Docker demo

## Goal

Make the RAG ⇄ GraphRAG demo runnable by anyone with **only a Mistral API key**.
Setting `LOKAL=true` in `.env` switches the app from the author's external services
(Qdrant + LightRAG behind ingresses) to **bundled Docker containers** that ship with
pre-built data. Clone → add `MISTRAL_API_KEY` → `docker compose up` → open the browser.

The repo target is `git@github.com:straussbastian/RAG-graphRAG_example.git`.

## Toggle (additive — `LOKAL=false`/unset keeps today's behavior exactly)

`LOKAL` is parsed in both config modules.

- **`LOKAL=true`**: the app talks to the bundled containers. The user sets only
  `MISTRAL_API_KEY`. All other URLs/keys default to the local compose services:
  - `QDRANT_URL` → `http://qdrant:6333`, `QDRANT_API_KEY` → `""`
  - `LIGHTRAG_URL` → `http://lightrag:9621`, `LIGHTRAG_API_KEY` → `local-demo-key`
  - `normalize_qdrant_url()`'s forced `:443` / stripped `:6333` is **skipped** in local
    mode (it would otherwise break `http://qdrant:6333`).
- **`LOKAL=false`/unset**: unchanged — external URLs/keys come from `.env`, `:443`
  normalization stays.

Mistral is always required (it is the one key end-users supply).

## Container topology (`docker-compose.yml`, local stack = the committed default)

```
qdrant    qdrant/qdrant            — volume + healthcheck
lightrag  ghcr.io/hkuds/lightrag   — Mistral binding (below), pre-built rag_storage volume, healthcheck
demo      the app                  — depends_on qdrant+lightrag healthy; seeds Qdrant at startup
```

LightRAG → Mistral (OpenAI-compatible binding), set on the `lightrag` service:
`LLM_BINDING=openai`, `LLM_BINDING_HOST=https://api.mistral.ai/v1`,
`LLM_MODEL=mistral-small-latest`, `EMBEDDING_BINDING=openai`,
`EMBEDDING_BINDING_HOST=https://api.mistral.ai/v1`, `EMBEDDING_MODEL=mistral-embed`,
`EMBEDDING_DIM=1024`, LLM+embedding keys = `${MISTRAL_API_KEY}`,
`LIGHTRAG_API_KEY=local-demo-key`.

Fallback if Mistral's `/v1/embeddings` misbehaves under the openai binding: use a
LightRAG-native Mistral embedding binding at build time. Result is identical; only an
implementation detail of the one-time build.

## Data seeding (pre-built, committed to the repo)

- **Qdrant.** `export_vectors.py` (run once by the author, using the external Qdrant
  credentials) scrolls all 249 vectors + payload → `data/qdrant_seed.json`. At app
  startup in local mode, a seed step creates collection `LizardDocu` (1024-dim, cosine)
  and upserts from the seed file. Idempotent (skips if already populated). IDs match
  `points.json`, so live-query placement keeps working.
- **LightRAG.** The author builds the graph once locally: bring up the `lightrag`
  container, ingest the reconstructed LizardDocu (concatenated from the 249 chunk
  contents in `points.json`) via `POST /documents/text`, wait for extraction →
  `lightrag_storage/` fills up and is committed. Then re-export `data/graph.json` from
  *this* local LightRAG (`build_graph.py` against local) so the static graph and the
  live highlight share the same node IDs.

## Secrets & repo hygiene

- **`.gitignore`** excludes `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`,
  `.DS_Store`, `data/reducer.pkl`, and the large standalone `embeddings.html` /
  `graph.html` (regenerable, ~6.6 MB).
- **Committed data**: `data/points.json`, `data/graph.json`, `data/qdrant_seed.json`,
  `lightrag_storage/`.
- **`env.example`**: `MISTRAL_API_KEY=` + `LOKAL=true` active; external `QDRANT_*` /
  `LIGHTRAG_*` present but commented under a "only needed when LOKAL=false" section.
- **README** gets a "Local Demo (Docker)" quickstart at the top.
- `.gitignore` is created **before** `git init` so `.env` is never staged.

## Tests

- Existing config tests stay green (non-local branch unchanged).
- New tests: local-mode config branch (defaults, no `:443` forcing) + the Qdrant seed
  helper (collection created, idempotent re-seed).

## One-time author cost (accepted)

Building the LightRAG graph from ~293K characters costs a few hundred thousand Mistral
tokens once (~10–40 min). After that, end-users pay nothing at startup — only their own
live queries.

## Implementation order (cheap/safe first, expensive/uncertain last)

1. Config toggle + tests.
2. `.gitignore` + `env.example`.
3. `docker-compose.yml` (3 services) + README quickstart.
4. `export_vectors.py` → `data/qdrant_seed.json` (cheap, no LLM).
5. Qdrant startup seeding + test.
6. LightRAG one-time build → `lightrag_storage/` + re-export `graph.json` (expensive).
7. Full `docker compose up` verification across all 5 views + live queries.
8. `git init`, commit (confirm `.env` unstaged), push.
