# SLMForge

Plug-and-play Small Language Model (SLM) builder. Point it at a folder of data, click Build, and SLMForge produces a fine-tuned LoRA adapter, an eval report, and a ready-to-use local serving endpoint with auto-generated usage docs.

Built by a 5-student pod at Newton School of Technology (NST) as part of the Summer Profile Building Drive 2026.

---

## What It Does

SLMForge turns any folder of data into a domain-specific SLM in two ways. On the terminal, `cd` into your data and run `slmforge build`. In the browser, run `slmforge ui`, paste file paths into stacked rows with a `+` button, and click Build. The same engine sits behind both surfaces -- producing identical artefacts (LoRA adapter, model card, eval report, USAGE.md) in a predictable on-disk layout. When the build finishes you see the verdict against a fixed ship gate, plus copy-paste curl / Python / JS snippets to call your new SLM from anywhere that already speaks the OpenAI client protocol.

**For Builders:** One command turns raw data into a deployable SLM. No hyperparameter spelunking, no glue code, no "wait, which version of vLLM?". Sensible defaults; override anything.

**For Operators:** Multi-LoRA serving means one base model + many fine-tuned adapters from one endpoint. Hot-swap between them per request with `model: <build_id>`. Same OpenAI-shape API; drop into existing code by changing one line.

**Internal-data safe:** No real NST data ever enters the repo. Repo policy, pre-commit hook, CI guard, and the dataset-loader contract all block internal paths. Recipes train on synthetic + permissively-licensed public datasets only. The internal team adds an `internal:` source adapter later, in their own deployment, without touching engine code.

---

## Key Features

- Plug-and-play SLM builder -- one CLI command or one click in localhost UI turns a folder of data into a fine-tuned SLM
- Two surfaces, same engine -- CLI (Typer + rich) for terminals, React + Vite web app for everyone else
- Multi-format ingestion -- jsonl, csv, parquet, txt-folder all auto-detected and loaded
- Auto task detection -- classification, summarisation, QA, instruction, chat; user can always override
- LoRA + QLoRA fine-tune harness -- Phi-3-mini, Llama 3.1 8B, Qwen 2.5 7B, DeepSeek V3 distill out of the box
- Run planner with VRAM + walltime estimator before training starts
- Built-in eval harness -- per-task metric suite + optional LLM-as-judge + ship-gate verdict + Pareto plot
- Multi-LoRA serving -- vLLM with hot-swap adapters and an OpenAI-compatible `/v1/chat/completions` endpoint
- Auto-generated USAGE.md -- every build ships curl + Python + JS snippets that just work
- Live progress streaming -- WebSocket-driven, used by both the CLI (rich live panel) and the Web UI (live chart + log)
- Checkpoint + resume -- pre-empted shared-GPU runs restart from the last epoch
- Four bundled recipes -- Interview Coach, Feedback Summariser, Question Assistant, Iterative Editor, all on synthetic + public data
- Internal-data safe -- repo policy + CI guard + dataset-loader contract block any internal NST data from ever entering the pod's working tree
- Drop-in upgrade path -- internal team enables the `internal:` source adapter later and re-runs the same harness on real data

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Backend | FastAPI + WebSocket |
| CLI | Typer + rich |
| Fine-tune | Hugging Face transformers + peft (LoRA) + bitsandbytes (QLoRA) |
| Base models | Phi-3-mini, Llama 3.1 8B, Qwen 2.5 7B, DeepSeek V3 distill |
| Serving | vLLM (OpenAI-compatible API) |
| Datasets | Hugging Face datasets + polars + pyarrow |
| Retrieval (Question Assistant recipe) | sentence-transformers + FAISS + Meilisearch |
| Eval | sklearn metrics + ROUGE + sacrebleu + LLM-as-judge harness |
| Database | SQLite by default; Postgres-ready |
| Cache / Queue | Redis + RQ |
| Frontend | React + Vite + TypeScript |
| Migrations | Alembic |
| Deployment | Docker + Docker Compose |
| CI | GitHub Actions / Drone |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+ (for the Web UI)
- Docker + Docker Compose
- Hugging Face account + token (some base models are gated): `huggingface-cli login`
- Shared-GPU access (filed via Issue 4 in M1; required to actually train)

### Setup

```bash
# Clone
git clone git@github.com:newton-school-ai/slmforge.git
cd slmforge

# Python environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install                   # hooks for ASCII + internal-data guard

# Environment variables
cp .env.example .env
# Edit .env with your HF_TOKEN (free at huggingface.co/settings/tokens)

# Services (Redis only by default; SQLite is local file)
docker compose up -d redis

# Database
alembic upgrade head

# Run backend
uvicorn slmforge.api.main:app --reload --port 8000
curl http://localhost:8000/health    # {"status":"ok"}

# Run frontend (separate terminal)
cd frontend && npm install && npm run dev
# Opens http://localhost:3000
```

### Build Your First SLM

```bash
# A bundled recipe (no data required)
slmforge build --recipe feedback_summariser --auto

# Or from any folder of data
cd /path/to/your/data
slmforge build

# Or from the Web UI (recommended for first run)
slmforge ui
```

### Talk to Your SLM

```bash
slmforge serve <build_id>
# Endpoint: http://localhost:8000/v1/chat/completions

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<build_id>",
    "messages": [{"role": "user", "content": "Summarise: ..."}]
  }'
```

From Python (drop into existing OpenAI client code by changing one line):
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="<build_id>",
    messages=[{"role": "user", "content": "Summarise: ..."}],
)
print(resp.choices[0].message.content)
```

---

## Project Structure

```
src/slmforge/
  api/                  FastAPI routes (build, run, eval, serve, library)
  engine/               build orchestrator, planner, run state
  data/                 sources/, ingest, builder, source_guard
  task/                 detector, templates, per-task metric picker
  finetune/             registry, LoRA, QLoRA, train, checkpoint
  eval/                 metric suite, LLM-judge, ship-gate, Pareto, report
  serve/                vLLM wrapper, adapter registry, usage_doc generator
  cli/                  Typer entrypoint, prompts, rich-based TTY
  utils/                shared helpers

src/recipes/
  interview_coach/      synthetic transcripts grounded in a public bank
  feedback_summariser/  public docs + synthetic faculty-style comments
  question_assistant/   synthetic queries + histories over public bank
  iterative_editor/     public drafts + labelled feedback operators

frontend/src/
  pages/                Home, NewBuild, Run, EvalReport, Serve, Library
  components/           SourceRow, AdvancedDisclosure, LiveChart, LogStream, MetricCard

docs/
  ENGINE_API.md         Contract between CLI / UI / engine (locked in M1)
  DATASET_CARD_TEMPLATE.md
  MODEL_CARD_TEMPLATE.md
```

Full architecture, pipeline, and milestone breakdown lives in `_internal/PROJECT_CONTEXT.md`.

---

## Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) -- Branch strategy, PR workflow, coding standards, data policy
- [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) -- Full setup, daily workflow, building an SLM, troubleshooting
- [MILESTONES.md](MILESTONES.md) -- M1-M10 with acceptance criteria and defense questions
- [ISSUES_TRACKER.md](ISSUES_TRACKER.md) -- All 42 issues with Why / What / Files / Test / Acceptance / Depends-on
- [GITHUB_ISSUES.md](GITHUB_ISSUES.md) -- Full issue bodies (source for `scripts/create_github_issues.sh`)
- [POD_GUIDE.md](POD_GUIDE.md) -- Pod roles, sprint timeline, Q&A, PR review checklist
- [docs/ENGINE_API.md](docs/ENGINE_API.md) -- Engine API contract (v0)

---

## Pod

| Role | Owns |
|------|------|
| Maintainer | Repo architecture, engine API contract, M1 + M10 |
| Contributor 1 | M2 Data Layer + Source Adapters |
| Contributor 2 | M3 + M4 Task System + Fine-Tune Engine |
| Contributor 3 | M5 + M6 Eval Engine + Serving Engine |
| Contributor 4 | M7 + M8 CLI + Web UI |
| All four | M9 Bundled Recipes + M10 Release |

Faculty Q&A every 2-3 days. Any contributor may be asked to defend any choice across the whole repo -- the pod model expects every contributor to understand every line.

---

## Milestones

| # | Name | Status |
|---|------|--------|
| M1 | Scaffold + Engine API Contract | Todo |
| M2 | Data Layer + Source Adapters | Todo |
| M3 | Task Type System | Todo |
| M4 | Fine-Tune Engine | Todo |
| M5 | Eval Engine | Todo |
| M6 | Serving Engine | Todo |
| M7 | CLI | Todo |
| M8 | Web UI | Todo |
| M9 | Four Bundled Recipes | Todo |
| M10 | Packaging + Production Polish | Todo |

---

## Bundled Recipes

| Recipe | Task | Built on |
|--------|------|----------|
| Interview Coach | Multi-turn chat (DSA / SysDesign / Behavioural) | Public LeetCode-style HF datasets + pod-authored synthetic transcripts |
| Feedback Summariser | Summarisation (doc + comments -> rework summary) | cnn_dailymail / samsum / xsum + pod-authored comment templates |
| Question Assistant | Instruction + retrieval (chat over a question bank) | Public LeetCode-style bank + synthetic queries + synthetic histories |
| Iterative Editor | Multi-turn chat (draft -> revise -> refine) | IteraTeR / WritingPrompts / OSS-licensed essays + labelled feedback operators |

Every recipe is reproducible with `slmforge build --recipe <name> --auto` and serves as a CI smoke test for the engine.

---

## License

MIT

---

NST Engineering - SLMForge | Summer Profile Building Drive 2026
