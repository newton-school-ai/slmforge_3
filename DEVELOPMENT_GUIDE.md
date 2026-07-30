# Development Guide - SLMForge

## Prerequisites

- Python 3.11+
- Node.js 20+ and npm
- Docker and Docker Compose
- Git (SSH key configured with GitHub)
- GitHub CLI: `brew install gh && gh auth login`
- Hugging Face account + token (for some base models): `huggingface-cli login`
- Shared-GPU access (filed via M1 Issue 4; required to actually train)

## Initial Setup (From Scratch)

### 1. Clone and Branch

```bash
git clone git@github.com:newton-school-ai/slmforge.git
cd slmforge
git checkout dev
git checkout -b feature/issue-N-your-feature dev
```

### 2. Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:
- `HF_TOKEN` (get from huggingface.co/settings/tokens)
- `SLMFORGE_DB_URL=sqlite:///./slmforge.db` (default; swap to Postgres for shared deploys)
- `REDIS_URL=redis://localhost:6379/0`
- `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `JWT_SECRET_KEY` (only for shared-deploy mode; localhost mode has no auth)
- `SLMFORGE_GPU_HOST` (filled in once Issue 4 ticket is granted)
- `SLMFORGE_JUDGE_MODEL` (optional; only if you turn LLM-as-judge on in eval)

Generate a JWT secret:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 3. Start Services (Docker)

```bash
docker compose up -d redis
```

This starts Redis (job queue + WebSocket pub/sub). The API and frontend run locally during dev.

### 4. Python Environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # editable install + dev tools
pre-commit install                 # hooks for ASCII + internal-data guard
```

### 5. Database Setup

```bash
alembic upgrade head
```

(Seed data lands in M1 Issue 2 once schemas exist.)

### 6. Run Backend

```bash
uvicorn slmforge.api.main:app --reload --port 8000
curl http://localhost:8000/health
# {"status":"ok"}
```

### 7. Run Frontend (Web UI)

```bash
cd frontend
npm install
npm run dev
# Opens http://localhost:3000
```

### 8. Run the CLI Smoke

```bash
slmforge --help
slmforge --version
slmforge build --help
```

## Daily Workflow

### Starting Work

```bash
git checkout dev
git pull origin dev
git checkout -b feature/issue-N-name dev
```

### Running Tests

```bash
# Fast unit tests
pytest tests/unit -v

# Integration tests (shared GPU required)
pytest tests/integration -v -m gpu

# With coverage
pytest tests/ --cov=src/slmforge --cov-report=term-missing

# Lint
ruff check src/
black --check src/
```

### Submitting Work

```bash
git add -A
git commit -m "feat: short description"
git push origin feature/issue-N-name

# Open PR on GitHub targeting dev
gh pr create --base dev --title "Issue N - Title" --body "Closes #N"
```

### Keeping Branch Updated

```bash
git checkout dev
git pull origin dev
git checkout feature/issue-N-name
git merge dev
# Resolve conflicts if any
```

## Building Your First SLM

### Quick Sanity (Bundled Recipe)

```bash
slmforge build --recipe feedback_summariser --auto
# Watch progress in the terminal; takes < 20 min on shared GPU
```

When it finishes you will see a panel with the `build_id`, serve command, and curl example. Confirm with:

```bash
slmforge usage <build_id>
slmforge serve <build_id>
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<build_id>","messages":[{"role":"user","content":"Summarise: ..."}]}'
```

### From Your Own Data Folder

```bash
cd /path/to/your/data
slmforge build
# Walks you through task type + base model + LoRA defaults
```

### From the Web UI

```bash
slmforge ui
# Opens http://localhost:3000
# Pick file paths, hit + to add more, click Build
```

## Docker (Full Stack)

```bash
# Build and start everything (api + redis)
docker compose up -d

# View logs
docker compose logs -f api

# Stop
docker compose down

# Rebuild after changes
docker compose up -d --build api
```

## API Testing

```bash
# Health check
curl http://localhost:8000/health

# Create a build (skeleton; full payload lands in M3)
curl -X POST http://localhost:8000/builds \
  -H "Content-Type: application/json" \
  -d '{"sources":[],"task_type":"auto"}'

# List builds
curl http://localhost:8000/builds

# Stream a run (WebSocket)
wscat -c ws://localhost:8000/builds/<build_id>/stream
```

## Troubleshooting

### Pre-commit hook blocked a commit
- Read the message; it tells you which file violated ASCII or internal-data rules.
- Fix the file. To override the ASCII check for a specific known-good file, talk to the maintainer first.

### "src refspec main does not match any"
- You committed on a branch that is not `main`. Either rename the branch or `git branch main` to create main pointing at HEAD, then push.

### "Upgrade to GitHub Pro or make this repository public" on rulesets
- Free-plan private repos don't support new rulesets. Use classic branch protection via:
  ```bash
  gh api -X PUT repos/<org>/<repo>/branches/main/protection --input - <<JSON
  { ... see README of newssnap-ai for the JSON shape ... }
  JSON
  ```

### `gh project create` says missing OAuth scopes
```bash
gh auth refresh -s project,read:project
```

### Redis connection refused
```bash
docker compose up -d redis
```

### CUDA / MPS / CPU mismatch on a notebook
- vLLM and the LoRA harness pick the device automatically. If running outside SLMForge, use:
  ```python
  import torch

  if torch.cuda.is_available():
      device = torch.device("cuda")
  elif torch.backends.mps.is_available():
      device = torch.device("mps")
  else:
      device = torch.device("cpu")
  ```

### "Out of memory" during fine-tune
- 8B-class bases need QLoRA (4-bit) on shared GPU. Make sure `--base llama-3.1-8b-instruct` triggers QLoRA automatically (M4 Issue 17) -- if it doesn't, that's a bug to file.

### vLLM server boots slow on first run
- First boot downloads + caches the base model. Subsequent boots are fast.

### Frontend can't reach backend
- Check both ports are free (`lsof -i :3000`, `lsof -i :8000`). The Vite dev server proxies `/builds` to 8000; if you change ports, update `frontend/vite.config.ts`.

### Alembic "target database is not up to date"
```bash
alembic upgrade head
```

### "Hugging Face 401 unauthorized" pulling a gated base
- Run `huggingface-cli login`, accept the model's gated terms on the HF website, then retry.

## Project Context

Always re-read `_internal/PROJECT_CONTEXT.md` if you forget the scope, ship gate, milestone numbers, or data strategy. The PROJECT_CONTEXT is the single source of truth for project intent.
