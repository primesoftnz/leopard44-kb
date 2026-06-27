# Leopard 44 KB

An offline knowledge base for the Leopard 44 catamaran. Ask questions about your boat in
plain English and get cited answers — no internet required after the initial model download.

Built for owners of the Robertson & Caine Leopard 44 (and the Sunsail 444 charter variant).
Query technical specifications, known issues, sailing performance, and onboard systems from
the command line, web UI, or by voice — all running locally on your hardware.

---

## Prerequisites

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Python | 3.10+ | 3.11 or 3.12 recommended |
| [uv](https://docs.astral.sh/uv/) | any recent | Fast Python package manager (`curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| [Ollama](https://ollama.com) | latest | Local LLM runtime — runs the language model offline |
| RAM | 8 GB | Sufficient for the 3B-parameter tier (lower quality answers) |
| RAM | 16 GB | Recommended — runs the 7B-parameter tier (good quality) |
| RAM + GPU | 16 GB + VRAM | Fastest responses; GPU VRAM used automatically by Ollama |

The setup script detects your available RAM and selects the appropriate Ollama model tier
automatically.

---

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/primesoftnz/leopard44-kb.git
cd leopard44-kb

# 2. Run the setup script (installs dependencies, pulls Ollama models, loads demo seed)
./setup.sh

# 3. Launch the server
./start.command        # macOS / Linux double-click or terminal
```

On Windows, use `setup.bat` and `start.bat` instead.

**macOS note (Gatekeeper quarantine):** If `start.command` is blocked by macOS after downloading
via a browser, run once from the terminal to clear the quarantine flag:

```bash
chmod +x start.command
xattr -d com.apple.quarantine start.command
```

Then double-click normally in Finder. Files cloned via `git clone` do not have this restriction.

After setup, open [http://localhost:8000](http://localhost:8000) to use the web UI, or query
from the terminal:

```bash
uv run l44 ask "What is the LOA of a Leopard 44?"
uv run l44 ask "What are the known issues with the saloon windows?"
```

---

## Architecture

```
                      Query path (offline-absolute)
                      ─────────────────────────────
  User query
  CLI / Web UI / Voice
        │
        ▼
  Query engine
  ├── Vector KNN (sqlite-vec)   ◄──── shared/ layer  (factory specs, known issues,
  ├── FTS5 BM25                        sailing performance — ships with the package)
  └── RRF fusion
        │
        ▼
  Ollama (local LLM)            ◄──── vessel/ layer  (your maintenance logs,
        │                               WhatsApp exports, photos — private, never committed)
        ▼
  Cited answer
  with [1] shared: / [2] vessel: source references


                      Capture path (online, at dock)
                      ──────────────────────────────
  Photo  ──►  Vision model (local qwen2.5vl or cloud)  ──►  Owner confirms  ──►  vessel/
```

The two knowledge layers are stored separately:

- **shared/** — model-generic Leopard 44 facts (ships with this package, updated via `git pull`)
- **vessel/** — your private vessel data (stored in `~/.local/share/leopard44-kb/`, never committed)

---

## What ships vs. what you bring

### What ships with the package

The `shared/leopard44/` directory contains ten owner-authored reference documents covering the
Leopard 44 model class — written from public-domain research, not extracted from the copyrighted
factory manual:

| Document | Topic |
|----------|-------|
| [Technical specifications](shared/leopard44/technical-specifications.md) | Dimensions, displacement, sail area, engines, tanks |
| [Sailing performance](shared/leopard44/sailing-performance.md) | Speed, upwind ability, motion comfort |
| [Onboard systems](shared/leopard44/onboard-systems.md) | Electrical, plumbing, engines, electronics |
| [Interior layout](shared/leopard44/interior-layout.md) | Cabin configs, galley, cockpits, storage |
| [Known issues](shared/leopard44/known-issues.md) | Design flaws, common failures, maintenance patterns |
| [Upgrades and modifications](shared/leopard44/upgrades-modifications.md) | Safety and quality-of-life improvements |
| [Bluewater capability](shared/leopard44/bluewater-capability.md) | Ocean crossing suitability, passage prep |
| [Competitive comparison](shared/leopard44/competitive-comparison.md) | L44 vs. Lagoon, Catana, Nautitech, FP, Bali |
| [Market value](shared/leopard44/market-value.md) | Pricing, depreciation, charter vs. private impact |
| [Ex-charter buying guide](shared/leopard44/ex-charter-buying-guide.md) | Survey points, wear patterns, refit checklist |

### What you bring (first-run ingestion)

The vessel layer ships empty. Add your own documents after setup:

```bash
# Ingest your vessel's maintenance log (PDF or Markdown)
uv run l44 ingest path/to/maintenance-log.pdf --layer vessel

# Ingest a WhatsApp export (your private vessel chat — stays local)
uv run l44 ingest path/to/WhatsApp-chat.txt --layer vessel

# Ingest a photo with AI-assisted caption (requires internet at capture time)
uv run l44 capture photo path/to/photo.jpg

# Query after ingestion
uv run l44 ask "When did I last service the starboard engine?"
```

**WhatsApp from the L44 owners group:** If you have an export from the community owners group
(model-generic, anonymised), ingest it with `--layer community` instead:

```bash
uv run l44 ingest path/to/l44-owners-group.txt --layer community
```

Use `--layer vessel` (the default) for your personal vessel data. Use `--layer community`
only for the public owners-group export — this keeps your private records separate from
community-sourced information.

---

## Advanced: cloud LLM backend (not offline)

The default backend is Ollama (local, offline). For machines without enough RAM to run a
local model, you can route generation through [OpenRouter](https://openrouter.ai) using your
own API key:

```bash
export L44_LLM_BACKEND=openrouter
export OPENROUTER_API_KEY=sk-or-...
uv run l44 serve
```

This option requires internet access and sends your queries to a remote API — it is not
offline. The default Ollama path is recommended for privacy and offline use.

---

## Contributing

Contributions to the shared knowledge layer (model-generic Leopard 44 facts, Yanmar engine
docs, systems references) are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
fork-and-pull-request flow, anonymisation rules, and content guidelines.

---

## License

MIT — see [LICENSE](LICENSE).

Copyright (c) 2026 Greg Stevenson.
