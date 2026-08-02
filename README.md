# Dataset Generator

An **agentic dataset generator** that reads your documents, scrapes the web, pulls your
runbooks and Confluence pages, and turns all of it into a deduplicated, quality-gated,
citation-carrying dataset — then keeps itself up to date on a schedule.

> ### It runs on a LOCAL LLM
> Generation, quality judging and embeddings all go to **Ollama on your own machine**.
> Your runbooks, internal wiki pages and error logs never leave the box. There is no
> cloud provider in this codebase, and `config.py` warns you if `llm.base_url` is not a
> local or private address. Set `provider = "none"` and it still produces a dataset —
> just extractively instead of with a model.

---

## Quick start

```bash
cd dataset-generation

# 0. Optional but recommended parsers (it runs without them — see "Fallbacks")
pip install -r requirements.txt

# 1. Make sure the local model is up. If the model named in config.toml is not
#    pulled, datagen substitutes the closest installed one rather than failing —
#    but pulling the one you actually want gives better records.
ollama serve
ollama pull nomic-embed-text        # for semantic dedupe

# 2. Start the local search engine (optional; falls back to DuckDuckGo)
python searxng/setup.py

# 3. Check everything before you spend GPU time on it
python -m datagen doctor

# 4. Build from the sample corpus that ships with the repo
python -m datagen build

# 5. Look at what it produced
python -m datagen inspect -n 5
python -m datagen status

# 6. Teach it something of your own
python -m datagen learn -q "How do I free a GPU?" -a "Scale the endpoint to zero replicas."

# 7. Then hand it over: it now keeps itself current, on its own, forever
python -m datagen watch
```

Prefer clicking to typing? `python -m datagen serve` opens a local web UI that does all
of the above.

Output lands in `exports/` — `train.jsonl`, `eval.jsonl`, plus alpaca/sharegpt/chatml
variants and `rag_chunks.jsonl`.

---

## What it does

```
              ┌─────────── SOURCES ────────────┐
   PDFs ─┐    │  files      local documents    │
  DOCX  ─┤    │  runbooks   procedures         │        ┌── quality gate ──┐
  PPTX  ─┼───▶│  confluence your wiki          │───┐    │ heuristics       │
 images ─┤    │  web        crawl + robots.txt │   │    │ grounding check  │
   web  ─┤    │  keywords   search & scrape    │   │    │ local LLM judge  │
keyword ─┘    └────────────────────────────────┘   │    └──────────────────┘
                                                   ▼             │
                        documents ─▶ chunks ─▶ dedupe ─▶ generate ─▶ accepted
                                                (3 layers)  (local)      │
                                                        qa │ instruction │
                                             troubleshooting │ glossary  ▼
                                              exports/  +  server/pcai/learned.json
```

**Everything is incremental.** Content hashes live in SQLite, so a re-run skips documents
that have not changed and only regenerates what actually moved.

---

## Commands

| Command | What it does |
|---|---|
| `serve` | **Web UI** — upload documents, manage keywords, run builds, browse records |
| `doctor` | Checks the local model, which parsers are active, whether each source resolves |
| `build` | One pass over every configured source. `--full` ignores caches, `--only files runbooks` narrows it, `--limit 20` caps chunks for a quick test |
| `agent` | The agent plans its own sequence of tool calls toward the objective in `config.toml` |
| `scrape "<keyword>"` | Search the web for one keyword and ingest the best results |
| `ingest <path>` | Ingest a file or folder. `--runbook` parses procedures structurally |
| `learn` | Teach it from **your own input** — text, files, URLs, or a Q&A pair you wrote |
| `confluence` | Pull the configured spaces (`--space OPS PCAI --attachments`) |
| `watch` | Run forever, updating itself. `--once` for a single cycle |
| `export` | Re-export from `data/` without regenerating anything |
| `analyze` | **Is this trainable?** Token stats, truncation, leakage, dataset card |
| `status` | Totals, run history, and the leads queued for next time |
| `inspect` | Print sample records. `--rejected` shows what the quality gate threw out |

---

## Web UI — `datagen serve`

```bash
python -m datagen serve          # opens http://127.0.0.1:8800
```

Ten tabs covering everything you'd otherwise do on the command line:

| Tab | |
|---|---|
| **Overview** | Record/chunk/document counts, corpus contents, recent runs |
| **Upload** | Drag-and-drop documents into `corpus/docs` or `corpus/runbooks` |
| **Keywords** | Edit the keyword list (written straight into `config.toml`), try one before committing it, and set the dataset subject |
| **Run** | Build / full rebuild / agent / re-export, with a live log |
| **Browse** | Read the actual records, filter by kind, search, or view what the quality gate rejected |
| **Teach** | Add a Q&A pair, a solved case, or paste raw notes |
| **Analyze** | Token stats, truncation and leakage checks |
| **Exports** | List and download every generated file |
| **Confluence** | Credentials, space picker, page limits, connection test |
| **Settings** | Model picker, generation and quality settings — configure once, not every run |

### Settings

Styled to match the Kalam configuration page — HPE green `#01A982`, the same design
tokens, and the same model-picker pattern: an **Installed models** header with **Rescan**,
a dropdown listing what is actually installed with parameter counts and sizes, quick-pick
chips underneath, and **+ Enter manually** for a model you have not pulled yet. Embedding
models are detected and listed separately under *(vector RAG)* — picking a chat model as
your embedder is an easy mistake that quietly degrades dedupe.

Also covers pairs per chunk, workers, chunk size, and the quality threshold. Record kinds
are set in `config.toml` (`generation.kinds`). Everything is written back to `config.toml` **preserving your comments** — the
file is parsed before saving and the write is refused if the result would not be valid
TOML.

A non-local `base_url` is rejected outright. This project is local-only by design, and
silently shipping your corpus to a cloud endpoint because of a typo is not a failure mode
worth allowing.

### Confluence

Credentials go to `.env` (git-ignored); everything else to `config.toml`. **Test
connection** verifies auth before you commit to a run, and once connected the page lists
the spaces your token can actually see — click to add them rather than guessing space
keys. A stored token is only ever sent back to the browser masked, and leaving the field
blank keeps the existing one.

It's the same stack as everything else — **stdlib only**, no Flask, no CDN, no
build step. The page is one self-contained HTML file.

**It is bound to `127.0.0.1` deliberately.** This UI writes files into your corpus and
starts jobs, so it is not something to expose. There is no authentication, because there
is nothing to authenticate to on a loopback interface. If you pass `--host 0.0.0.0` you
are handing anyone on your network the ability to write files and run jobs on your
machine — don't, unless you have put your own authentication in front of it.

Within those bounds it is defensive about input: uploads are extension-allowlisted,
size-capped at 80 MB, and path-checked against the corpus directory (a `../../../` in a
filename is stripped, not honoured); downloads cannot escape the exports directory; and
no endpoint builds a shell command, so there is nothing to inject into. Those paths are
covered by tests.

One job runs at a time — the local model is a single resource, and two concurrent builds
would fight over it and over the SQLite state.

---

## Teaching it yourself — `datagen learn`

Everything else goes out and finds material. `learn` takes material **you** hand it and
folds it into the same dataset, knowledge base and exports.

```bash
# raw material — chunked, generated and quality-gated like any other source
datagen learn "MLIS endpoints hold their GPU allocation until scaled to zero."
datagen learn --file incident-2026-07.md --kind runbook
datagen learn --url https://docs.example.com/some-page
kubectl logs mlis-pod-x | datagen learn --title "MLIS crash log" --kind log
datagen learn                       # interactive paste, Ctrl-Z (Win) / Ctrl-D

# things YOU know — stored verbatim, never paraphrased
datagen learn -q "How do I free a GPU?" -a "Scale the endpoint to zero replicas."
datagen learn --problem "endpoint 503 after upgrade" \
              --resolution "Bucket credentials expired. Delete the secret and reconnect."
```

The two halves behave differently on purpose:

| Input | Path |
|---|---|
| text / file / URL / stdin | Normal pipeline: chunk → dedupe → local model generates → quality gate |
| `--question` + `--answer`, `--problem` + `--resolution` | **Bypasses generation and the quality gate.** Stored exactly as written, `score = 1.0`, `generator = human:input` |

That bypass is deliberate. The quality gate exists to catch a 7B model inventing things;
running it against a human correction would be backwards — its grounding check would
reject a *true* answer purely because you didn't paste the source you got it from.

Human rows are traceable forever: they carry a `human://` URL and a `human-authored` tag,
so you can always separate what a person asserted from what a model synthesised:

```bash
python -c "import json;[print(r['kind'],r['source_url']) for r in map(json.loads,open('data/records.jsonl',encoding='utf-8')) if r['generator'].startswith('human')]"
```

Re-learning the same pair is idempotent — record IDs are content-derived, so it updates
rather than duplicating.

---

## Search: local SearXNG

The keyword/discovery path defaults to a **local SearXNG** container. It aggregates
Google, Bing, Brave, DuckDuckGo and StackOverflow in one query, isn't scraping anyone's
HTML, and won't rate-limit you.

```bash
python searxng/setup.py            # start + verify (idempotent)
python searxng/setup.py --check    # verify only
python searxng/setup.py --logs
python searxng/setup.py --stop
```

The script generates the instance secret, brings up the container, waits for health, and
then **proves the JSON API answers**. That last step matters: SearXNG ships with
`search.formats` set to `html` only, so the JSON API returns a bare `403` with no
explanation. `searxng/settings.yml` enables `json` and disables the bot limiter (the
instance is bound to `127.0.0.1` only). On this machine it returned 40 results across
four engines where DuckDuckGo alone returned 6.

**If the container is down, the run does not fail** — `web_search()` falls back to
DuckDuckGo and logs how to start SearXNG. `datagen doctor` shows its status either way.

---

## The agentic part

`python -m datagen agent` runs an **observe → plan → act** loop. The local model is given
the objective, the tool list and a scratchpad of what has happened so far, and picks one
tool per step:

| Tool | |
|---|---|
| `search_web` | See what exists for a query without downloading it |
| `scrape_keyword` | Search *and* download the best-ranked pages |
| `scrape_url` / `crawl_site` | One page, or breadth-first across a docs site |
| `ingest_files` / `ingest_runbooks` | Local documents and procedures |
| `search_confluence` | CQL or free-text against your wiki, attachments included |
| `build_dataset` | Chunk → dedupe → generate → quality gate |
| `assess_coverage` | What is covered, what is thin |
| `propose_sources` | Save leads for the next run |
| `export_dataset` / `finish` | Land the work and stop |

Guard rails, because a 7B planner will absolutely try all of these: a step budget, a
consecutive-error budget, a repeat detector that blocks identical calls, and a
**heuristic planner** that takes over if the model returns garbage three times running.
The heuristic planner drives the same tools in a fixed sensible order, so
`agent` still works with no model at all.

### How it updates itself

`[schedule] enabled = true` in the shipped config, so this is the intended way to run it:
start it once and leave it.

```bash
python -m datagen watch                     # uses schedule.interval_min (6h)
python -m datagen watch --interval 360      # or override it
python -m datagen watch --once --no-agent   # one plain pipeline cycle
```

1. `watch` wakes on `schedule.interval_min` and re-runs the agent.
2. Every source is re-checked: local docs, runbooks, **Confluence**, the crawl and the
   keyword searches. Unchanged documents are skipped by content hash; changed ones have
   their stale chunks and records dropped and regenerated.
3. After each run the agent asks the local model *what is missing*, and writes concrete
   keywords/URLs into the `discovered` table.
4. The next run's web and keyword connectors pick those up automatically.
5. Every `full_rebuild_every` cycles it rebuilds from scratch to clear any drift.
6. Each cycle re-exports and rewrites `server/pcai/learned.json`, so the Kalam assistant
   keeps improving without anyone running a command.

**If Ollama is down, the cycle is skipped, not degraded.** An unattended loop that keeps
going without a model would quietly fill the dataset with sliced-up source text that you
would then have to find and delete by hand. Skipping costs one interval and loses nothing,
because documents only count as processed once their records exist.

On Windows, run it under Task Scheduler or NSSM. Interrupting it is always safe — state
is committed to SQLite after every step.

---

## Sources

Configure in `config.toml`. Each block has a `name` and optional `tags` that follow the
data all the way into the exported records.

**`[[sources.files]]`** — PDF, DOCX, PPTX, XLSX, MD, TXT, HTML, CSV, JSON, code, **and
images**. Drop files into `corpus/docs/`.

### Screenshots and diagrams

An image cannot be parsed, only *described*, so `[llm.vision]` points at a local vision
model and its description becomes the document's text — chunked, generated from and
quality-gated exactly like a PDF's.

```toml
[llm.vision]
enabled = true
model   = "moondream"     # 1.7 GB, fast, good at reading screenshots
                          # llava is bigger and better on diagrams
```

```bash
ollama pull moondream
python -m datagen ingest ./mlis-503-error.png
```

The prompt is written for this material specifically: transcribe every error message, log
line, command and label *exactly as written*; for a diagram, name the components and say
how they connect; don't describe the window chrome or the cursor. Asking a vision model to
"describe this image" gets you "a screenshot of a computer screen", which is worth nothing
in a training set.

This applies everywhere `parse_bytes` is used, not just local files — images the crawler
finds and **Confluence attachments** go through the same path, which matters because
wiki pages document failures with screenshots more often than with text.

The filename is kept as a heading (`Image: mlis-503-error.png`), since it is often the only
thing tying a screenshot to its subject.

### Read this before you trust image records

**A vision model does not transcribe, it paraphrases — and an image description is the
only source there is.** Every other record in this dataset can be checked against its
source text by the grounding score. An image record cannot: the description *is* the
source, so if the model misread it, the record is confidently wrong and nothing downstream
will catch it.

Measured on a clean 520×140 PNG of black text on white, with `moondream`:

| In the image | `moondream` read |
|---|---|
| `ERROR: MLIS endpoint returned` | `ERR MIB END POINT RETURNS` |
| `503 Service Unavailable` | `5003 SERVICE UNABILIOUS` |
| `pod mlis-7f9c in CrashLoopBackOff` | `pod mics 776 in Crash Loopback.off` |

It also volunteered "insufficient disk space" and "a hardware or software failure", neither
of which is in the image. Train on that and you teach the assistant the error code `5003`.

So:

- **`moondream` (1B) is for triage, not for training data.** It gets the gist and mangles
  every identifier. `ollama pull llava` for anything you intend to fine-tune on; when the
  configured model is missing, substitution prefers the capable models and picks moondream
  last, deliberately.
- **Image-derived text is labelled.** Every one starts `Image: <filename> (described by
  <model>)`, so you can find them: `python -m datagen inspect --kind image`, or filter
  `data/chunks.jsonl` on `"kind": "image"`.
- **For a screenshot that really matters, type it yourself.** `datagen learn --problem/
  --resolution` stores your words verbatim, bypasses generation, and scores 1.0. Thirty
  seconds of typing beats a paraphrase you have to audit.

Other limits: images over `max_bytes` (12 MB) are skipped; long prompts make small vision
models return *nothing*, so the prompt is deliberately two short sentences with a
one-sentence retry behind it; with no vision model installed, images are skipped with one
log line and everything else runs normally.

**`[[sources.runbooks]]`** — parsed *structurally*, not as flat text. The connector
recognises symptom / cause / preconditions / steps / verification / rollback / escalation
sections, extracts ordered steps and referenced commands, and re-emits a normalised
document. That is why generated troubleshooting records keep the step order instead of
producing "run the drain command" with no context. See
`corpus/runbooks/mlis-endpoint-unavailable.md` for the shape it reads best.

**`[[sources.confluence]]`** — Cloud and Server/DC. Storage-format XHTML is cleaned with
macros *unwrapped* rather than dropped, so code blocks, info/warning panels and status
labels survive. PDF/DOCX attachments are pulled through the same parsers.

```bash
# .env
CONFLUENCE_BASE_URL=https://your-org.atlassian.net/wiki
CONFLUENCE_USER=you@example.com     # leave empty for a Server/DC PAT
CONFLUENCE_TOKEN=...
```

**Space keys are verified before searching.** CQL naming a space that does not exist
returns an empty result set with a `200` — indistinguishable from "the wiki has nothing"
unless you look, and the reason a connected wiki can report `fetched: 0` run after run. So
the connector checks your `spaces` against the wiki, logs the keys your token *can* read,
and falls back to `text_search` across every readable space:

```toml
spaces      = ["PCAI", "OPS"]        # checked at run time
text_search = ["Private Cloud AI", "MLIS", "runbook", "troubleshooting"]
cql         = ""                     # raw CQL overrides both
```

That fallback is what makes it useful on a wiki you have access to but have not
catalogued. Set `spaces` properly once you know the keys — it is cheaper and more precise.

**`[[sources.web]]`** — breadth-first crawl with a domain allow-list, deny patterns,
robots.txt, per-host delay, and hard caps on pages/depth/size. Linked PDFs are parsed,
not skipped.

**`[sources.keywords]`** — the "find everything about X" path. Queries your local SearXNG
(falling back to DuckDuckGo's no-JS endpoint), ranks results — vendor docs and `docs.*`
hosts up, Pinterest/Quora and tag-archive pages out — then scrapes the winners.

**`datagen learn`** — anything you hand it directly. See above.

---

## Quality

Duplicated chunks become duplicated training rows, and a confident wrong answer is worse
than no answer. So:

**Dedupe, three layers.** Exact (sha256 of normalised text) → near (64-bit SimHash with a
banded index, so it is not O(n²)) → semantic (cosine over local embeddings). Layer 2 is
what kills the boilerplate headers and version-bumped duplicate pages a crawl produces in
bulk. Dedupe runs against *history*, not just the current batch.

**Quality gate, cheap checks first.** Length bounds, questions that reference "the text
above", non-answers, page chrome, unbalanced code fences, garbled extraction → then a
grounding score → then the local model scoring faithfulness, usefulness and
self-containment.

**Grounding scores identifiers strictly and prose leniently.** The obvious metric — how
many of the answer's tokens appear in the source — is actively harmful: it scores a
verbatim copy 1.0 and a well-written explanation 0.2, because words like *cause*,
*diagnose* and *verify* are what a good answer adds and a source rarely repeats. Training
on the copies and discarding the explanations is the opposite of the goal. What actually
signals an off-source answer is an **identifier** the model invented — a command, a flag,
a path, an error code — so those are checked strictly while prose is not. A runbook answer
that explains `503 → CrashLoopBackOff` in its own words passes; a glossary entry defining
TLS from the model's own knowledge still fails.

Short answers are kept. `"Which auth broker does PCAI use?" → "Keycloak"` is a perfect
record, and a high `quality.min_answer_chars` (default 12) silently deletes exactly the
crisp factual lookups an ops assistant is asked for.

Rejected records are **quarantined, not deleted** — they go to `data/quarantine.jsonl`
with a reason so you can tune `quality.min_score` rather than guess:

```bash
python -m datagen inspect --rejected -n 10
```

**Train/eval split is by source document**, never by row. Rows from one chunk are
near-duplicates; letting them straddle the split gives you an eval number that means
nothing.

---

## Is it trainable? — `datagen analyze`

Counts don't tell you whether a dataset will fine-tune well. Run this **before** booking
GPU time:

```bash
python -m datagen analyze --max-seq-len 2048
python -m datagen analyze --json --strict     # for CI: non-zero exit on any warning
```

```
  TOKEN LENGTHS (estimated — verify with your trainer's tokenizer)
  instruction            min       3   p50      14   p90      19   p95      22   max      33
  output                 min      14   p50      34   p90      78   p95      91   max     100
  instruction+output     min      22   p50      49   p90     100   p95     105   max     123

  at max_seq_len=2048: OK (0.0%)

  INTEGRITY
    duplicate questions      0 exact, 0 near
    train/eval leakage       0
```

It checks the things that silently ruin a run, roughly in order of how often they do:

1. **Truncation** — rows over `max_seq_len` get cut mid-answer, teaching the model to stop
   early. It reports the count, the worst offenders, and the limit that would cover p99.
2. **Leakage** — the same question in train *and* eval makes your eval loss a lie.
   Document-level splitting mostly prevents it, but two sources describing the same thing
   still collide.
3. **Surviving duplicates** — over-weighted rows.
4. **Imbalance** — one kind or source dominating teaches format, not knowledge.
5. **Volume** — it will tell you plainly when there are too few rows to be worth training
   on, and suggest using the data for RAG instead.
6. **Degenerate rows** — empty, one-line, or truncated-looking answers.

Token counts are **estimates** (character- and word-based heuristics, no tokenizer
dependency) and deliberately err high, so a truncation warning is worth checking rather
than worth ignoring. Confirm against your trainer's real tokenizer before fixing
`max_seq_len`.

### DATASET_CARD.md

Every export writes a dataset card next to the data — provenance, how it was built, which
local model wrote it, composition tables, length percentiles, split methodology, and an
honest limitations section (synthetic content, **unverified source licensing**, inherited
source bias, what was and wasn't human-reviewed). Keep it beside any checkpoint you train
so that in six months you can still answer "where did this data come from?".

---

## Output formats

| File | Use |
|---|---|
| `dataset.jsonl` | Everything: scores, reasons, provenance, source context |
| `train.jsonl` / `eval.jsonl` | The split, without source context |
| `train.alpaca.jsonl` | `{instruction, input, output}` — classic SFT |
| `train.sharegpt.jsonl` | axolotl / LLaMA-Factory |
| `train.chatml.jsonl` | OpenAI-style `{messages:[...]}` |
| `rag_chunks.jsonl` | The chunks themselves, for retrieval instead of fine-tuning |
| `GLOSSARY.md` + `glossary.jsonl` | A readable A–Z glossary of every term your corpus defines |
| `manifest.json` | Counts by kind and source, mean score, which model generated it |

### The glossary

`glossary` is a generation kind *and* an export format. The generator asks the local
model which terms each chunk actually **defines** — products, components, acronyms,
resource types, config keys — and refuses to define anything from its own knowledge.

Unlike the other formats it is not one row per record. The same term gets defined in a
dozen chunks, so entries are **merged corpus-wide**: the highest-scored definition wins,
aliases (acronym ↔ expansion) are unioned, and *every* source that discussed the term is
cited under the entry.

```markdown
### MLIS

MLIS is a service that hosts inference endpoints on the GPU worker pool.

*Also known as: Machine Learning Inference Service*

<sub>Sources: [PCAI Architecture Notes](…), [MLIS endpoint runbook](…)</sub>
```

With no model running, an extractive fallback still finds definitions by pattern —
`MLIS (Machine Learning Inference Service)` acronym expansions (validated against the
initials, so `The GPU (which we installed last year)` is correctly rejected), plus
`X is a …` and `X — …` sentences.

Turn it off by dropping `"glossary"` from `generation.kinds` and `export.formats`.

### Feeding the Kalam PCAI assistant

`export.kalam_kb_path` points at `../server/pcai/learned.json`, the `LearnedDoc[]` the
existing PCAI chatbot already reads. Every export writes chunks and troubleshooting cases
there, tagged `datagen://<dataset>/`, replacing only its own previous entries and leaving
anything you added through the Kalam UI alone. Hit **Train** in the PCAI panel (or
`kalam train`) and the chatbot picks it up.

---

## Fallbacks — why this does not break

Every dependency is optional and probed at runtime:

| Missing | What happens |
|---|---|
| **`llm.model` is not pulled** | Resolved automatically: first by tag (`qwen2.5:7b-instruct` matches an installed `…-q4_K_M`), then to the closest installed chat model of the same family. Embedding and vision models are ranked last so they can never be picked to write your dataset. One warning line names the substitute and the `ollama pull` that would avoid it |
| **`embeddings.model` is not pulled** | Substituted with any installed embedder, or embeddings are switched off for the run — semantic dedupe goes with them, exact and SimHash stay |
| **`vision.model` is not pulled** | Substituted with any installed vision model, else image support turns off and images are skipped with one line. Everything else runs |
| **The local model is down** | A one-shot `build` falls back to extractive generation: headings become questions, ordered steps become procedures, error lines become troubleshooting prompts. Grounded by construction — the answer *is* the source text. Set `allow_extractive_fallback = false` to hard-fail instead. **`watch` does not do this** — it skips the cycle (see below) |
| A call fails fatally (404 / 401 / model not found) | The LLM is disabled for the rest of the run after **one** error line, instead of retrying per chunk. A missing model used to cost 20 minutes of identical 404s |
| `pypdf` | A built-in extractor inflates FlateDecode streams and reads the text operators. Fine for digital PDFs, no OCR |
| `python-docx` / `python-pptx` / `openpyxl` | Built-in OOXML unzip-and-strip |
| `trafilatura` / `bs4` | Built-in `HTMLParser` stripper that drops script/style/nav |
| Embeddings unavailable | Semantic dedupe off, exact + SimHash still run |
| Search engine blocked | The keyword is saved as a lead and retried next run |
| A source raises | Logged, that source is skipped, the run continues |
| A host does not resolve | Remembered for the run, so the other 40 links pointing at it are skipped instead of costing ~20s each in retries |
| A URL contains non-ASCII | Percent-encoded (IDNA for the host). A single link with an em dash in its slug used to abort the whole crawl with `UnicodeEncodeError` |
| **The run is interrupted mid-generation** | Nothing is lost. Documents and chunks are marked processed *only* after their records are on disk, so the next run redoes exactly the work that did not finish |

`python -m datagen doctor` tells you which path each one is on.

---

## Layout

```
dataset-generation/
├── config.toml              all configuration
├── .env.example             credentials (never in config.toml)
├── corpus/                  your documents go here
├── data/                    state.db, records.jsonl, quarantine.jsonl
├── exports/                 the dataset
├── searxng/                 local search: compose file, settings, setup script
├── tests/test_datagen.py    84 tests, no network, no LLM
└── datagen/
    ├── __main__.py          CLI
    ├── learn.py             input you provide -> dataset records
    ├── analyze.py           trainability report + dataset card
    ├── web.py + webui/       local control panel (stdlib http.server)
    ├── config.py            TOML + env, local-address guard
    ├── llm.py               Ollama / OpenAI-compatible client
    ├── state.py             SQLite: hashes, records, leads, run history
    ├── chunking.py          heading-aware chunking
    ├── dedupe.py            simhash + banded index + semantic
    ├── generators.py        prompts + extractive fallbacks
    ├── quality.py           heuristics, grounding, LLM judge
    ├── exporters.py         all output formats + Kalam bridge
    ├── pipeline.py          the build
    ├── connectors/          parsers, files, runbooks, web, search, confluence
    └── agent/               loop, tools, scheduler
```

Tests: `python tests/test_datagen.py` (or `python -m pytest tests -q`).

---

## Tuning

**Better output**: raise `generation.pairs_per_chunk`, or switch `llm.model` to a bigger
local model — `gemma4:latest` (9.6 GB) is noticeably better than the 7B default here at
the cost of speed. Check what you have with `doctor`.

**Faster**: lower `generation.max_workers` if the GPU is thrashing — a local model is one
process on one GPU, and more than ~4 concurrent requests usually makes it *slower*, not
faster. Use `build --limit 20` to sanity-check prompt changes before a full run.

**Too much rejected**: `inspect --rejected` first — the reason is on every row. If the
reasons are `ungrounded`, the model is inventing identifiers that are not in the source;
use a bigger model or lower `chunking.max_chars`. If they are heuristic rejections, the
source text is probably navigation chrome and the fix is a tighter `deny_patterns`. If the
judge is rejecting answers you think are fine, lower `quality.min_score` — a 7B judge is
noisy, and nothing is deleted either way.

**Too few records**: this is usually sources, not settings. Three keywords and one seed URL
produce a few dozen rows; a supervised fine-tune realistically wants 500+, and below ~100
you are more likely to damage the base model than teach it. Add keyword terms, add crawl
seeds, drop more files into `corpus/docs/`, and connect Confluence. `analyze` will tell you
plainly when the volume is still too low and the data is better used for RAG.
