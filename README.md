# compress-o-bot
<img width="512" height="512" alt="ChatGPT Image Sep 20, 2025, 09_16_19 AM" src="https://github.com/user-attachments/assets/dfaa6900-31fc-4d10-b755-ef4726e99305" />

ChatGPT/OpenAI conversation archive compressor.

`compress-o-bot` turns an OpenAI export `conversations.json` into a compact, searchable “memory pack”:
- Split one giant archive into per-thread JSON files
- Chunk each thread into summarizer-ready slices
- Summarize each chunk (semantic + sentiment) and build indices
- Roll up chunk summaries into per-thread summaries
- Pack thread summaries into markdown shard files + a JSON index

### Requirements
- Go (recent version recommended)
- For AI stages: `OPENAI_API_KEY` set in your environment

### Quick start
- **Option A (recommended)**: run the full pipeline:

```bash
export OPENAI_API_KEY="..."
go run ./cmd/archive-pipeline \
  -conversations docs/peanut-gallery/conversations.json \
  -base-dir docs/peanut-gallery
```

- **Option B**: run stages individually:

```bash
# 1) split conversations.json into per-thread JSON files
go run ./cmd/archive-splitter -in docs/peanut-gallery/conversations.json -out docs/peanut-gallery/threads

# 2) chunk threads (uses OpenAI to choose turn breakpoints)
go run ./cmd/thread-chunker -in docs/peanut-gallery/threads -out docs/peanut-gallery/threads/chunks -model gpt-5-mini

# 3) summarize chunks (semantic + sentiment) and build indices
go run ./cmd/chunk-summarizer -in docs/peanut-gallery/threads/chunks -out docs/peanut-gallery/threads/summaries -model gpt-5-mini

# 4) roll up chunk summaries into per-thread summaries
go run ./cmd/thread-rollup -in docs/peanut-gallery/threads/summaries -out docs/peanut-gallery/threads/thread_summaries -model gpt-5-mini

# 5) pack thread summaries into markdown “memory shards”
go run ./cmd/memory-pack -mode semantic   -in docs/peanut-gallery/threads/thread_summaries           -out docs/peanut-gallery/threads/memory_shards
go run ./cmd/memory-pack -mode sentiment  -in docs/peanut-gallery/threads/thread_sentiment_summaries -out docs/peanut-gallery/threads/memory_shards_sentiment
```

### Flags (what they do / when to use)
- **`cmd/archive-pipeline`** (orchestration)
  - `-conversations`: input `conversations.json` export.
  - `-base-dir`: output root; writes into `<base-dir>/threads/...`.
  - `-model`: default model used for chunking + semantic summary + semantic rollup.
  - `-sentiment-model`: override model used for *sentiment* passes (chunk sentiment + thread sentiment rollup).
  - `-sentiment-prompt-file`: path to a file containing a custom *sentiment prompt header*; the tool appends a required `SECURITY:`/schema tail.
  - `-from-stage` / `-only-stage`: resume at a stage or run just one stage (`split|chunk|summarize|rollup|pack`).
  - `-overwrite`: clobber existing outputs (disables resumability); otherwise stages try to skip work when outputs exist.
  - `-pretty`: human-readable JSON for outputs that support it.
  - `-concurrency`, `-batch-size`: throughput tuning for OpenAI calls in summarization/rollup.
  - `-max-chunks`: cap work for smoke tests.

- **`cmd/archive-splitter`** (export → per-thread JSON)
  - `-in`, `-out`: input export and output directory.
  - `-array-field`: if the top-level JSON is an object, name of the field containing the conversations array.
  - `-pretty`, `-overwrite`: formatting and overwrite behavior.

- **`cmd/thread-chunker`** (threads → chunks; uses OpenAI)
  - `-in`: a thread file OR a directory of thread files.
  - `-out`: output chunk directory (per-thread subdirs are created).
  - `-model`: model used for breakpoint detection.
  - `-target-turns`: desired turns per chunk.
  - `-api-key`: optional override for `OPENAI_API_KEY`.

- **`cmd/chunk-summarizer`** (chunks → per-chunk summaries + index + glossary; uses OpenAI)
  - `-in`, `-out`: chunks input and summaries output.
  - `-model`: semantic summary model.
  - `-sentiment-model`: sentiment summary model override (common to run heavier here).
  - `-sentiment-prompt-file`: custom sentiment prompt header file.
  - `-resume`: skip chunks that already have both semantic+sentiment outputs.
  - `-reindex`: rebuild `index.json`/`sentiment_index.json` from outputs at the end.
  - `-glossary`, `-glossary-max-terms`, `-glossary-min-count`: glossary persistence and prompt sizing.

- **`cmd/thread-rollup`** (chunk summaries → per-thread summaries; uses OpenAI)
  - `-in`: summaries directory (expects `*.summary.json` + `glossary.json`).
  - `-out`: semantic thread summaries output.
  - `-sentiment-out`: sentiment thread summaries output (empty disables sentiment rollup).
  - `-model` / `-sentiment-model`: semantic vs sentiment rollup models.
  - `-resume`, `-reindex`, `-max-chunks-per-thread`: control reruns and splitting large threads into parts.

- **`cmd/memory-pack`** (thread summaries → markdown shards + JSON index)
  - `-mode`: `semantic` or `sentiment`.
  - `-in`, `-out`: input thread summary dir and output shard dir.
  - `-max-bytes`: target shard size (UTF-8 bytes).
  - `-index*` flags: control index truncation/size for downstream retrieval.

### Outputs (default paths)
- `docs/peanut-gallery/threads/`: split threads + derived artifacts
  - `chunks/`: chunk JSON files
  - `summaries/`: per-chunk semantic + sentiment summaries + indices
  - `thread_summaries/` and `thread_sentiment_summaries/`: per-thread rollups
  - `memory_shards/` and `memory_shards_sentiment/`: markdown shard files + `*_memory_index.json`

### Notes
- The AI stages are designed to be resumable; see each command’s flags (`-resume`, `-overwrite`, etc.).
- For best results, run commands from the repo root so relative `./cmd/...` paths resolve.

<img width="256" height="256" alt="ChatGPT Image Sep 20, 2025, 09_38_01 AM" src="https://github.com/user-attachments/assets/6aa839be-523f-4f9b-a1d6-7b4dfe6a1214" />
