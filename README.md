# LLM Hallucination Detector — Verification Lab

Checks every claim in an AI-written response against a source document, scores faithfulness, and rewrites the response with verified facts. FastAPI backend, hand-built HTML/CSS/JS dashboard.

## How it works 

```
response ──► claim extraction ──► verification ──► score ──► correction ──► re-verification
                (LLM)         NLI model + LLM        │          (LLM)        (loops until clean)
                              evidence-cited         └─► internal-conflict check
```

| Stage | File | What it does |
|---|---|---|
| Claim extraction | `claim_extraction.py` | LLM splits the response into standalone claims (pronouns resolved) |
| Evidence verification | `entailment_check.py` | LLM must cite the exact source sentences behind every verdict; verdicts with no valid citation are downgraded |
| NLI verification | `nli_verifier.py` | Optional DeBERTa NLI model checks each claim against the most relevant source sentences |
| Ensemble + cascade | `ensemble.py` | Combines both verifiers; disagreement becomes `needs_review`; cascade mode only sends doubtful claims to the LLM |
| Calibration | `calibration.py` | Maps raw confidence to real probabilities (isotonic regression) |
| Scoring | `scoring.py` | Supported claims divided by all claims; also reports resolved faithfulness and verification coverage |
| Internal consistency | `consistency.py` | Finds claims that contradict each other, independent of the source |
| Correction loop | `correction.py`, `pipeline.py` | Fixes contradicted claims, re-extracts and re-verifies the result, repeats until clean or the pass limit |
| Server | `server.py` | `/api/check`, `/api/config`, `/api/health`, and the static dashboard |

Verdict labels: `supported`, `contradicted`, `unsupported` (source is silent), `needs_review` (verifiers disagree, or calibrated confidence is low).

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env        # edit .env and add GROQ_API_KEY
uvicorn server:app --reload
```

Open http://127.0.0.1:8000.

Groq is the default provider. Configure it with:

```bash
set LLM_PROVIDER=groq
set GROQ_API_KEY=your_key
```

### Enable the NLI verifier (cascade / full modes)

```bash
pip install -r requirements-nli.txt
export ENABLE_NLI=true
```

The first request downloads the NLI model from Hugging Face, so allow time and memory for it. Without NLI the app runs in LLM-only mode and the UI says so.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` or `ollama` |
| `GROQ_API_KEY`, `GROQ_MODEL` | – / `openai/gpt-oss-20b` | Groq settings |
| `OLLAMA_MODEL` | `llama3` | Ollama model |
| `ENABLE_NLI` | `false` | Turn on the NLI verifier |
| `NLI_MODEL` | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | Hugging Face NLI model |
| `VERIFIER_MODE` | `cascade` | `cascade`, `full` or `llm` (falls back to `llm` without NLI) |
| `CASCADE_THRESHOLD` | `0.9` | NLI confidence needed to skip the LLM |
| `MAX_ITERATIONS` | `3` | Max correct-and-reverify passes |
| `ENABLE_CONSISTENCY` | `true` | Internal-conflict check (one extra LLM call without NLI) |
| `CONFLICT_THRESHOLD` | `0.8` | NLI contradiction score that flags a conflict |
| `CALIBRATOR_PATH` | `calibrator.json` | Fitted calibrator, used automatically if present |
| `REVIEW_BELOW` | `0.5` | Calibrated confidence below this becomes `needs_review` |
| `MAX_SOURCE_CHARS`, `MAX_RESPONSE_CHARS`, `MAX_UPLOAD_BYTES` | 60000 / 12000 / 5000000 | Input limits |
| `CORS_ORIGINS` | empty | Comma-separated origins allowed cross-origin (none by default) |
| `RATE_LIMIT_PER_MINUTE` | `30` | Per-client limit on `/api/check` (0 disables) |
| `LLM_CACHE_DIR` | empty | Cache LLM replies on disk (used by the eval suite) |
| `RAG_ENABLE_SEMANTIC` | `true` | Try dense semantic retrieval when optional dependencies are installed |
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence Transformer embedding model |
| `RAG_TOP_K` | `4` | Maximum retrieved chunks per claim |
| `RAG_CONTEXT_CHARS` | `6000` | Maximum assembled evidence context |
| `API_AUTH_TOKEN` | empty | Optional bearer token required by `/api/check` |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a scripted fake LLM and a fake NLI predictor, so they run offline.

## Calibration

You need a CSV of labelled claims with columns `source_text,claim,gold_label` (`gold_label` is `supported`, `contradicted` or `unsupported`). Use several hundred claims or more.

```bash
python -m scripts.fit_calibration labels.csv --out calibrator.json
python -m scripts.reliability_report held_out.csv --calibrator calibrator.json --out reliability.png
```

`fit_calibration` reports ECE (expected calibration error) on a held-out split, then saves a calibrator fitted on everything. Run `reliability_report` on claims that were not used for fitting.

## Evaluation

```bash
python -m eval.run_eval --jsonl data.jsonl --limit 100 --configs a,b,c,d,e
python -m eval.run_eval --limit 100            # HaluEval summarization via Hugging Face datasets
```

`data.jsonl` has one JSON object per line: `id`, `source`, `response`, `label` (1 = hallucinated, 0 = faithful).

Configurations: **a** single-LLM baseline, **b** + evidence citation, **c** + NLI ensemble, **d** cost-aware cascade, **e** cascade + calibration. Results go to `eval/results.csv` and `eval/results.md`. Configurations that need NLI or a calibrator are skipped with a note if those are missing. Use `--no-cache` for timing runs.

## Deploy to Render with Groq

Docker, health check path `/api/health`, environment variables `LLM_PROVIDER=groq`, `GROQ_API_KEY`, `GROQ_MODEL`. The default image is LLM-only. To include NLI, build with `--build-arg INSTALL_NLI=true`, set `ENABLE_NLI=true`, and use an instance with enough memory for PyTorch. To include semantic RAG, build with `--build-arg INSTALL_RAG=true` or install `requirements-rag.txt`; the first run downloads the embedding model. Set `API_AUTH_TOKEN` for API bearer-token protection.

## Known limits

- Verification is grounded in the user-provided source. It does not establish truth beyond that source.
- Semantic retrieval and FAISS are optional. Without the extra dependencies, the system falls back to BM25 retrieval.
- FAISS indexes are built in memory per request; this is not a persistent multi-user document vector database.
- The built-in rate limiter is in-memory and per process. Multi-worker deployments need a shared limiter.
- API bearer-token protection is optional and must be configured for public deployments. Use HTTPS and a secret manager.
- A passing test suite is not a measured hallucination accuracy result. Run the labelled evaluation datasets before making accuracy claims.

## RAG-based evidence retrieval (added)

The verifier now uses a retrieval-augmented evidence path before claim checking:

1. Split the source into overlapping sentence-aware chunks.
2. Retrieve relevant chunks for the extracted claims with BM25.
3. When the optional semantic stack is installed, combine BM25 and dense
   sentence-transformer retrieval using reciprocal-rank fusion. FAISS
   `IndexFlatIP` is used as the in-memory vector index; NumPy cosine ranking is
   the fallback when FAISS is unavailable.
4. Pass the compact retrieved evidence context to the existing LLM/NLI
   verification ensemble.
5. Map evidence citations back to exact offsets in the original source so the
   dashboard can highlight the original document.

BM25-only retrieval works with the base requirements. To enable semantic
retrieval and FAISS:

```bash
pip install -r requirements-rag.txt
```

The first semantic run downloads the configured sentence-transformer model.
Set `RAG_ENABLE_SEMANTIC=false` to disable semantic retrieval, or configure
`RAG_EMBEDDING_MODEL`, `RAG_TOP_K` and `RAG_CONTEXT_CHARS`.

The faithfulness score now divides supported claims by all claims, including
claims requiring review. `resolved_faithfulness_score` and
`verification_coverage` are also reported separately. An unsupported claim
means the supplied source did not establish it; it is not automatically proof
that the claim is false.

## Security configuration

Set `API_AUTH_TOKEN` to require `Authorization: Bearer <token>` on `/api/check`.
For public deployments, configure HTTPS, trusted proxy handling, and a
production secret manager. The built-in in-memory rate limiter is per process;
use a shared limiter for multi-worker deployments.

## Six-phase engineering upgrade checklist

- [x] Phase 1 — scoring semantics, review coverage, contradiction handling,
      bounded uploads, safer rate-limit identity, optional API bearer auth.
- [x] Phase 2 — sentence-aware chunking, BM25 retrieval, optional dense
      embeddings, FAISS vector search, reciprocal-rank fusion, source-offset
      provenance, and dashboard retrieval metrics.
- [x] Phase 3 — retain the LLM/NLI cascade, ensemble disagreement handling,
      and confidence calibration. Validate calibration with held-out data.
- [x] Phase 4 — retain iterative correction, force another correction pass for
      newly introduced unsupported claims, and include internal conflicts in
      correction decisions.
- [x] Phase 5 — retain response-level HaluEval evaluation and add a labelled
      retrieval evaluation utility reporting recall and MRR.
- [x] Phase 6 — API token option, safer error responses, upload read limits,
      configuration documentation, and optional Docker install switches.

### Retrieval evaluation dataset

Create a JSONL file where each row has `id`, `source`, `query`, and `relevant`
(an array of exact or distinctive source sentence snippets), then run:

```bash
python -m eval.retrieval_eval data/retrieval_eval.jsonl --top-k 4
```

The report measures retrieval relevance only. It does not substitute for a
labelled end-to-end hallucination benchmark.

### Run all tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

### Enable the full optional stack

```bash
pip install -r requirements-rag.txt
pip install -r requirements-nli.txt
```

Set `ENABLE_NLI=true` to activate DeBERTa. Semantic RAG loads the configured
Sentence Transformer model lazily; set `RAG_ENABLE_SEMANTIC=false` for BM25-only
operation. Set `API_AUTH_TOKEN` for bearer-token protection on `/api/check`.


## Six-phase end-to-end workflow (expanded)

The application now runs and reports six explicit phases:

1. **Knowledge ingestion:** paste text, upload up to 10 TXT/MD/PDF/DOCX files, and/or supply up to 5 public HTTPS HTML/TXT URLs. Sources are bounded, labelled and retained in a provenance manifest.
2. **Intelligent retrieval:** BM25 retrieval is always available; optional Sentence Transformers + FAISS/NumPy provide dense retrieval and reciprocal-rank fusion.
3. **Claim extraction:** split the candidate answer into atomic factual claims.
4. **Multi-layer verification:** evidence-cited LLM verification, optional DeBERTa NLI, ensemble/cascade decisions and internal contradiction checks.
5. **Correction and confidence:** iterative correction and re-verification, confidence calibration when a fitted calibrator exists, and an abstention recommendation when too many claims are unsupported or need review.
6. **Evaluation and reporting:** claim verdicts, faithfulness/coverage metrics, retrieval metrics, phase execution report and the offline/labelled evaluation CLI.

### Multi-source ingestion safety

URL ingestion is limited to user-supplied public HTTPS URLs, standard port 443, HTML/plain-text content, an 8-second timeout and 2 MB per URL. Redirect destinations are revalidated; private, loopback and reserved IP addresses are blocked. This is URL ingestion, not a general web search engine or an automatic arXiv connector. Do not treat fetched pages as inherently trustworthy; choose reputable sources.

Install the optional semantic stack with `pip install -r requirements-rag.txt`. DOCX parsing is included through `python-docx` in the base requirements.

### Evaluation and honest performance reporting

Run `pytest` for offline regression tests. For an empirical detector score, run the labelled evaluation workflow against HaluEval/TruthfulQA or a curated, held-out dataset, record precision/recall/F1/AUROC and retrieval Recall@k/MRR, and compare against a baseline. Passing software tests is not the same as a 9.5/10 detection-accuracy benchmark.

### Certificate export and multi-worker rate limiting

After a verification run, use **Download verification certificate** in the results panel to export the certificate JSON and verification summary. The export contains hashes/evidence spans rather than embedding the full source document.

The default rate limiter is process-local. For multiple workers/replicas, configure `RATE_LIMIT_REDIS_URL=redis://...` to share a Redis-backed fixed-window counter. If Redis is unavailable, the application logs the failure and falls back to the per-process limiter; this fallback is not globally shared. Protect Redis with network controls and credentials in production.

### Experiment runbook and prior-art notes

See [`docs/EXPERIMENT_RUNBOOK.md`](docs/EXPERIMENT_RUNBOOK.md) for two-dataset benchmark commands, seeds, metadata retention, calibration, guard false-positive auditing, and NLI checks. See [`docs/PRELIMINARY_PRIOR_ART_SEARCH.md`](docs/PRELIMINARY_PRIOR_ART_SEARCH.md) for a preliminary non-legal prior-art search summary. No empirical benchmark or calibration scores are included until the experiments are actually run.
