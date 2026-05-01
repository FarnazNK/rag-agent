# RAG Agent

A production-style LangGraph + RAG agent with a built-in evaluation harness.

Built as a portfolio project for AI engineering roles that call for **agent
orchestration, retrieval infrastructure, and evaluation tooling**. The code
is small on purpose — under 2,000 lines — but every piece reflects a design
decision I'd defend in a code review.

---

## What's in the box

Three things, wired together:

1. **An agent** — a LangGraph DAG that routes queries, rewrites them for
   better retrieval, grades the retrieved context, and loops if the grade
   is poor. Topology, not a ReAct loop, so control flow is explicit and
   debuggable.
2. **A hybrid RAG pipeline** — dense (Chroma) + sparse (BM25) retrieval
   fused with reciprocal rank fusion. The vector store sits behind a
   `Protocol`, so swapping Chroma for Pinecone is one class.
3. **An eval harness** — YAML-defined test cases, four pluggable scorers
   (exact match, substring, retrieval recall, LLM-as-judge), a runner
   with Rich console output, JSON export, and a `--fail-under` threshold
   that makes it CI-ready.

A sample sample corpus (PTO policy, performance reviews, onboarding) and
a matching eval suite ship with the repo so the whole thing runs end-to-end
with one command.

---

## Architecture

```
                       ┌──────────┐
              START ──►│  route   │
                       └────┬─────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
     refuse          answer_directly          retrieve
        │                   │                   │
        ▼                   ▼                   ▼
   ┌─────────┐         ┌──────────┐       ┌──────────────┐
   │ refuse  │         │  direct  │       │ rewrite_query│
   │  node   │         │  answer  │       └──────┬───────┘
   └────┬────┘         └─────┬────┘              │
        │                    │                   ▼
        │                    │             ┌──────────┐
        │                    │             │ retrieve │  (dense + BM25 + RRF)
        │                    │             └─────┬────┘
        │                    │                   │
        │                    │                   ▼
        │                    │             ┌──────────┐
        │                    │             │  grade   │
        │                    │             └─────┬────┘
        │                    │                   │
        │                    │      ┌────────────┴────────────┐
        │                    │      │                         │
        │                    │  relevant               irrelevant + iter<N
        │                    │      │                         │
        │                    │      ▼                         │
        │                    │ ┌──────────┐                   │
        │                    │ │ generate │◄───── irrelevant + iter≥N
        │                    │ └────┬─────┘
        │                    │      │
        └────────────────────┴──────┴─────► END
```

The state (`AgentState`) is a Pydantic model that flows through every node.
Each node returns a partial-state dict — LangGraph merges the updates.

---

## Design decisions worth pointing at

A few places where I made a non-obvious choice and the reasoning behind it:

**Topology, not ReAct.** A ReAct-style "let the LLM decide what to do next"
loop is seductive but hard to evaluate — every failure mode looks the same.
An explicit graph makes the router, rewriter, and grader addressable as
separate components. You can measure router accuracy independently of
generation quality. You can swap the grader without touching the retriever.

**Hybrid retrieval, not pure dense.** Vector search is great for semantic
recall but fragile for acronyms, IDs, and exact product names — all of
which show up constantly in HR / enterprise queries ("what about PTO?",
"find me the PIP policy"). BM25 on the same corpus catches those. RRF is
the cheapest robust fusion method; the `k=60` constant is the default from
Cormack et al. — no tuning needed.

**Grader with a fail-open policy.** If the grader's JSON output is malformed,
we treat the chunks as relevant rather than blocking the user. A broken
grader should not cause user-visible failures. It does get logged, loudly.

**LangSmith is optional.** If `LANGSMITH_API_KEY` is set, traces flow there
automatically via LangChain's auto-tracing. If not, we emit structured logs
and the agent runs identically. Observability shouldn't be a hard dependency
on a specific vendor.

**Lazy `Agent` import.** `from rag_agent import Agent` works, but the
top-level `__init__.py` uses PEP 562 lazy loading so that importing
`rag_agent.schemas` doesn't drag in LangGraph. This matters for
deployment scenarios where the eval scorers run in a smaller environment.

**Scorers are a Protocol, not a base class.** Any callable with the right
signature is a scorer. Adding a new metric is one function, not a subclass.

---

## How this maps to AI engineering roles

Common requirements for AI engineering / ML platform roles and where each
one lives in the code:

| Requirement                                | Where it lives in the code                              |
|---------------------------------------------|---------------------------------------------------------|
| Evaluation framework (offline evals)        | `src/rag_agent/evals/`                              |
| Evaluation datasets + test harnesses        | `data/eval_datasets/hr_smoke.yaml`, `evals/runner.py`   |
| Automated scoring pipelines                 | `scripts/run_evals.py` with `--fail-under` for CI       |
| Hallucination / response quality metrics    | `LLMJudgeScorer`, `RetrievalRecallScorer`               |
| LangGraph agent orchestration               | `src/rag_agent/graph.py`                            |
| Multi-turn conversation state               | `AgentState.messages` with `add_messages` reducer       |
| RAG pipelines + retrieval optimization      | `src/rag_agent/retrieval.py` (hybrid + RRF)         |
| Vector database                             | `src/rag_agent/vectorstore.py` (Chroma, swappable)  |
| LLM observability / tracing                 | `src/rag_agent/observability.py` (+ LangSmith opt.) |
| Build-vs-buy abstractions                   | `VectorStore` protocol, `build_llm` provider factory    |
| Production-grade Python                     | Pydantic schemas, tenacity retries, structlog, tests    |
| Domain expertise signal                      | `data/sample_corpus/` + `data/eval_datasets/`           |

---

## Running it

### Setup

```bash
# Clone + install
pip install -e ".[dev]"

# Set API keys
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY (for generation) and OPENAI_API_KEY (for embeddings)
```

### Index the sample corpus

```bash
rag-agent ingest --data-dir data/sample_corpus
```

### Ask it something

```bash
rag-agent ask "How many PTO days do new hires get?"

# With intermediate state visible
rag-agent ask "What does it take to get promoted?" --trace
```

### Run the eval suite

```bash
python scripts/run_evals.py

# Filter by tag
python scripts/run_evals.py --tag smoke

# Skip the expensive LLM judge during fast iteration
python scripts/run_evals.py --no-judge

# CI mode: fail the run if pass rate drops below threshold
python scripts/run_evals.py --fail-under 0.9 --out results.json
```

### Run unit tests

```bash
pytest                         # everything
pytest tests/test_graph.py     # just the graph wiring (mocked LLMs, no network)
```

---

## What I'd build next

The three things I'd prioritize if this became a real project, in order:

1. **Production tracing** — LangSmith integration is wired but the project
   would benefit from custom trace attributes (retrieval scores, iteration
   count, grader confidence) surfaced in the traces for easier debugging.
2. **Chunk-level citations** — the answer prompt asks for `[source: file]`
   citations, but a real system should track which specific chunks
   supported which claims. Useful for hallucination detection in eval.
3. **Evaluation dashboard** — the JSON output from `run_evals.py` is
   structured but reading it is tedious. A small Streamlit or Next.js
   dashboard that compares runs over time would be the obvious next step.

A few things I deliberately *didn't* build and would flag up front in an
interview:

- **No fine-tuning pipeline.** Some roles list RLHF / LoRA as nice-to-haves.
  I chose breadth (eval + agent + RAG end-to-end) over depth in any one
  area. Fine-tuning without an eval framework to measure its effect is
  the wrong order anyway.
- **No production deployment.** No Docker, no Lambda, no k8s. The project
  is structured so a deployment layer can wrap `Agent` cleanly — but
  adding one would have been noise for a portfolio.
- **No multi-agent topology.** Some roles call for it. I think multi-agent
  systems are usually the wrong abstraction when a well-designed single-
  agent graph would do, and wanted this project to reflect that opinion.

---

## Project structure

```
rag-agent/
├── src/rag_agent/
│   ├── __init__.py          # public API surface, PEP 562 lazy imports
│   ├── agent.py             # public Agent class
│   ├── config.py            # pydantic-settings
│   ├── graph.py             # LangGraph topology
│   ├── llm.py               # LLM provider factory + retries
│   ├── observability.py     # structlog + optional LangSmith
│   ├── prompts.py           # centralized prompt templates
│   ├── retrieval.py         # hybrid dense + BM25 + RRF
│   ├── schemas.py           # Pydantic state + chunk types
│   ├── vectorstore.py       # Chroma wrapper behind a Protocol
│   ├── cli.py               # typer CLI
│   └── evals/
│       ├── dataset.py       # YAML loader + EvalCase / EvalDataset
│       ├── scorers.py       # ExactMatch / Contains / RetrievalRecall / LLMJudge
│       └── runner.py        # runs cases, aggregates, prints report
├── tests/
│   ├── test_schemas.py      # state contract
│   ├── test_retrieval.py    # RRF fusion + BM25 determinism
│   ├── test_scorers.py      # scorer logic
│   └── test_graph.py        # graph topology with mocked LLMs
├── scripts/
│   └── run_evals.py         # CLI eval runner
├── data/
│   ├── sample_corpus/       # sample markdown docs
│   └── eval_datasets/       # YAML test cases
├── .env.example
├── pyproject.toml
└── README.md
```

---

*Built by Farnaz Nasehi.*
