# AI QA Agent

[![Unit Tests](https://github.com/fernando-benavides95/AI-QA-AGENT/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/fernando-benavides95/AI-QA-AGENT/actions/workflows/unit-tests.yml)
[![Evaluation Tests](https://github.com/fernando-benavides95/AI-QA-AGENT/actions/workflows/evaluation-tests.yml/badge.svg)](https://github.com/fernando-benavides95/AI-QA-AGENT/actions/workflows/evaluation-tests.yml)

A personal QA lab for learning AI test engineering. A multi-agent system converts plain-English feature descriptions into structured test suites through an iterative generator–critic loop — and the system itself is the subject being tested.

The testing problem is concrete: AI agents are non-deterministic. The same input rarely produces the same output. This breaks the core assumption of traditional automated testing and calls for a different kind of pyramid — one where each layer represents how much uncertainty it is designed to tolerate, not just how the tests are written.

Built with LangGraph, LangChain, Google Gemini, DeepEval, and Tavily.

## What this demonstrates

**Multi-agent orchestration**
- Generator–critic loop with LangGraph, structured output with Pydantic, feedback accumulation across iterations
- Two independent graphs with clean handoff — Feature Validator feeds the Generator–Critic pipeline
- Human-in-the-loop via LangGraph's `interrupt()` / `Command(resume=)` mechanism

**Agentic tool use (ReAct pattern)**
- LLM decides whether and when to call tools via `bind_tools()` and `ToolNode`
- Tavily web search called only when the feature involves unfamiliar technology or protocols
- `finalize_feature` tool signals readiness — the LLM chooses when it has enough context

**AI testing pyramid — four layers**
- **Layer 1** — Does each component produce the right structure? Deterministic unit tests: pure functions, routing logic, Pydantic schemas — no LLM calls
- **Layer 2** — Is a single generated output actually good? LLM-as-judge: DeepEval Answer Relevancy and custom G-Eval rubric
- **Layer 3** — Is the system stable across many outputs? Probabilistic assertions: range-based assertions, tolerance thresholds, drift detection across repeated runs
- **Layer 4** — How does it behave under pressure? Adversarial and regression: red-teaming, prompt injection, golden dataset

**Evaluation techniques**
- LLM-as-judge (manual and DeepEval wrapper)
- G-Eval: rubric-based evaluation encoding QA acceptance criteria as scoring steps
- Prompt regression: golden dataset with per-entry relevancy thresholds
- Non-determinism as a measurable property — tolerance windows instead of exact assertions

## How it works

### Full pipeline (`run_agent.py`)

```
You → Feature Validator → Generator → Critic → Test Suite
```

1. The **Feature Validator** receives raw user input, classifies it as a software feature, and opens a clarifying conversation — one question at a time. It uses web search (Tavily) when the feature involves unfamiliar technology. Once it has enough context it presents a refined feature description for user approval before handing off.
2. The **Generator** agent produces a structured test suite (positive, negative, and edge cases) from the refined description.
3. The **Coverage Tool** runs a deterministic analysis — category distribution, priority breakdown, duplicate detection.
4. The **Critic** agent reviews the test suite against the coverage report and either approves or sends actionable feedback to the Generator.
5. Steps 2–4 repeat until the Critic approves or the iteration limit is reached.

### Direct generator access (`run_generator_app.py`)

Skips the Feature Validator and feeds a feature description directly to the Generator–Critic loop. Used for Layer 1–4 tests.

## Project structure

```
.
├── app/
│   ├── feature_validator.py  # Feature Validator graph, ValidatorState, tools
│   ├── agents.py             # Generator and Critic agents, AgentState
│   ├── graph.py              # Generator–Critic LangGraph workflow
│   ├── models.py             # Pydantic schemas: TestCase, TestSuite, CriticResponse, FeatureValidation
│   └── tools.py              # analyze_test_coverage tool and report formatter
├── test/
│   ├── conftest.py           # Shared pytest fixtures
│   ├── unit/                 # Layer 1 — deterministic, no LLM calls
│   ├── evaluation/           # Layer 2 — LLM-as-judge quality evaluation (DeepEval)
│   ├── consistency/          # Layer 3 — structural invariants across repeated runs
│   ├── adversarial/          # Layer 4 — adversarial inputs, prompt injection, regression
│   ├── conversational/       # Layer 5 — multi-turn flow and tool-use evaluation (planned)
│   └── fixtures/             # golden_dataset.json and other test data
├── .github/workflows/
│   ├── unit-tests.yml        # Runs on every push — no LLM calls, no cost
│   └── evaluation-tests.yml  # Manual trigger only — LLM tests are too slow and flaky for CI
├── run_agent.py              # Full pipeline entry point
├── run_generator_app.py      # Direct generator entry point
├── pytest.ini
└── requirements.txt
```

## Setup

**1. Prerequisites**

Python 3.10 or higher.

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure environment**

Create a `.env` file in the project root:
```
GOOGLE_API_KEY="your_key_here"
GEMINI_MODEL_NAME="your_model_here"
TAVILY_API_KEY="your_key_here"
```

- Google API key: [Google AI Studio](https://aistudio.google.com/app/apikey)
- Tavily API key: [tavily.com](https://tavily.com) — free tier available. Used by the Feature Validator for web search. If omitted, search degrades gracefully and the validator continues without it.

## Usage

**Full pipeline — Feature Validator + Generator–Critic:**
```bash
python run_agent.py
```

**Direct generator access:**
```bash
python run_generator_app.py
```

Type `exit` at any prompt to quit.

## Testing

Each layer tolerates more uncertainty than the one below it. Layer 1 tolerates none — it tests deterministic scaffolding with no LLM involved. Each layer above accepts more variance in exchange for testing something closer to real model behavior. Human evaluation sits above all four as the ground truth that automated layers approximate but cannot replace.

```
                            Human evaluation — ground truth, not automated ↑
            /\
           /  \         Layer 4 — Adversarial + Regression
          / L4 \        Red-teaming, prompt injection, golden dataset regression
         /------\
        /        \      Layer 3 — Consistency / Non-determinism
       /    L3    \     Probabilistic assertions across repeated runs
      /------------\
     /              \   Layer 2 — LLM Evaluation
    /      L2        \  Answer Relevancy, G-Eval (DeepEval) — real LLM calls, cost $
   /------------------\
  /                    \ Layer 1 — Deterministic unit tests
 /          L1          \ Pure functions, routing logic, Pydantic schemas — no LLM, CI safe
/------------------------\
```

Layer 1 is the only CI-safe layer. Layers 2–4 make real LLM API calls — variable output, API cost, and results that can differ between runs for reasons unrelated to code changes. Running them on every push would produce false failures and erode trust in the suite. The `evaluation-tests.yml` workflow is manual-trigger only for this reason.

### Layer 1 — Does each component produce the right structure?

Deterministic pytest tests on pure functions, LangGraph routing logic, state transitions, and Pydantic schema validation. No LLM calls. Tests AI components like any other software — orchestration logic, error handling, and state management are independent of model behavior and must be correct regardless of what the model returns.

```bash
pytest -m unit    # ~5s, no cost
```

`test/unit/`

### Layer 2 — Is a single generated output actually good?

LLM-as-judge evaluation via DeepEval. The challenge here is definitional: there is no single correct answer, so "good" must be encoded as an explicit rubric before any evaluation can happen.

- **Answer Relevancy** — does the generated test suite address what the feature description actually asks?
- **G-Eval** — a custom rubric encoding QA acceptance criteria as step-by-step scoring instructions

One agent run, two judge calls. Evaluator model variance stacks on top of generator variance — scores are directionally meaningful, not exact, and should be read as signal rather than ground truth.

```bash
pytest -m evaluation    # ~2 min, LLM cost
```

`test/evaluation/`

### Layer 3 — Is the system stable across many outputs?

Repeated runs with probabilistic assertions and calibrated tolerance thresholds. The core mindset shift from traditional testing: tolerance windows replace exact-value assertions. The goal is not to assert that any two outputs are identical — they won't be — but to detect whether a prompt change causes output drift, count bloat, or structural instability across runs.

```bash
pytest -m consistency    # ~15 min, LLM cost (5 full agent runs)
```

`test/consistency/`

### Layer 4 — How does it behave under pressure?

Two approaches under this layer, both in `test/adversarial/`:

**Adversarial** — red-teaming with ambiguous, contradictory, and injection-attempt inputs. Tests that the system fails gracefully rather than producing confidently wrong output.

**Regression** — a golden dataset with per-feature relevancy thresholds acts as a safety net for prompt changes. A threshold breach signals that a change shifted model behavior on known-good inputs, not just on edge cases.

```bash
pytest -m adversarial    # ~10 min, LLM cost
pytest -m regression     # ~10 min, LLM cost (3 golden dataset entries)
```

`test/adversarial/`

### Layer 5 — Multi-turn flow and tool-use evaluation *(planned)*

Evaluation of the Feature Validator's conversational behavior: does it ask the right clarifying questions, invoke Tavily search at the right moment, and route correctly to the approval step? The folder exists; the tests do not yet.

`test/conversational/`

### Running the suite

| Command | Layer | Est. runtime | Notes |
|---|---|---|---|
| `pytest -m unit` | 1 | ~5s | No LLM calls — CI safe |
| `pytest -m evaluation` | 2 | ~2 min | 1 agent run + 2 judge calls |
| `pytest -m consistency` | 3 | ~15 min | 5 full agent runs |
| `pytest -m adversarial` | 4 | ~10 min | adversarial inputs + judge |
| `pytest -m regression` | 4 | ~10 min | 3 golden dataset entries |
| `pytest -m llm` | 2–4 | ~35 min | All LLM layers |
| `pytest` | all | ~35 min | Full suite |