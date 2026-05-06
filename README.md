# AI QA Agent

[![Unit Tests](https://github.com/fernando-benavides95/AI-QA-AGENT/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/fernando-benavides95/AI-QA-AGENT/actions/workflows/unit-tests.yml)
[![Evaluation Tests](https://github.com/fernando-benavides95/AI-QA-AGENT/actions/workflows/evaluation-tests.yml/badge.svg)](https://github.com/fernando-benavides95/AI-QA-AGENT/actions/workflows/evaluation-tests.yml)

A learning lab for AI test engineering. A multi-agent system that takes a plain-English feature description through a conversational refinement loop, then generates a structured test suite through an iterative generator–critic loop.

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

**AI testing pyramid — five layers**
- **Layer 1** — Deterministic unit tests: pure functions, routing logic, Pydantic schemas, no LLM
- **Layer 2** — LLM-as-judge: DeepEval with Answer Relevancy and custom G-Eval rubric
- **Layer 3** — Non-determinism: range-based assertions across repeated runs
- **Layer 4** — Adversarial + regression: red-teaming, prompt injection, golden dataset
- **Layer 5** — Conversational: multi-turn flow evaluation, tool-use decisions, approval routing

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
│   ├── conversational/       # Layer 5 — multi-turn flow and tool-use evaluation
│   └── fixtures/             # golden_dataset.json and other test data
├── .github/workflows/
│   ├── unit-tests.yml        # Runs on every push — no LLM calls, no cost
│   └── evaluation-tests.yml  # Manual trigger — Layer 2 LLM-as-judge
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

The test suite follows a five-layer AI testing pyramid.

```
            /\
           /  \         Layer 5 — Conversational
          / L5 \        Multi-turn flow, tool-use decisions, approval routing
         /------\
        /        \      Layer 4 — Adversarial + Regression
       /    L4    \     Golden dataset, prompt regression, adversarial inputs
      /------------\
     /              \   Layer 3 — Consistency / Non-determinism
    /      L3        \  Statistical assertions across repeated runs
   /------------------\
  /                    \ Layer 2 — LLM Evaluation
 /         L2           \ Answer Relevancy, G-Eval (DeepEval) — real LLM calls, cost $
/------------------------\
          L1               Layer 1 — Deterministic unit tests
                           Tools, schemas, routing logic — no LLM, CI safe
```

Above Layer 1, **every test makes real LLM API calls**. Use pytest markers to control cost:

| Command | Layer | Est. runtime | Notes |
|---|---|---|---|
| `pytest -m unit` | 1 | ~5s | No LLM calls — CI safe |
| `pytest -m evaluation` | 2 | ~2 min | 1 agent run + 2 judge calls |
| `pytest -m consistency` | 3 | ~15 min | 5 full agent runs |
| `pytest -m adversarial` | 4 | ~10 min | adversarial inputs + judge |
| `pytest -m regression` | 4 | ~10 min | 3 golden dataset entries |
| `pytest -m conversational` | 5 | ~10 min | multi-turn validator flows |
| `pytest -m llm` | 2–5 | ~45 min | All LLM layers |
| `pytest` | all | ~45 min | Full suite |
