# Multi-Agent Test Case Generator

[![Unit Tests](https://github.com/fernando-benavides95/AI-QA-AGENT/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/fernando-benavides95/AI-QA-AGENT/actions/workflows/unit-tests.yml)
[![Evaluation Tests](https://github.com/fernando-benavides95/AI-QA-AGENT/actions/workflows/evaluation-tests.yml/badge.svg)](https://github.com/fernando-benavides95/AI-QA-AGENT/actions/workflows/evaluation-tests.yml)

A learning lab for AI test engineering. It implements a generator–critic multi-agent loop that produces structured software test cases from a plain-English feature description.

Built with LangGraph, LangChain, Google Gemini, and DeepEval.

## What this demonstrates

This project explores AI test engineering techniques hands-on:

- **Multi-agent orchestration** — generator–critic loop with LangGraph, structured output with Pydantic, feedback accumulation across iterations
- **LLM-as-judge** — manual implementation and DeepEval wrapper; scoring LLM output probabilistically instead of asserting exact values
- **G-Eval** — rubric-based evaluation where QA domain knowledge is encoded as scoring criteria
- **Adversarial testing** — prompt injection (authority-mimicry attack), out-of-domain inputs, graceful degradation
- **Prompt regression** — golden dataset with per-entry quality thresholds; detects when a prompt change degrades output
- **Non-determinism as a property** — range-based assertions across repeated runs instead of fixed expected values
- **AI testing pyramid** — four-layer structure adapted from the traditional pyramid for LLM-based systems

## How it works

1. The **Generator** agent produces a structured test suite (positive, negative, and edge cases) for a given feature.
2. The **Coverage Tool** runs a deterministic analysis of the suite — category distribution, priority breakdown, duplicate detection.
3. The **Critic** agent reviews the test cases using both its own reasoning and the coverage report, then either approves or sends feedback to the Generator.
4. The loop repeats until the Critic approves or the iteration limit is reached.

## Project structure

```
.
├── app/
│   ├── agents.py       # Generator and Critic agents, AgentState
│   ├── graph.py        # LangGraph workflow and node orchestration
│   ├── models.py       # Pydantic schemas: TestCase, TestSuite, CriticResponse
│   └── tools.py        # analyze_test_coverage tool and report formatter
├── test/
│   ├── conftest.py           # Shared pytest fixtures
│   ├── unit/                 # Layer 1 — deterministic, no LLM calls
│   ├── evaluation/           # Layer 2 — LLM-as-judge quality evaluation (DeepEval)
│   ├── consistency/          # Layer 3 — structural invariants across repeated runs
│   ├── adversarial/          # Layer 4 — adversarial inputs, prompt injection, regression
│   └── fixtures/             # golden_dataset.json and other test data
├── pytest.ini
├── run_generator_app.py
├── requirements.txt
└── .env                # GOOGLE_API_KEY, GEMINI_MODEL_NAME
```

## Setup

**1. Prerequisites**

Python 3.10 or higher is required.

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure environment**

Create a `.env` file in the project root:
```
GOOGLE_API_KEY="your_key_here"
GEMINI_MODEL_NAME="gemini-1.5-pro"
```

A key can be obtained from [Google AI Studio](https://aistudio.google.com/app/apikey). If `GEMINI_MODEL_NAME` is omitted it defaults to `gemini-1.5-pro`.

## Usage

```bash
python run_generator_app.py
```

Enter a feature description at the prompt. The agent will iterate until the Critic approves or the maximum iteration count is reached.

**Example input:**
```
A file upload feature for a user's profile picture. It only accepts .jpg and .png files under 5MB. It must prevent any executable scripts from being uploaded.
```

## Testing

The test suite follows an AI-adapted testing pyramid.

```
          /\
         /  \         Layer 4 — Adversarial + Regression
        / L4 \        Golden dataset, prompt regression, adversarial inputs
       /------\
      /        \      Layer 3 — Consistency / Non-determinism
     /    L3    \     Statistical assertions across repeated runs
    /------------\
   /              \   Layer 2 — LLM Evaluation
  /      L2        \  Answer Relevancy, G-Eval (DeepEval) — real LLM calls, cost $
 /------------------\
/                    \
        L1             Layer 1 — Deterministic unit tests
                       Tools, schemas, graph structure — no LLM, CI safe
```

IMPORTANT NOTE: above Layer 1, **every test will call LLM**. Use pytest markers to control what runs where:

| Command | Layer | Est. runtime | Notes |
|---|---|---|---|
| `pytest -m unit` | 1 | ~5s | No LLM calls — CI safe |
| `pytest -m evaluation` | 2 | ~2 min | 1 agent run + 2 judge calls |
| `pytest -m consistency` | 3 | ~15 min | 5 full agent runs |
| `pytest -m adversarial` | 4 | ~10 min | 5 agent runs, 1 with judge |
| `pytest -m regression` | 4 | ~10 min | 3 golden dataset entries |
| `pytest -m llm` | 2–4 | ~35 min | All LLM layers in one shot |
| `pytest` | all | ~35 min | Full suite |
