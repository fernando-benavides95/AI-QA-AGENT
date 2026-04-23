# Multi-Agent Test Case Generator

A learning lab for AI test engineering. It implements a generator–critic multi-agent loop that produces structured software test cases from a plain-English feature description.

Built with LangGraph, LangChain, and Google Gemini. DeepEval will be integrated for evaluation.

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
│   └── test_agent.py   # Evaluation tests (DeepEval — in progress)
├── run_generator_app.py
├── requirements.txt
└── .env                # GOOGLE_API_KEY, GEMINI_MODEL_NAME
```

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure environment**

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
