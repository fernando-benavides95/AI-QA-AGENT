# Multi-Agent Test Case Generator

This project implements a multi-agent system for generating and refining software test cases, leveraging Google's Gemini models orchestrated by LangGraph. It includes a feedback loop where a "Generator" agent creates test cases based on a JSON structure, and a "Critic" agent reviews them for comprehensive coverage, especially focusing on edge cases and boundary conditions. DeepEval is integrated to measure the quality of the generated test cases.

## Features

*   **Intelligent Test Case Generation**: A "Generator" agent creates detailed test cases (description, input data, expected output, priority, category) based on a given feature description and a JSON structure.
*   **Critical Review & Feedback Loop**: A "Critic" agent evaluates the generated test cases for completeness, edge-case coverage, and adherence to best practices. If improvements are needed, it provides feedback, sending the process back to the Generator for refinement.
*   **LangGraph Orchestration**: The agents and their interactions are seamlessly managed using LangGraph, enabling complex workflows and iterative processes.
*   **Configurable Gemini Model Integration**: Utilizes `langchain_google_genai` to harness the advanced capabilities of Google's Gemini models (e.g., `gemini-1.5-pro`, `gemini-1.5-flash-preview`). The model name is configurable via the `.env` file.

## Project Structure

```
.
├── .env                      # Environment variables (e.g., GOOGLE_API_KEY, GEMINI_MODEL_NAME)
├── app/
│   ├── __init__.py           # Python package initializer
│   ├── agents.py             # Defines Generator and Critic agents, test case structure (JSON)
│   ├── graph.py              # Defines the LangGraph workflow and agent orchestration
│   └── tools.py              # Contains utility functions, e.g., JSON sanitization
├── test/
│   ├── __init__.py           # Python package initializer
│   └── test_agent.py         # DeepEval test script for evaluating generated test cases
└── requirements.txt          # List of Python dependencies
```

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.9+**
*   **Google API Key**: An API key for Google Gemini models. You can obtain one from the [Google AI Studio](https://aistudio.google.com/app/apikey).

## Setup

**Configure your Google API Key and Gemini Model**:
    Create or update the `.env` file in the root directory of the project:
    ```
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
    GEMINI_MODEL_NAME="gemini-1.5-pro" # Or "gemini-1.5-flash-preview", etc.
    ```
    *   Replace `"YOUR_GEMINI_API_KEY"` with your actual API key.
    *   Set `GEMINI_MODEL_NAME` to your desired Gemini model. If omitted, `gemini-1.5-pro` will be used by default.

## Usage

### Running the Test Case Generation Graph

The agent can be ran by executing a simple UI `run_generator_app.py`. You can run the file to generate test cases.

