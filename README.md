# AI-Powered Candidate Screening Assistant

An intelligent candidate screening system that processes raw job applications and produces structured qualification decisions, grounded in a real knowledge base, with an autonomous agent layer for follow-up actions.

## Architecture

```
Raw Application Text
        │
        ▼
┌─────────────────┐     ┌──────────────────────┐
│  Input Parser    │     │  Knowledge Base (RAG) │
│  (Gemini + Pydantic)│  │  ChromaDB + ST       │
└────────┬────────┘     └──────────┬───────────┘
         │                         │
         ▼                         ▼
┌─────────────────────────────────────────────┐
│         Screening Agent (LangGraph)          │
│  ┌─────────┐  ┌───────────┐  ┌───────────┐ │
│  │ Qualify  │→ │  Decide   │→ │  Execute  │ │
│  │         │  │  Action   │  │  Tools    │ │
│  └─────────┘  └───────────┘  └───────────┘ │
│                      │                       │
│              ┌───────▼───────┐              │
│              │ Approval Gate │              │
│              │  (Human CLI)  │              │
│              └───────────────┘              │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ Outreach Email │
              │  (Personalized)│
              └────────────────┘
```

## Features

- **Input Parsing**: Extracts structured candidate profiles (name, skills, experience, education) from free-form text
- **RAG-Grounded Qualification**: Decisions cite specific job descriptions, rubrics, and hiring criteria from ChromaDB
- **Confidence Scoring**: Every decision includes a confidence score; low confidence → automatic human review flag
- **Autonomous Agent**: LangGraph-based agent decides: qualify, schedule interview, send notification, or escalate
- **Human-in-the-Loop**: State-modifying actions (scheduling, sending) require CLI approval before execution
- **Schema Validation**: All LLM outputs are Pydantic-validated with automatic retry on failure
- **Comprehensive Logging**: Every LLM call is logged with function name, token counts, latency, and prompt version
- **Personalized Outreach**: Generates emails referencing specific candidate skills and role — not generic templates
- **Eval Harness**: 17 labelled test cases with pytest + custom accuracy reporting by difficulty

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 3.1 Flash lite|
| Agent Framework | LangGraph |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Schema Validation | Pydantic v2 |
| Logging | JSON Lines (JSONL) |
| CLI Formatting | Rich |
| Testing | pytest |

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Gemini API key
# Edit .env file:
GOOGLE_API_KEY=your-api-key-here

# 3. Run the screening assistant
python main.py
```

## Usage

### Interactive Mode
```bash
python main.py
# Choose option 1 for the sample Ayesha application
# Choose option 2 to paste your own application text
# Choose option 3 to load from a file
```

### Run Evals
```bash
# Pytest
pytest eval/test_screening.py -v

# Custom report
python eval/run_eval.py

# With prompt version tag (for A/B comparison)
python eval/run_eval.py v2.0
```

## Project Structure

```
├── config.py              # Central configuration
├── schemas.py             # Pydantic models
├── logger.py              # LLM call logger
├── llm_service.py         # Gemini wrapper with validation + retry
├── parser.py              # Input parsing
├── rag.py                 # ChromaDB knowledge base
├── qualifier.py           # RAG-grounded qualification
├── tools.py               # Tool definitions
├── approval.py            # Human-in-the-loop gate
├── agent.py               # LangGraph screening agent
├── outreach.py            # Personalized email drafter
├── main.py                # CLI entry point
├── knowledge_base/        # Markdown documents (JDs, rubrics, criteria)
├── eval/                  # Eval harness
│   ├── test_cases.json    # 17 labelled test cases
│   ├── test_screening.py  # pytest tests
│   └── run_eval.py        # Custom eval report
└── logs/                  # LLM call logs
```

## Knowledge Base Documents

- `backend_engineer_jd.md` — Backend Engineer job description
- `frontend_engineer_jd.md` — Frontend Engineer job description
- `data_engineer_jd.md` — Data Engineer job description
- `engineering_rubric.md` — Scoring rubric with 5 dimensions and weights
- `hiring_criteria.md` — General hiring criteria and guidelines

## Confidence Threshold

Default threshold: **0.7**

- Confidence ≥ 0.7 → Decision stands as-is
- Confidence < 0.7 → Automatically flagged for human review, regardless of the LLM's decision
