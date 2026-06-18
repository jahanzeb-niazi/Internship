# Tool-Calling Agent Prototype

Working agent with **ReAct loop**, **action logging** (input/output), **graceful error handling**, and **human approval gates**.

## Setup

```bash
cd learning-agent
pip install -r requirements.txt
copy .env.example .env   # paste GEMINI_API_KEY from https://aistudio.google.com/apikey
```

## Run

```bash
python main.py "What's the weather in Tokyo?"
python main.py --mock "list my tasks"    # no API key
python main.py --show-logs               # view action log
python main.py                           # interactive (demo | logs | quit)
```

## Features

| Feature | Where | What it does |
|---------|-------|--------------|
| Tool calling | `tools.py` | 4 tools: weather + tasks |
| ReAct loop | `react_agent.py` | Reason -> Act -> Observe -> Repeat |
| Action logging | `action_logger.py` | Logs every tool call with input/output to console + `action_log.jsonl` |
| Error handling | `tools.py` | `run_tool()` catches errors, returns `ToolResult` — never crashes |
| Human approval | `approval.py` | `create_task` / `delete_task` require y/n before execution |

## Action log format

Each tool call is logged as JSON (one line per event):

```json
{
  "timestamp": "2026-06-16T12:00:00+00:00",
  "session_id": "a1b2c3d4",
  "step": 1,
  "tool": "get_forecast",
  "input": {"city": "Tokyo"},
  "status": "success",
  "output": {"city": "Tokyo", "temp_c": 22, "condition": "sunny"},
  "approval_required": false
}
```

Statuses: `pending` -> `approved`/`denied`/`success`/`error`

## Error handling

- **Unknown tool** -> logged as error, LLM gets error payload
- **Bad arguments** -> caught, logged, LLM can explain to user
- **API soft errors** (e.g. city not found) -> returned as output, LLM explains
- **LLM API failure** -> caught in ReAct loop, user sees friendly message

## Try error cases

```
What's the weather in Mars?     # soft error — city not in DB
Delete task 99                  # task not found (after approval)
Add task:                       # empty title validation error
```

## File map

```
learning-agent/
├── action_logger.py   # input/output logging
├── react_agent.py   # ReAct + logging + approval orchestration
├── tools.py         # tool registry + safe run_tool()
├── approval.py      # human-in-the-loop gates
├── gemini_llm.py    # Google Gemini backend
├── mock_llm.py      # offline backend
├── mock_apis.py     # fake weather + tasks
├── main.py          # CLI
└── action_log.jsonl # generated at runtime
```

## Deliverable checklist

- [x] Working tool-calling agent (Gemini + mock)
- [x] Action logging with input/output
- [x] Graceful tool error handling
- [x] Human approval for write/delete actions
