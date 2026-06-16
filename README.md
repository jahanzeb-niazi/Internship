# Learning Agent — Tool Calling, ReAct & Gemini

A **minimal Python agent** to learn these concepts hands-on:

| Topic | File | What you learn |
|-------|------|----------------|
| Mock APIs | `mock_apis.py` | Fake weather + task services (no network) |
| Tool calling | `tools.py` | How tools are defined and registered |
| LLM tool choice | `gemini_llm.py` / `mock_llm.py` | How the model picks a tool from descriptions |
| ReAct loop | `react_agent.py` | Reason -> Act -> Observe -> Repeat |
| Approval gates | `approval.py` | Human must approve risky actions |

## Setup (Gemini — default)

1. Get a free API key: [Google AI Studio](https://aistudio.google.com/apikey)
2. Install dependencies:

```bash
cd learning-agent
pip install -r requirements.txt
```

3. Create your `.env` file:

```bash
copy .env.example .env
```

4. Paste your key into `.env`:

```
GEMINI_API_KEY=your_actual_key_here
GEMINI_MODEL=gemini-2.0-flash
```

## Quick start

```bash
python main.py                          # interactive (Gemini)
python main.py "What's the weather in Tokyo?"
python main.py --mock "list my tasks"   # offline, no API key
```

Try these prompts:

```
What's the weather in Tokyo?
List my tasks
Add task: buy groceries
Delete task 1
```

Type `demo` to run all examples. Type `quit` to exit.

---

## Gemini vs Mock

| | **Gemini** (default) | **Mock** (`--mock`) |
|---|---------------------|---------------------|
| Tool selection | Real model reads descriptions | Keyword rules in code |
| API key | Required | Not needed |
| Best for | Realistic behavior | Reading decision logic |

Both use the **same** ReAct loop, tools, mock APIs, and approval gates.

---

## How Gemini tool calling works (deeper)

### What gets sent to Gemini

Every `decide()` call sends:

1. **System instruction** — agent rules (`gemini_llm.py`)
2. **Function declarations** — from `tools_for_gemini()` (name, description, parameters)
3. **Conversation history** — user message, then model function calls + your tool results

### What Gemini returns

Either:

- **`function_call`** — `{ name: "get_forecast", args: { city: "Tokyo" } }` -> ReAct ACT step
- **Plain text** — final answer -> done

### Why we disable automatic function calling

The SDK can auto-run tools for you. We turn that off:

```python
automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
```

So **you** still see each ReAct step and **approval gates** run before risky tools execute.

### Multi-turn flow (real Gemini conversation)

```
Turn 1  USER:   "Weather in Tokyo?"
        MODEL:  function_call(get_forecast, {city: "Tokyo"})
        YOU:    run tool -> {temp: 22, sunny}
        USER*:  function_response with result  (* SDK role=user)

Turn 2  MODEL:  "It's 22C and sunny in Tokyo."
```

Open `gemini_llm.py` and trace: `reset()` -> `decide()` -> `record_tool_result()` -> `decide()`.

---

## Step-by-step walkthrough

### Step 1 — Mock APIs (`mock_apis.py`)

Agents call external services. We simulate:

- **Weather**: `fetch_weather("Tokyo")` -> temp, condition, humidity
- **Tasks**: in-memory list with list, create, delete

### Step 2 — Tools (`tools.py`)

```python
Tool(
    name="get_forecast",
    description="...",       # Gemini reads this to decide WHEN to call
    parameters={...},        # JSON schema for arguments
    requires_approval=False, # safe reads = no gate
    handler=fetch_weather,   # what actually runs
)
```

`tools_for_gemini()` converts these to Gemini's `function_declarations` format.

### Step 3 — LLM backends

- **`gemini_llm.py`** — real AI Studio API, manual function calling
- **`mock_llm.py`** — keyword rules for offline learning

Both implement the same interface (`reset`, `decide`, `record_tool_result`, `record_denial`).

### Step 4 — ReAct loop (`react_agent.py`)

```
REASON  -> llm.decide()           (Gemini or mock)
ACT     -> run_tool()              (+ approval if needed)
OBSERVE -> llm.record_tool_result()
REPEAT  -> llm.decide() again until FinalAnswer
```

### Step 5 — Human-in-the-loop (`approval.py`)

| Tool | Approval? | Why |
|------|-----------|-----|
| `get_forecast`, `list_tasks` | No | Read-only |
| `create_task`, `delete_task` | **Yes** | Writes / destructive |

---

## File map

```
learning-agent/
├── mock_apis.py    # fake weather + task backend
├── tools.py        # tool definitions + run_tool()
├── llm_types.py    # ToolCall, FinalAnswer, LLMBackend protocol
├── gemini_llm.py   # Google AI Studio (real LLM)
├── mock_llm.py     # offline keyword LLM
├── config.py       # loads GEMINI_API_KEY from .env
├── approval.py     # CLI approval prompt
├── react_agent.py  # ReAct loop
├── main.py         # CLI entry
├── .env.example    # copy to .env
└── README.md
```


