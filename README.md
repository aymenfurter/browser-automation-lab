# Browser Automation Lab

A comparative study of long-horizon browser automation using two distinct orchestration approaches. Both agents perform the same task — searching a target site for 20 Azure topics, capturing AI overviews and extracting presentation links — but differ fundamentally in how they manage context over extended multi-step sessions.

## The Two Approaches

| | LangGraph + Azure OpenAI | GitHub Copilot SDK |
|---|---|---|
| **LLM** | Azure OpenAI `gpt-4.1` | Copilot (via `gh` auth) |
| **Orchestration** | Graph-based (StateGraph with cycles) | Event-driven session |
| **Compaction** | Manual — summarize + `RemoveMessage` | Automatic — infinite sessions |
| **Auth** | `DefaultAzureCredential` (managed identity / `az login`) | `gh auth token` |
| **Infrastructure** | Azure OpenAI resource (Bicep) | None |

### Option 1: LangGraph + Playwright

A [StateGraph](https://langchain-ai.github.io/langgraph/concepts/low_level/#stategraph) models the agent as a cyclic directed graph:

```
Agent (LLM) ──→ Tools (Playwright) ──→ Should Continue?
     ↑                                        │
     └──── Compact (if messages > 100) ←──────┘
```

The LLM decides which tool to call, the tool executes against the browser, and a conditional edge checks whether compaction is needed before looping back. When the message history grows past 100 messages, older entries are summarized into a progress note and removed via [`RemoveMessage`](https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/).

Key files:
- `langgraph_agent/graph.py` — Graph definition, system prompt, conditional routing
- `langgraph_agent/compaction.py` — Threshold check, safe-cut logic (never splits tool_call/response pairs)
- `langgraph_agent/tools.py` — Playwright tools (navigate, click, fill, extract)

### Option 2: GitHub Copilot SDK + Playwright

An [event-driven session](https://github.com/nicolo-ribaudo/github-copilot-sdk-python) where the Copilot runtime manages the conversation loop. Tools are registered via `@define_tool` and invoked when the model emits `ExternalToolRequestedData`. Context compaction happens transparently inside the runtime ("infinite sessions") — no manual summarization needed.

Key files:
- `copilot_sdk_agent/agent.py` — Session setup, event handlers, tool registration
- `copilot_sdk_agent/tools.py` — `@define_tool` Playwright browser tools

## Live Dashboard

A web UI runs both agents side-by-side with live streaming:

```bash
python -m uvicorn web.server:app --port 8080
```

Features:
- Live browser screenshots (updated each step)
- Step-by-step activity log with tool calls
- AI Overviews panel showing captured content
- Compaction event highlighting
- Full logs tab

## Getting Started

### Prerequisites

- Python 3.11+
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) — for Option 1
- [GitHub CLI](https://cli.github.com/) with Copilot access — for Option 2
- An Azure subscription — for Option 1

### 1. Clone and install

```bash
git clone https://github.com/aymenfurter/browser-automation-lab.git
cd browser-automation-lab

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
playwright install chromium
```

### 2. Provision Azure OpenAI (Option 1 only)

```bash
az login

az deployment sub create \
  --location swedencentral \
  --name browser-auto-lab \
  --template-file infra/main.bicep \
  --parameters environmentName=browser-auto-lab \
               principalId=$(az ad signed-in-user show --query id -o tsv)
```

Then configure your endpoint:

```bash
cp .env.example .env
# Set AZURE_OPENAI_ENDPOINT to your deployed resource
```

### 3. Run agents standalone

```bash
# Option 1: LangGraph
python -m langgraph_agent.run

# Option 2: Copilot SDK (requires `gh auth login`)
python -m copilot_sdk_agent.run
```

### 4. Run the dashboard (both side-by-side)

```bash
uvicorn web.server:app --port 8080
# Open http://localhost:8080
```

## How Compaction Works

### LangGraph — Manual compaction

```python
COMPACTION_THRESHOLD = 100  # messages before triggering
keep_recent = 20            # messages to preserve

async def compact_messages(state):
    # Find a safe cut point (never split tool_call/response pairs)
    old = state["messages"][1:-keep_recent]
    summary = await llm.ainvoke([summarize_prompt] + old)
    return {
        "messages": [RemoveMessage(id=m.id) for m in old]
                   + [SystemMessage(content=f"[PROGRESS]: {summary}")]
    }
```

References:
- [LangGraph — Managing conversation history](https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/)
- [LangGraph — Delete messages](https://langchain-ai.github.io/langgraph/how-tos/memory/delete-messages/)

### Copilot SDK — Automatic compaction

```python
# The runtime handles it. You just observe:
session.on(SessionCompactionStartData, lambda e: print("Compacting..."))
session.on(SessionCompactionEndData, lambda e: print("Done"))
```

The Copilot runtime compresses older turns internally, maintaining a sliding context window without developer intervention.

## Project Structure

```
browser-automation-lab/
├── web/
│   ├── server.py           # FastAPI + SSE endpoints
│   ├── runners.py          # Instrumented agent runners (both)
│   ├── events.py           # AgentEmitter — SSE event broadcasting
│   └── dashboard.html      # Live UI
├── langgraph_agent/
│   ├── graph.py            # StateGraph + system prompt
│   ├── compaction.py       # Manual compaction logic
│   ├── tools.py            # Playwright tools
│   ├── state.py            # TypedDict state schema
│   └── run.py              # CLI entry point
├── copilot_sdk_agent/
│   ├── agent.py            # Session + event handlers
│   ├── tools.py            # @define_tool browser tools
│   └── run.py              # CLI entry point
├── shared/
│   ├── config.py           # 20 search queries, Azure config
│   └── models.py           # Shared Pydantic models
├── infra/
│   ├── main.bicep          # Subscription-level deployment
│   └── modules/
│       ├── openai.bicep    # Azure OpenAI + gpt-4.1
│       └── storage.bicep   # Storage account
├── pyproject.toml          # Dependencies
├── .env.example            # Environment template
└── azure.yaml              # Azure Developer CLI manifest
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | — | Your Azure OpenAI resource URL |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4.1` | Model deployment name |
| `AZURE_OPENAI_API_VERSION` | `2024-12-01-preview` | API version |
| `HEADLESS` | `true` | Set `false` to watch the browser |

## Troubleshooting

| Issue | Fix |
|---|---|
| `DefaultAzureCredential` fails | Run `az login` or check RBAC assignment |
| Copilot SDK timeout | Run `gh auth login` and verify Copilot access |
| `TimeoutError` on selectors | Page loading slowly — timeout is 30s, increase if needed |
| `GraphRecursionError` | Already set to 500 — raise further in `web/runners.py` |
| Rate limit (429) | Increase Azure OpenAI TPM quota via portal |

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph — Conditional Edges](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges)
- [GitHub Copilot SDK (Python)](https://github.com/nicolo-ribaudo/github-copilot-sdk-python)
- [Playwright Python API](https://playwright.dev/python/docs/api/class-page)
- [Azure OpenAI — Managed Identity](https://learn.microsoft.com/azure/ai-services/openai/how-to/managed-identity)
- [Azure Bicep](https://learn.microsoft.com/azure/azure-resource-manager/bicep/overview)
