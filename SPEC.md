# Browser-Based Task Automation: Dual-Orchestrator Sample

## Objective

Build a sample project demonstrating browser-based task automation using **two orchestration approaches** side-by-side, with a focus on how each handles **long-horizon context management (compaction)**.

---

## Orchestrators

### 1. LangGraph + Playwright (Low-Level Orchestration)

- **Philosophy**: Full control over the agent loop via a directed state graph.
- **Stack**: `langgraph`, `langchain`, `playwright` (async API), LLM of choice (OpenAI/Anthropic).
- **Key Concepts**:
  - `StateGraph` with typed state (TypedDict/Pydantic) holding messages, page state, session metadata.
  - Nodes: Perception (AXTree extraction), Reasoning (LLM decision), Action (Playwright tool execution).
  - Conditional edges for dynamic routing (success → next step, error → retry/replan).
  - Checkpointer for fault tolerance and resume capability.
  - **Manual compaction** — developer must implement context window management explicitly.
- **When to use**: Complex enterprise flows, custom retry logic, fine-grained state management, when you need full control over how/when context is compacted.

### 2. GitHub Copilot SDK + Playwright (Platform-Embedded Orchestration)

- **Philosophy**: Embed a production-tested agentic runtime (the same engine behind GitHub Copilot CLI) into your application, with custom browser tools.
- **Stack**: `github-copilot-sdk` (Python), `playwright`, GitHub authentication or BYOK.
- **Status**: Public preview (announced Jan 2026, now multi-language).
- **Architecture**: SDK → JSON-RPC (stdio) → Copilot CLI (server mode) → LLM.
- **Key Concepts**:
  - `CopilotClient()` → `create_session(model=..., tools=[...])` — creates a managed agentic session.
  - **`@define_tool` decorator** with Pydantic params — register Playwright browser actions as tools with auto-generated JSON schemas.
  - **Permission handling** — `on_permission_request` callback controls tool execution approval.
  - **Event-driven** — subscribe to `AssistantMessageData`, `SessionIdleData`, streaming deltas.
  - **Infinite sessions** — automatic context window compaction handled by the runtime.
  - **Multi-model** — GPT-5, Claude Sonnet, etc.; use `list_models()` at runtime.
  - **Auth options** — GitHub OAuth, `GITHUB_TOKEN` env var, or BYOK (OpenAI/Anthropic/Azure keys).
- **When to use**: Apps in the GitHub ecosystem, teams wanting production-grade orchestration without building their own planner/tool loop, when you want compaction handled automatically.

---

## Sample Task

Both orchestrators will execute the **same task** to enable direct comparison:

> **Task**: "Go to ${SLIDEFINDER_URL} and perform a batch of 12 searches on Azure topics. For each search, extract the titles and links of the resulting presentations."

## Sample Task

Both orchestrators will execute the **same task** to enable direct comparison:

> **Task**: "Go to ${SLIDEFINDER_URL} and perform a batch of 12 searches on Azure topics. For each search, extract the titles and links of the resulting presentations."

**Search queries** (12 total — enough to trigger compaction):
1. Azure Kubernetes Service
2. Azure DevOps Pipelines
3. Azure Bicep Infrastructure as Code
4. Azure Container Apps
5. Azure Functions Serverless
6. Azure API Management
7. Azure Service Bus Messaging
8. Azure Monitor Observability
9. Azure Key Vault Secrets
10. Azure Front Door CDN
11. Azure Cosmos DB
12. Azure Virtual Network Security

This task exercises:
- Navigation to a specific URL
- Repeated form interactions (12 search cycles)
- Waiting for dynamic results to load after each search
- Data extraction (presentation titles and links) across many iterations
- **Context accumulation** — forces compaction to trigger (each search cycle adds ~500-1000 tokens of AXTree + tool calls)
- Structured output generation (aggregated results across all searches)

---

## Deployment: Bicep

The sample includes a Bicep template for deploying the agent as an Azure Container App, suitable for running scheduled batch searches in production.

### Infrastructure Components

```
┌─────────────────────────────────────────────────────┐
│  Resource Group: rg-browser-automation              │
│                                                     │
│  ┌─────────────────┐    ┌────────────────────────┐ │
│  │ Container App   │    │ Azure Key Vault        │ │
│  │ (agent runtime) │───▶│ (API keys, GH token)   │ │
│  └─────────────────┘    └────────────────────────┘ │
│         │                                           │
│         ▼                                           │
│  ┌─────────────────┐    ┌────────────────────────┐ │
│  │ Container App   │    │ Storage Account        │ │
│  │ Environment     │    │ (results output)       │ │
│  │ (with Chromium) │    └────────────────────────┘ │
│  └─────────────────┘                               │
└─────────────────────────────────────────────────────┘
```

### What gets deployed

| Resource                  | Purpose                                          |
|---------------------------|--------------------------------------------------|
| Container App Environment | Hosting environment for the agent container      |
| Container App             | Runs the Python agent with headless Chromium     |
| Key Vault                 | Stores `OPENAI_API_KEY`, `GITHUB_TOKEN`          |
| Storage Account           | Blob container for JSON result output            |
| Managed Identity          | Connects Container App to Key Vault + Storage    |
| Log Analytics Workspace   | Agent execution logs and telemetry               |

### Bicep structure

```
infra/
├── main.bicep              # Orchestrates all modules
├── main.bicepparam         # Parameter values
├── modules/
│   ├── container-app.bicep # Container App + Environment
│   ├── keyvault.bicep      # Key Vault + secrets
│   ├── storage.bicep       # Storage Account + blob container
│   └── monitoring.bicep    # Log Analytics workspace
```

### Key design decisions

- **Chromium in container**: Base image uses `mcr.microsoft.com/playwright/python:v1.40.0` which includes browsers pre-installed.
- **Managed Identity**: No credentials in env vars — Key Vault references used for secrets.
- **Scheduled execution**: Container App job with a cron trigger for periodic batch runs.
- **Results persistence**: Agent writes structured JSON to blob storage after each run.

---

## Deep Dive: Context Compaction

### The Problem

Browser agents accumulate state rapidly — each perception step adds an AXTree snapshot (potentially thousands of tokens), each action adds tool call/response pairs, and reasoning steps grow the message history. Without compaction, the context window overflows within a few iterations.

### Approach A: LangGraph — Manual Compaction via Summarize + Remove

In LangGraph, **you own the compaction strategy**. The approach: when messages exceed a threshold, summarize older messages into a single system message and permanently remove the originals from state using `RemoveMessage`.

```python
from langchain_core.messages import RemoveMessage, SystemMessage

def compaction_node(state: AgentState) -> dict:
    """Summarize old messages and remove originals from state."""
    messages = state["messages"]

    if len(messages) <= 10:
        return {}  # No compaction needed yet

    # Keep last 4 messages intact, summarize everything older
    old_messages = messages[:-4]
    summary = llm.invoke(
        [SystemMessage(content="Summarize this browser agent conversation concisely.")]
        + old_messages
    )

    # Remove old messages, insert summary
    delete_ops = [RemoveMessage(id=m.id) for m in old_messages]
    summary_msg = SystemMessage(content=f"[Conversation summary]: {summary.content}")

    return {"messages": delete_ops + [summary_msg]}

def should_compact(state: AgentState) -> str:
    """Conditional edge: route to compaction if messages exceed threshold."""
    if len(state["messages"]) > 10:
        return "compact"
    return "continue"
```

### Approach B: Copilot SDK — Automatic Compaction (Infinite Sessions)

The Copilot SDK handles compaction **transparently** via its "infinite sessions" feature. When the conversation nears the context window limit, the runtime automatically compacts older turns in the background.

```python
from copilot import CopilotClient
from copilot.session import PermissionHandler

async with CopilotClient() as client:
    async with await client.create_session(
        model="gpt-5",
        on_permission_request=PermissionHandler.approve_all,
        tools=[navigate_browser, extract_text, click_element],
        # Infinite sessions enabled by default — no config needed.
        # Optionally customize thresholds:
        # infinite_sessions=InfiniteSessionConfig(
        #     compact_threshold=0.8,  # Compact when 80% of window used
        # ),
    ) as session:
        await session.send(TASK_PROMPT)
        # ...agent runs indefinitely without context overflow
```

**What happens internally**:
- The runtime monitors token usage per turn.
- When approaching the limit, older turns are summarized and replaced with a compressed representation.
- The workspace directory persists full history on disk for recovery.
- The developer sees no difference — the agent continues reasoning with full historical context (compressed).

### Compaction Comparison

| Aspect                    | LangGraph (Manual)                          | Copilot SDK (Automatic)                   |
|---------------------------|---------------------------------------------|-------------------------------------------|
| Who implements it?        | Developer (summarize + RemoveMessage)       | Runtime (built-in)                        |
| Strategy control          | Full (custom threshold, summary prompt)     | Configurable thresholds only              |
| When it triggers          | Custom condition (e.g., msg count > 10)     | Automatic at ~80% context usage           |
| Visibility                | Full (you see every compaction step)        | Opaque (handled in background)            |
| Best for                  | Domain-specific compression needs           | General-purpose "just works" scenarios    |

---

## Project Structure

```
browser-automation-lab/
├── README.md                   # Setup instructions and usage
├── SPEC.md                     # This file
├── pyproject.toml              # Dependencies (uv/pip)
├── .env.example                # API key placeholders
├── shared/
│   ├── __init__.py
│   ├── models.py               # Shared Pydantic models (PresentationResult, etc.)
│   └── config.py               # Shared config (LLM provider, headless mode, search queries)
├── langgraph_agent/
│   ├── __init__.py
│   ├── graph.py                # StateGraph definition (nodes, edges, compaction)
│   ├── state.py                # AgentState TypedDict
│   ├── compaction.py           # Compaction strategy (summarize + remove)
│   ├── nodes/
│   │   ├── perceive.py         # AXTree/screenshot extraction
│   │   ├── reason.py           # LLM reasoning node
│   │   └── act.py              # Playwright action execution
│   ├── tools.py                # Playwright tools (navigate, click, extract, type)
│   └── run.py                  # Entrypoint: python -m langgraph_agent.run
├── copilot_sdk_agent/
│   ├── __init__.py
│   ├── tools.py                # Playwright browser tools (@define_tool)
│   ├── agent.py                # CopilotClient session setup with infinite sessions
│   └── run.py                  # Entrypoint: python -m copilot_sdk_agent.run
├── infra/
│   ├── main.bicep              # Top-level Bicep orchestration
│   ├── main.bicepparam         # Parameter values
│   └── modules/
│       ├── container-app.bicep # Container App + Environment
│       ├── keyvault.bicep      # Key Vault + secrets
│       ├── storage.bicep       # Storage Account + blob container
│       └── monitoring.bicep    # Log Analytics workspace
└── tests/
    ├── test_langgraph_agent.py
    └── test_copilot_sdk_agent.py
```

---

## Technical Requirements

### Python & Dependencies

| Package              | Version   | Purpose                            |
|----------------------|-----------|------------------------------------|
| python               | >=3.11    | Runtime                            |
| langgraph            | >=0.2     | Graph orchestration                |
| langchain            | >=0.3     | LLM abstractions, tool interfaces |
| langchain-openai     | >=0.2     | OpenAI chat model binding          |
| playwright           | >=1.40    | Browser automation                 |
| github-copilot-sdk   | latest    | Copilot agentic runtime (preview)  |
| pydantic             | >=2.0     | Data models                        |
| python-dotenv        | >=1.0     | Env var loading                    |

### Environment Variables

```
OPENAI_API_KEY=sk-...        # For LangGraph agent (or ANTHROPIC_API_KEY)
HEADLESS=true                # true for CI, false for debugging
GITHUB_TOKEN=ghp-...         # For Copilot SDK (or use `gh auth login`)
```

---

## Architecture: LangGraph Agent

```
┌────────────────────────────────────────────────────────────┐
│                       StateGraph                            │
│                                                            │
│  ┌──────────┐   ┌─────────┐   ┌──────┐   ┌───────────┐  │
│  │ Perceive │──▶│  Reason │──▶│ Act  │──▶│ Check Done│  │
│  └──────────┘   └─────────┘   └──────┘   └───────────┘  │
│       ▲              │                         │          │
│       │         [trim_messages                 ▼          │
│       │          before LLM]            [done? → END]    │
│       │                                        │          │
│       │         ┌────────────┐                 │          │
│       │         │  Compact   │◀── (threshold)──┘          │
│       │         │ (summarize │                            │
│       │         │ + remove)  │                            │
│       │         └────────────┘                            │
│       │              │                                    │
│       └──────────────┘                                    │
└────────────────────────────────────────────────────────────┘
```

**State Schema**:
```python
class AgentState(TypedDict):
    messages: list[BaseMessage]             # Conversation history (subject to compaction)
    page_content: str                       # Current AXTree snapshot
    current_url: str                        # Active page URL
    extracted_data: list[PresentationResult]  # Accumulated results
    step_count: int                         # Loop guard
    error: str | None                       # Last error for re-planning
```

**Flow**:
1. **Perceive** → Extract AXTree from current Playwright page.
2. **Reason** → `trim_messages` applied before LLM call. LLM decides next action.
3. **Act** → Execute the Playwright command.
4. **Check Done** → If complete, route to END. If messages exceed threshold, route to **Compact**.
5. **Compact** → Summarize old messages, `RemoveMessage` originals, insert summary. Loop back to Perceive.

---

## Architecture: GitHub Copilot SDK Agent

```
Your Application
       ↓
  SDK Client (CopilotClient)
       ↓ JSON-RPC (stdio)
  Copilot CLI (server mode)
       ↓                    ↑
  LLM (GPT-5, etc.)    [infinite sessions:
                         auto-compaction]
```

```python
import asyncio
from pydantic import BaseModel, Field
from copilot import CopilotClient, define_tool
from copilot.generated.session_events import AssistantMessageData, SessionIdleData
from copilot.session import PermissionHandler
from playwright.async_api import async_playwright

# Define browser tools using @define_tool + Pydantic
class NavigateParams(BaseModel):
    url: str = Field(description="URL to navigate to")

@define_tool(description="Navigate browser to URL, return page title")
async def navigate_browser(params: NavigateParams) -> str:
    await page.goto(params.url, wait_until="domcontentloaded")
    return f"Navigated to: {await page.title()} ({page.url})"

class TypeTextParams(BaseModel):
    selector: str = Field(description="CSS selector of input field")
    text: str = Field(description="Text to type")

@define_tool(description="Type text into an input field")
async def type_text(params: TypeTextParams) -> str:
    await page.locator(params.selector).fill(params.text)
    return f"Typed '{params.text}' into {params.selector}"

class ClickParams(BaseModel):
    selector: str = Field(description="CSS selector of element to click")

@define_tool(description="Click an element on the page")
async def click_element(params: ClickParams) -> str:
    await page.locator(params.selector).click()
    return f"Clicked {params.selector}"

class ExtractParams(BaseModel):
    selector: str = Field(description="CSS selector to extract text from")

@define_tool(description="Extract text from elements matching selector")
async def extract_text(params: ExtractParams) -> str:
    elements = await page.locator(params.selector).all_text_contents()
    return "\n".join(elements)

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        global page
        page = await browser.new_page()

        async with CopilotClient() as client:
            async with await client.create_session(
                model="gpt-5",
                on_permission_request=PermissionHandler.approve_all,
                tools=[navigate_browser, type_text, click_element, extract_text],
                # Infinite sessions = automatic compaction (enabled by default)
            ) as session:
                done = asyncio.Event()
                result_text = ""

                def on_event(event):
                    nonlocal result_text
                    match event.data:
                        case AssistantMessageData() as data:
                            result_text = data.content
                        case SessionIdleData():
                            done.set()

                session.on(on_event)
                await session.send(
                    "Go to ${SLIDEFINDER_URL} and search for "
                    "'Azure Kubernetes Service'. Extract the titles and "
                    "links of the resulting presentations."
                )
                await done.wait()

        await browser.close()
    return result_text

asyncio.run(main())
```

**Key differentiators**:
- **`@define_tool` decorator** — tools defined with Pydantic models, auto JSON schema generation.
- **No orchestration loop** — Copilot's engine handles planning, tool invocation, error recovery.
- **Compaction is invisible** — infinite sessions handle context window management automatically.
- **Permission system** — `on_permission_request` controls tool execution approval.

---

## Comparison Dimensions

| Dimension              | LangGraph + Playwright                  | GitHub Copilot SDK                       |
|------------------------|-----------------------------------------|------------------------------------------|
| Lines of code          | ~200-300                                | ~80-120                                  |
| Customizability        | Full (custom nodes/edges/compaction)    | Medium (custom tools, hooks)             |
| Compaction strategy    | Manual (trim, summarize, remove)        | Automatic (infinite sessions)            |
| Compaction visibility  | Full — you see/control every step       | Opaque — runtime handles it              |
| Error recovery         | Explicit (conditional edges)            | Built-in (production engine)             |
| Perception control     | Manual (AXTree or vision)               | Via registered tools                     |
| State inspection       | Full access to typed state              | Session-level access                     |
| Human-in-the-loop      | Native (`interrupt()`)                  | Session pause/resume                     |
| Learning curve         | Steep                                   | Moderate                                 |
| Production readiness   | High (checkpointing, durability)        | High (same engine as Copilot CLI)        |
| Auth / Infra           | BYO everything                          | GitHub auth or BYOK                      |
| MCP support            | Manual integration                      | Native                                   |

---

## Implementation Plan

1. **Scaffold project** — `pyproject.toml`, directory structure, shared models.
2. **Implement shared layer** — `PresentationResult` model, config with 12 search queries, env setup.
3. **Build LangGraph agent** — State, nodes, graph wiring, **compaction node** with summarize + remove.
4. **Build Copilot SDK agent** — Tool definitions, session setup with infinite sessions, result parsing.
5. **Build Bicep infra** — Container App, Key Vault, Storage, Managed Identity, Log Analytics.
6. **Add test harness** — Verify both agents produce equivalent results.
7. **Write README** — Setup instructions, run commands, deploy commands, compaction behavior comparison.

---

## Success Criteria

- Both agents complete 12 SlideFinder searches end-to-end in a single session.
- Results are structured as `list[PresentationResult]` grouped by search query.
- LangGraph agent demonstrates **explicit compaction** — visible summarization logs when message threshold exceeded (~after search 3-4).
- Copilot SDK agent demonstrates **automatic compaction** — infinite sessions handle context transparently across all 12 searches.
- Bicep templates deploy successfully with `az deployment group create`.
- Container App runs the agent headlessly and writes results to blob storage.
- README clearly documents local run, Azure deploy, and compaction differences.

---

## Notes & Considerations

- **Copilot SDK is in public preview** — API surface may change. Pin version in `pyproject.toml`.
- The Copilot SDK requires GitHub authentication (`gh auth login` or `GITHUB_TOKEN`). Document this clearly.
- For the LangGraph compaction demo, intentionally log when compaction triggers so it's visible in output.
- The LangGraph agent should use a lower compaction threshold (e.g., 10 messages) to ensure compaction fires during the demo task, even though the task itself is short.
- Both agents share the same Playwright browser instance pattern to keep the comparison fair.
