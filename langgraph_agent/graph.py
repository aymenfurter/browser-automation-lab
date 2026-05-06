"""LangGraph state graph definition.

This is the core orchestration: a ReAct-style agent loop where the LLM
decides which browser tools to call, with a compaction node that triggers
when messages grow too large.

Graph flow:
  agent → (tool calls?) → tools → (compact?) → agent → ... → END
"""

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from langgraph_agent.compaction import compact_messages, should_compact
from langgraph_agent.state import AgentState
from langgraph_agent import tools as browser_tools
from shared.config import (
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    SLIDEFINDER_URL,
)


def _make_llm() -> AzureChatOpenAI:
    """Create AzureChatOpenAI instance with token-based auth."""
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        temperature=0,
    )


# ─── Wrap browser functions as LangChain tools ──────────────────────────────


@tool
async def navigate(url: str) -> str:
    """Navigate the browser to a URL. Returns the page title after load."""
    return await browser_tools.navigate(url)


@tool
async def click(selector: str) -> str:
    """Click an element on the page identified by CSS selector."""
    return await browser_tools.click(selector)


@tool
async def fill_input(selector: str, text: str) -> str:
    """Fill a text input field identified by CSS selector with the given text."""
    return await browser_tools.fill_input(selector, text)


@tool
async def press_key(selector: str, key: str) -> str:
    """Press a keyboard key on the element matching CSS selector."""
    return await browser_tools.press_key(selector, key)


@tool
async def wait_for_selector(selector: str, timeout_ms: int = 10000) -> str:
    """Wait for an element matching CSS selector to appear on the page."""
    return await browser_tools.wait_for_selector(selector, timeout_ms)


@tool
async def extract_links(selector: str) -> str:
    """Extract all link text and href from <a> elements matching CSS selector.
    Returns one link per line in format: title | url
    """
    return await browser_tools.extract_links(selector)


@tool
async def get_page_text(selector: str) -> str:
    """Get the text content of all elements matching a CSS selector."""
    return await browser_tools.get_page_text(selector)


ALL_TOOLS = [navigate, click, fill_input, press_key, wait_for_selector, extract_links, get_page_text]


# ─── System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a browser automation agent. You control a real browser via tools.

YOUR TASK:
Go to {SLIDEFINDER_URL} and search for each of the following queries one by one.
For each query, extract the AI Overview text AND the presentation links from the search results.

FIRST: After navigating to the site, you MUST click the privacy/cookie consent "Accept" button
(selector: .disclaimer-btn-agree) before doing anything else.

QUERIES:
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
13. Azure AI Search
14. Azure Logic Apps Integration
15. Azure Event Grid
16. Azure Static Web Apps
17. Azure Machine Learning
18. Azure Defender for Cloud
19. Azure Load Testing
20. GitHub Copilot for Azure

WORKFLOW FOR EACH SEARCH:
1. Fill the search input (selector: input[placeholder="Search slides..."]) with the query
2. Press Enter on that input to submit
3. Wait for results to load (wait for selector: [aria-label="Search results"] a)
4. Get the AI Overview text (selector: .ai-overview-content)
5. Extract links from: [aria-label="Search results"] a
6. Move to the next query (clear input and repeat)

IMPORTANT RULES:
- Navigate to the site first before searching
- Click the Accept button on the privacy notice FIRST
- You MUST capture the AI Overview (.ai-overview-content) for every query
- After extracting results for one query, proceed immediately to the next
- When all 20 searches are done, respond with DONE and list all results in this exact format:
  ## Results
  ### [Query Name]
  **AI Overview:** [overview text]
  - [Title](url)
  - [Title](url)
  ### [Next Query]
  ...
- Do NOT skip any query
- Do NOT add fallback logic - if something fails, let it fail
"""


# ─── Graph construction ─────────────────────────────────────────────────────


def build_graph():
    """Build and compile the LangGraph agent with compaction."""

    llm = _make_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    async def agent_node(state: AgentState) -> dict:
        """The LLM reasoning node. Decides what tool to call next."""
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        """After agent responds, check if it wants to use tools or is done."""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return "done"

    def after_tools(state: AgentState) -> str:
        """After tools execute, check if compaction is needed."""
        return should_compact(state)

    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("compact", compact_messages)

    # Set entry point
    graph.set_entry_point("agent")

    # Agent decides: call tools or finish
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "done": END,
    })

    # After tools: check compaction threshold, then back to agent
    graph.add_conditional_edges("tools", after_tools, {
        "compact": "compact",
        "continue": "agent",
    })

    # After compaction: back to agent
    graph.add_edge("compact", "agent")

    return graph.compile()


def get_initial_messages() -> list:
    """Get the initial messages to kick off the agent."""
    return [SystemMessage(content=SYSTEM_PROMPT)]
