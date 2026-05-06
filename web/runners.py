"""Agent runners for the web UI.

Wraps both agents to emit events via AgentEmitter during execution.
"""

import asyncio

from playwright.async_api import async_playwright

from shared.config import HEADLESS, SEARCH_QUERIES, SLIDEFINDER_URL
from web.events import AgentEmitter


async def run_both_agents(run_id: str) -> None:
    """Run LangGraph and Copilot SDK agents in parallel."""
    await asyncio.gather(
        _run_langgraph_agent(run_id),
        _run_copilot_sdk_agent(run_id),
        return_exceptions=True,
    )


async def _run_langgraph_agent(run_id: str) -> None:
    """Run the LangGraph agent with event emission."""
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.tools import tool
    from langchain_openai import AzureChatOpenAI
    from langgraph.graph import END, StateGraph
    from langgraph.prebuilt import ToolNode

    from langgraph_agent.compaction import compact_messages, should_compact
    from langgraph_agent.state import AgentState
    from langgraph_agent.graph import SYSTEM_PROMPT, _make_llm
    from shared.config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        emitter = AgentEmitter(run_id, "langgraph", page)

        await emitter.emit_status("running", "Starting LangGraph agent")

        # Create tools that emit events
        @tool
        async def navigate(url: str) -> str:
            """Navigate the browser to a URL."""
            await page.goto(url, wait_until="domcontentloaded")
            title = await page.title()
            result = f"Navigated to: {title} ({page.url})"
            await emitter.emit_step("navigate", {"url": url}, result)
            return result

        @tool
        async def click(selector: str) -> str:
            """Click an element on the page."""
            await page.locator(selector).click()
            result = f"Clicked '{selector}'"
            await emitter.emit_step("click", {"selector": selector}, result)
            return result

        @tool
        async def fill_input(selector: str, text: str) -> str:
            """Fill a text input field."""
            await page.locator(selector).fill(text)
            result = f"Filled '{selector}' with: {text}"
            await emitter.emit_step("fill_input", {"selector": selector, "text": text}, result)
            return result

        @tool
        async def press_key(selector: str, key: str) -> str:
            """Press a keyboard key on an element."""
            await page.locator(selector).press(key)
            result = f"Pressed '{key}' on '{selector}'"
            await emitter.emit_step("press_key", {"selector": selector, "key": key}, result)
            return result

        @tool
        async def wait_for_selector(selector: str, timeout_ms: int = 30000) -> str:
            """Wait for an element to appear."""
            try:
                await page.wait_for_selector(selector, timeout=timeout_ms)
                result = f"Element '{selector}' appeared"
            except Exception as e:
                result = f"Timeout waiting for '{selector}' after {timeout_ms}ms. Page may still be loading - try again or use a different selector."
            await emitter.emit_step("wait_for_selector", {"selector": selector}, result)
            return result

        @tool
        async def extract_links(selector: str) -> str:
            """Extract link text and href from elements."""
            links = await page.locator(selector).all()
            results = []
            for link in links:
                text = (await link.text_content() or "").strip()
                href = await link.get_attribute("href") or ""
                if text and href:
                    results.append(f"{text} | {href}")
            result = "\n".join(results) if results else "No links found"
            await emitter.emit_step("extract_links", {"selector": selector}, result)
            return result

        @tool
        async def get_page_text(selector: str) -> str:
            """Get text content of elements."""
            elements = await page.locator(selector).all_text_contents()
            result = "\n".join(elements) if elements else "No text found"
            await emitter.emit_step("get_page_text", {"selector": selector}, result)
            return result

        all_tools = [navigate, click, fill_input, press_key, wait_for_selector, extract_links, get_page_text]

        try:
            llm = _make_llm()
            llm_with_tools = llm.bind_tools(all_tools)

            async def agent_node(state: AgentState) -> dict:
                messages = state["messages"]
                for attempt in range(5):
                    try:
                        response = await llm_with_tools.ainvoke(messages)
                        return {"messages": [response]}
                    except Exception as e:
                        if "429" in str(e) or "too_many_requests" in str(e).lower():
                            wait = 2 ** attempt * 5
                            await emitter.emit_log("langgraph", f"Rate limited, retrying in {wait}s...")
                            await asyncio.sleep(wait)
                        else:
                            raise
                raise RuntimeError("Rate limit retries exhausted")

            def should_continue(state: AgentState) -> str:
                last_message = state["messages"][-1]
                if last_message.tool_calls:
                    return "tools"
                return "done"

            def after_tools(state: AgentState) -> str:
                return should_compact(state)

            graph = StateGraph(AgentState)
            graph.add_node("agent", agent_node)
            graph.add_node("tools", ToolNode(all_tools))
            graph.add_node("compact", compact_messages)
            graph.set_entry_point("agent")
            graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "done": END})
            graph.add_conditional_edges("tools", after_tools, {"compact": "compact", "continue": "agent"})
            graph.add_edge("compact", "agent")
            compiled = graph.compile()

            initial_state = {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content="Begin the task. Start by navigating to the site."),
                ],
                "results": [],
            }

            final_content = ""
            current_state = initial_state

            while True:
                final_state = await compiled.ainvoke(current_state, config={"recursion_limit": 500})

                # Count completed searches by counting extract_links tool results
                searches_done = emitter.step_num and sum(
                    1 for s in emitter.steps
                    if s.tool_name == "extract_links"
                )

                # Get final AI message
                for msg in reversed(final_state["messages"]):
                    if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None) and hasattr(msg, "response_metadata"):
                        final_content = msg.content
                        break

                if searches_done >= len(SEARCH_QUERIES):
                    break

                # Agent stopped early - nudge it
                remaining = len(SEARCH_QUERIES) - searches_done
                await emitter.emit_log("langgraph", f"Agent paused after {searches_done}/{len(SEARCH_QUERIES)} searches, sending continuation...")
                current_state = {
                    "messages": final_state["messages"] + [
                        HumanMessage(content=f"You stopped too early. You have only completed {searches_done} out of {len(SEARCH_QUERIES)} searches. Continue immediately with query {searches_done + 1}. Do NOT summarize results yet.")
                    ],
                    "results": final_state.get("results", []),
                }

            await emitter.complete(final_content)

        except Exception as e:
            await emitter.emit_status("failed", f"{type(e).__name__}: {e}")
        finally:
            await browser.close()


async def _run_copilot_sdk_agent(run_id: str) -> None:
    """Run the Copilot SDK agent with event emission."""
    from copilot import CopilotClient, define_tool
    from copilot.generated.session_events import (
        AssistantMessageData,
        ExternalToolRequestedData,
        SessionCompactionStartData,
        SessionCompactionCompleteData,
        SessionIdleData,
    )
    from copilot.session import PermissionHandler
    from pydantic import BaseModel, Field

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        emitter = AgentEmitter(run_id, "copilot-sdk", page)

        await emitter.emit_status("running", "Starting Copilot SDK agent")

        # Define tools with event emission
        class NavigateParams(BaseModel):
            url: str = Field(description="URL to navigate to")

        @define_tool(description="Navigate the browser to a URL.")
        async def navigate_browser(params: NavigateParams) -> str:
            await page.goto(params.url, wait_until="domcontentloaded")
            title = await page.title()
            result = f"Navigated to: {title} ({page.url})"
            await emitter.emit_step("navigate", {"url": params.url}, result)
            return result

        class ClickParams(BaseModel):
            selector: str = Field(description="CSS selector of element to click")

        @define_tool(description="Click an element on the page.")
        async def click_element(params: ClickParams) -> str:
            await page.locator(params.selector).click()
            result = f"Clicked '{params.selector}'"
            await emitter.emit_step("click", {"selector": params.selector}, result)
            return result

        class FillInputParams(BaseModel):
            selector: str = Field(description="CSS selector")
            text: str = Field(description="Text to type")

        @define_tool(description="Fill a text input field.")
        async def fill_input(params: FillInputParams) -> str:
            await page.locator(params.selector).fill(params.text)
            result = f"Filled '{params.selector}' with: {params.text}"
            await emitter.emit_step("fill_input", {"selector": params.selector, "text": params.text}, result)
            return result

        class PressKeyParams(BaseModel):
            selector: str = Field(description="CSS selector")
            key: str = Field(description="Key to press")

        @define_tool(description="Press a keyboard key on an element.")
        async def press_key(params: PressKeyParams) -> str:
            await page.locator(params.selector).press(params.key)
            result = f"Pressed '{params.key}' on '{params.selector}'"
            await emitter.emit_step("press_key", {"selector": params.selector, "key": params.key}, result)
            return result

        class WaitForSelectorParams(BaseModel):
            selector: str = Field(description="CSS selector to wait for")
            timeout_ms: int = Field(default=30000, description="Timeout in ms")

        @define_tool(description="Wait for an element to appear on the page.")
        async def wait_for_selector(params: WaitForSelectorParams) -> str:
            try:
                await page.wait_for_selector(params.selector, timeout=params.timeout_ms)
                result = f"Element '{params.selector}' appeared"
            except Exception as e:
                result = f"Timeout waiting for '{params.selector}' after {params.timeout_ms}ms. Page may still be loading - try again or use a different selector."
            await emitter.emit_step("wait_for_selector", {"selector": params.selector}, result)
            return result

        class ExtractLinksParams(BaseModel):
            selector: str = Field(description="CSS selector for links")

        @define_tool(description="Extract link text and href from elements.")
        async def extract_links(params: ExtractLinksParams) -> str:
            links = await page.locator(params.selector).all()
            results = []
            for link in links:
                text = (await link.text_content() or "").strip()
                href = await link.get_attribute("href") or ""
                if text and href:
                    results.append(f"{text} | {href}")
            result = "\n".join(results) if results else "No links found"
            await emitter.emit_step("extract_links", {"selector": params.selector}, result)
            return result

        class GetPageTextParams(BaseModel):
            selector: str = Field(description="CSS selector")

        @define_tool(description="Get text content of elements.")
        async def get_page_text(params: GetPageTextParams) -> str:
            elements = await page.locator(params.selector).all_text_contents()
            result = "\n".join(elements) if elements else "No text found"
            await emitter.emit_step("get_page_text", {"selector": params.selector}, result)
            return result

        all_tools = [navigate_browser, click_element, fill_input, press_key, wait_for_selector, extract_links, get_page_text]

        # Build task prompt
        task_prompt = f"""Go to {SLIDEFINDER_URL} and search for each of the following queries one by one.
For each query, extract the AI Overview text AND the presentation links from the search results.

FIRST: After navigating to the site, you MUST click the privacy/cookie consent "Accept" button
(selector: .disclaimer-btn-agree) before doing anything else.

QUERIES:
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(SEARCH_QUERIES))}

WORKFLOW FOR EACH SEARCH (you MUST follow this exact sequence for EACH query individually):
1. Fill the search input (selector: input[placeholder="Search slides..."]) with the query text
2. Press Enter on that input to submit the search
3. Wait for results to appear (wait for selector: [aria-label="Search results"] a)
4. Get the AI Overview text (selector: .ai-overview-content)
5. Extract links from: [aria-label="Search results"] a
6. ONLY THEN move to the next query

CRITICAL RULES:
- Do ONE search at a time. Never call fill_input multiple times in a row.
- After each fill_input you MUST press Enter and wait for results before the next search.
- You MUST capture the AI Overview (.ai-overview-content) for every query.
- Navigate to the site first. Click Accept on privacy notice. Do all {len(SEARCH_QUERIES)} searches sequentially.
"""

        try:
            async with CopilotClient() as client:
                async with await client.create_session(
                    model="gpt-4.1",
                    on_permission_request=PermissionHandler.approve_all,
                    tools=all_tools,
                ) as session:
                    done = asyncio.Event()
                    final_content = ""
                    searches_done = 0

                    def on_event(event):
                        nonlocal final_content, searches_done
                        data = event.data
                        if isinstance(data, SessionCompactionStartData):
                            asyncio.ensure_future(emitter.emit_status("compacting", "Copilot is compacting context..."))
                        elif isinstance(data, SessionCompactionCompleteData):
                            asyncio.ensure_future(emitter.emit_status("running", "Compaction complete"))
                        elif isinstance(data, ExternalToolRequestedData):
                            if data.tool_name == "extract_links":
                                searches_done += 1
                        elif isinstance(data, AssistantMessageData):
                            final_content = data.content
                        elif isinstance(data, SessionIdleData):
                            done.set()

                    session.on(on_event)
                    await session.send(task_prompt)

                    # Keep nudging the agent until all queries are done
                    while True:
                        await done.wait()
                        if searches_done >= len(SEARCH_QUERIES):
                            break
                        # Agent stopped early - nudge it to continue
                        done.clear()
                        remaining = len(SEARCH_QUERIES) - searches_done
                        await emitter.emit_log("copilot-sdk", f"Agent paused after {searches_done}/{len(SEARCH_QUERIES)} searches, sending continuation...")
                        await session.send(
                            f"You have only completed {searches_done} out of {len(SEARCH_QUERIES)} searches. "
                            f"Continue with the remaining {remaining} queries. Do not stop until all are done."
                        )

            await emitter.complete(final_content)

        except Exception as e:
            await emitter.emit_status("failed", f"{type(e).__name__}: {e}")
        finally:
            await browser.close()
