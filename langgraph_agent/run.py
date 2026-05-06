"""Runner for the LangGraph browser automation agent.

This is the entry point. It:
1. Launches a Playwright browser
2. Builds the LangGraph agent graph
3. Runs the agent to completion
4. Prints structured results
"""

import asyncio
import json
import sys

from langchain_core.messages import HumanMessage
from playwright.async_api import async_playwright

from langgraph_agent.graph import build_graph, get_initial_messages
from langgraph_agent.tools import set_page
from shared.config import HEADLESS


async def run_agent():
    """Launch browser, run the LangGraph agent, return results."""
    print("=" * 60)
    print("[*] LangGraph Browser Automation Agent")
    print("=" * 60)
    print(f"   Headless: {HEADLESS}")
    print()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        page = await browser.new_page()
        set_page(page)

        print("[*] Browser launched")
        print("[*] Building agent graph...")

        graph = build_graph()

        print("[>] Starting agent execution\n")

        # Run the graph
        initial_state = {
            "messages": get_initial_messages() + [
                HumanMessage(content="Begin the task. Start by navigating to the site.")
            ],
            "results": [],
        }

        step = 0
        final_state = None

        async for event in graph.astream(
            initial_state,
            stream_mode="updates",
            config={"recursion_limit": 500},
        ):
            step += 1
            for node_name, update in event.items():
                if node_name == "agent" and "messages" in update:
                    msg = update["messages"][-1]
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            args_str = json.dumps(tc["args"], ensure_ascii=False)
                            print(f"   [>] Step {step}: {tc['name']}({args_str})")
                    elif hasattr(msg, "content") and msg.content and not msg.tool_calls:
                        # Final response
                        content_preview = msg.content[:200]
                        print(f"\n   [>] Agent response: {content_preview}...")
                        final_state = update
                elif node_name == "compact":
                    pass  # compaction logs itself

        await browser.close()

    print("\n" + "=" * 60)
    print("[DONE] Agent execution complete")
    print("=" * 60)

    if final_state and "messages" in final_state:
        last_msg = final_state["messages"][-1]
        if hasattr(last_msg, "content"):
            print("\n[=] FULL RESULTS:\n")
            print(last_msg.content)


def main():
    """Entry point."""
    asyncio.run(run_agent())


if __name__ == "__main__":
    main()
