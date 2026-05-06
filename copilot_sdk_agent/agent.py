"""Copilot SDK browser automation agent.

This module sets up a CopilotClient session with browser tools and sends
the task prompt. The Copilot runtime handles planning, tool invocation,
error recovery, and context compaction (infinite sessions) automatically.

Authentication: Uses local `gh` auth (default). Run `gh auth login` first.
"""

import asyncio

from copilot import CopilotClient
from copilot.generated.session_events import (
    AssistantMessageData,
    ExternalToolRequestedData,
    ExternalToolCompletedData,
    SessionCompactionStartData,
    SessionCompactionCompleteData,
    SessionIdleData,
)
from copilot.session import PermissionHandler
from playwright.async_api import async_playwright

from copilot_sdk_agent.tools import ALL_TOOLS, set_page
from shared.config import HEADLESS, SEARCH_QUERIES, SLIDEFINDER_URL

# The task prompt sent to the Copilot agent
TASK_PROMPT = f"""Go to {SLIDEFINDER_URL} and search for each of the following queries one by one.
For each query, extract the presentation titles and their URLs from the search results.

QUERIES:
{chr(10).join(f"{i+1}. {q}" for i, q in enumerate(SEARCH_QUERIES))}

WORKFLOW FOR EACH SEARCH:
1. Fill the search input (selector: input[placeholder="Search slides..."]) with the query
2. Press Enter on that input to submit
3. Wait for results to load (wait for selector: [aria-label="Search results"] a)
4. Extract links from: [aria-label="Search results"] a
5. Move to the next query (clear input and repeat)

IMPORTANT RULES:
- Navigate to the site first before searching
- After extracting results for one query, proceed immediately to the next
- When all searches are done, output all results in this format:
  ## Results
  ### [Query Name]
  - [Title](url)
  ### [Next Query]
  ...
- Do NOT skip any query
"""


async def run_agent():
    """Launch browser, create Copilot session, execute the task."""
    print("=" * 60)
    print("[*] GitHub Copilot SDK Browser Automation Agent")
    print("=" * 60)
    print(f"   Headless: {HEADLESS}")
    print(f"   Auth: local gh auth (default)")
    print(f"   Compaction: automatic (infinite sessions)")
    print()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        page = await browser.new_page()
        set_page(page)

        print("[*] Browser launched")
        print("[*] Connecting to Copilot CLI via SDK...")

        async with CopilotClient() as client:
            async with await client.create_session(
                model="gpt-4.1",
                on_permission_request=PermissionHandler.approve_all,
                tools=ALL_TOOLS,
            ) as session:
                done = asyncio.Event()
                final_content = ""
                tool_count = 0

                def on_event(event):
                    nonlocal final_content, tool_count
                    data = event.data
                    if isinstance(data, ExternalToolRequestedData):
                        tool_count += 1
                        print(f"   [>] [{tool_count}] {data.tool_name}")
                    elif isinstance(data, ExternalToolCompletedData):
                        pass
                    elif isinstance(data, SessionCompactionStartData):
                        print("\n[COMPACTION] Copilot is compacting context...")
                    elif isinstance(data, SessionCompactionCompleteData):
                        print("   [OK] Compaction complete\n")
                    elif isinstance(data, AssistantMessageData):
                        final_content = data.content
                    elif isinstance(data, SessionIdleData):
                        done.set()

                session.on(on_event)

                print("[>] Sending task prompt...\n")
                await session.send(TASK_PROMPT)
                await done.wait()

        await browser.close()

    print("\n" + "=" * 60)
    print("[DONE] Agent execution complete")
    print("=" * 60)

    if final_content:
        print("\n[=] FULL RESULTS:\n")
        print(final_content)

    return final_content
