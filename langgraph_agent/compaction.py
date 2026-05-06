"""Compaction node for the LangGraph agent.

When messages exceed a threshold, this node summarizes older messages into a
single system message and removes the originals from state using RemoveMessage.
This keeps the context window manageable across 12+ search iterations.
"""

from __future__ import annotations

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_core.messages import RemoveMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

from langgraph_agent.state import AgentState
from shared.config import (
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
)

# Compaction triggers when message count exceeds this threshold.
# Compaction triggers when message count exceeds this threshold.
# Each query generates ~10 messages (AI calls + tool responses).
# We want compaction roughly every 8-10 queries (not every 2).
COMPACTION_THRESHOLD = 100


def should_compact(state: AgentState) -> str:
    """Conditional edge: route to 'compact' or 'continue' based on message count."""
    if len(state["messages"]) > COMPACTION_THRESHOLD:
        return "compact"
    return "continue"


async def compact_messages(state: AgentState) -> dict:
    """Summarize older messages and remove them from state.

    Preserves the original system prompt (first message) and last 6 messages.
    Only removes complete message groups (assistant+tool pairs kept together).
    """
    messages = state["messages"]
    keep_recent = 20

    if len(messages) <= keep_recent + 1:
        return {}

    # Keep: first message (system prompt) + last N messages
    # Find a safe cut point - never split a tool_calls/tool_response pair
    candidate_end = len(messages) - keep_recent

    # Walk backwards from candidate_end to find a safe cut point
    # Safe = the message at cut point is NOT a ToolMessage
    from langchain_core.messages import AIMessage, ToolMessage

    while candidate_end > 1:
        msg = messages[candidate_end]
        if isinstance(msg, ToolMessage):
            candidate_end -= 1
        elif isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            candidate_end -= 1
        else:
            break

    old_messages = messages[1:candidate_end]  # skip system prompt at index 0

    if len(old_messages) < 4:
        return {}

    print(f"\n[COMPACTION] Summarizing {len(old_messages)} old messages...")

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        temperature=0,
    )
    summary_response = await llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "Summarize the following browser automation conversation. "
                    "List EXACTLY which search queries have been completed and their results. "
                    "Format as a bullet list of completed queries with their found links. "
                    "This summary will be used to continue the remaining searches."
                )
            ),
        ]
        + old_messages
    )

    # Remove old messages (NOT the system prompt), insert summary after system prompt
    delete_ops = [RemoveMessage(id=m.id) for m in old_messages]
    summary_msg = SystemMessage(
        content=f"[PROGRESS SO FAR]:\n{summary_response.content}"
    )

    remaining = len(messages) - len(old_messages)
    print(f"   Compacted into summary ({len(summary_response.content)} chars)")
    print(f"   Messages: {len(messages)} -> {remaining + 1}")

    return {"messages": delete_ops + [summary_msg]}
