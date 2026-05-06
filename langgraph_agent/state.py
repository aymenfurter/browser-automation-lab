"""Agent state definition for the LangGraph browser agent."""

from __future__ import annotations

from typing import Annotated, List

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from shared.models import SearchResult


class AgentState(TypedDict):
    """State that flows through the LangGraph agent.

    - messages: The conversation history (LLM + tool calls). Subject to compaction.
    - results: Accumulated search results across all queries.
    """

    messages: Annotated[List[BaseMessage], add_messages]
    results: List[SearchResult]
