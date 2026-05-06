"""Shared data models used by both agents."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel


class PresentationResult(BaseModel):
    """A single search result from SlideFinder."""

    title: str
    url: str


class SearchResult(BaseModel):
    """Results for a single search query."""

    query: str
    presentations: List[PresentationResult]
