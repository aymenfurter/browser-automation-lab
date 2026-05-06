"""Event emitter for agent instrumentation.

Both agents use this to emit step events (tool calls + screenshots)
which are broadcast to SSE clients via the web server.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playwright.async_api import Page


@dataclass
class StepEvent:
    """A single agent step (tool call + screenshot)."""
    run_id: str
    agent: str  # "langgraph" or "copilot-sdk"
    step_num: int
    tool_name: str
    tool_args: dict[str, Any]
    tool_result: str
    screenshot_path: str | None
    timestamp: float
    page_url: str = ""
    page_title: str = ""


@dataclass
class RunState:
    """Tracks state of a run for both agents."""
    run_id: str
    status: str = "running"  # running, completed, failed
    agents: dict[str, dict] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


# Global event queue for SSE broadcasting
_event_queues: list[asyncio.Queue] = []
_runs: dict[str, RunState] = {}


def subscribe() -> asyncio.Queue:
    """Subscribe to agent events. Returns a queue that receives StepEvents."""
    q: asyncio.Queue = asyncio.Queue()
    _event_queues.append(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    """Unsubscribe from events."""
    if q in _event_queues:
        _event_queues.remove(q)


def get_runs() -> dict[str, RunState]:
    return _runs


async def _broadcast(event: dict) -> None:
    """Send event to all subscribers."""
    for q in _event_queues:
        await q.put(event)


class AgentEmitter:
    """Instrumentation layer for an agent. Captures steps + screenshots."""

    def __init__(self, run_id: str, agent_name: str, page: Page):
        self.run_id = run_id
        self.agent = agent_name
        self.page = page
        self.step_num = 0
        self.steps: list[StepEvent] = []
        self._runs_dir = Path("runs") / run_id / "agents" / agent_name / "steps"
        self._runs_dir.mkdir(parents=True, exist_ok=True)

        # Register run
        if run_id not in _runs:
            _runs[run_id] = RunState(run_id=run_id)
        _runs[run_id].agents[agent_name] = {"status": "running", "steps": 0}

        # Note: we capture screenshots in emit_step after each tool call

    async def _on_navigation(self) -> None:
        """Capture screenshot on page navigation."""
        try:
            path = self._runs_dir / f"nav_{int(time.time()*1000)}.png"
            await self.page.screenshot(path=str(path), full_page=False)
        except Exception:
            pass  # page might be closed

    async def emit_step(self, tool_name: str, tool_args: dict, tool_result: str) -> None:
        """Record a step with screenshot."""
        self.step_num += 1
        screenshot_path: str | None = None

        try:
            filename = f"{self.step_num:04d}.png"
            full_path = self._runs_dir / filename
            await self.page.screenshot(path=str(full_path), full_page=False)
            screenshot_path = str(full_path)
        except Exception:
            pass

        page_url = self.page.url
        page_title = ""
        try:
            page_title = await self.page.title()
        except Exception:
            pass

        step = StepEvent(
            run_id=self.run_id,
            agent=self.agent,
            step_num=self.step_num,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result[:500],
            screenshot_path=screenshot_path,
            timestamp=time.time(),
            page_url=page_url,
            page_title=page_title,
        )
        self.steps.append(step)
        _runs[self.run_id].agents[self.agent]["steps"] = self.step_num

        # Save step metadata
        meta_path = self._runs_dir / f"{self.step_num:04d}.json"
        meta_path.write_text(json.dumps({
            "step": step.step_num,
            "tool_name": step.tool_name,
            "tool_args": step.tool_args,
            "tool_result": step.tool_result,
            "page_url": step.page_url,
            "page_title": step.page_title,
            "timestamp": step.timestamp,
            "screenshot": filename if screenshot_path else None,
        }, indent=2))

        # Broadcast to SSE clients
        await _broadcast({
            "type": "step",
            "run_id": self.run_id,
            "agent": self.agent,
            "step": self.step_num,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result": tool_result[:200],
            "page_url": page_url,
            "page_title": page_title,
            "screenshot_url": f"/api/runs/{self.run_id}/{self.agent}/screenshots/{self.step_num:04d}.png",
            "timestamp": step.timestamp,
        })

        # Also emit as a log entry
        args_short = ", ".join(f"{k}={str(v)[:40]}" for k, v in tool_args.items())
        result_short = tool_result[:100].replace("\n", " ")
        await _broadcast({
            "type": "log",
            "run_id": self.run_id,
            "agent": self.agent,
            "message": f"[step {self.step_num}] {tool_name}({args_short}) -> {result_short}",
            "timestamp": step.timestamp,
        })

    async def emit_status(self, status: str, message: str = "") -> None:
        """Emit a status change (compaction, complete, error)."""
        _runs[self.run_id].agents[self.agent]["status"] = status
        await _broadcast({
            "type": "status",
            "run_id": self.run_id,
            "agent": self.agent,
            "status": status,
            "message": message,
            "timestamp": time.time(),
        })
        # Also emit as log
        await _broadcast({
            "type": "log",
            "run_id": self.run_id,
            "agent": self.agent,
            "message": f"[STATUS] {status}: {message}" if message else f"[STATUS] {status}",
            "timestamp": time.time(),
        })

    async def emit_log(self, agent: str, message: str) -> None:
        """Emit a log message for the live logs panel."""
        await _broadcast({
            "type": "log",
            "run_id": self.run_id,
            "agent": agent,
            "message": message,
            "timestamp": time.time(),
        })

    async def complete(self, final_result: str = "") -> None:
        """Mark agent as complete."""
        _runs[self.run_id].agents[self.agent]["status"] = "completed"
        # Save final result
        result_path = self._runs_dir.parent / "result.md"
        result_path.write_text(final_result)
        await _broadcast({
            "type": "complete",
            "run_id": self.run_id,
            "agent": self.agent,
            "total_steps": self.step_num,
            "timestamp": time.time(),
        })

        # Check if all agents done
        run = _runs[self.run_id]
        if all(a.get("status") in ("completed", "failed") for a in run.agents.values()):
            run.status = "completed"
            await _broadcast({"type": "run_complete", "run_id": self.run_id, "timestamp": time.time()})
