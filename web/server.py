"""FastAPI web server for the Browser Automation Lab.

Provides:
- SSE endpoint for live event streaming
- REST endpoints for run control and history
- Static file serving for the dashboard
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from web.events import AgentEmitter, subscribe, unsubscribe, get_runs

app = FastAPI(title="Browser Automation Lab")

# Serve screenshots from runs directory
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)


@app.get("/")
async def index():
    """Serve the dashboard."""
    return FileResponse("web/dashboard.html")


@app.post("/api/run")
async def start_run(request: Request):
    """Start a new parallel run of both agents."""
    run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:6]}"

    # Import here to avoid circular imports
    from web.runners import run_both_agents

    async def _run_with_logging():
        import traceback as tb
        try:
            await run_both_agents(run_id)
        except Exception as e:
            print(f"[ERROR] run_both_agents failed: {e}")
            tb.print_exc()

    asyncio.create_task(_run_with_logging())

    return {"run_id": run_id, "status": "started"}


@app.get("/api/events")
async def sse_events(request: Request):
    """Server-Sent Events endpoint for live agent updates."""
    queue = subscribe()

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield f": keepalive\n\n"
        finally:
            unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/runs")
async def list_runs():
    """List all runs (current and past)."""
    runs = get_runs()
    result = []
    for run_id, run in sorted(runs.items(), key=lambda x: x[1].created_at, reverse=True):
        result.append({
            "run_id": run.run_id,
            "status": run.status,
            "created_at": run.created_at,
            "agents": run.agents,
        })
    return result


@app.get("/api/runs/{run_id}/{agent}/steps")
async def get_steps(run_id: str, agent: str):
    """Get all steps for a specific agent in a run."""
    steps_dir = RUNS_DIR / run_id / "agents" / agent / "steps"
    if not steps_dir.exists():
        return []
    steps = []
    for json_file in sorted(steps_dir.glob("*.json")):
        steps.append(json.loads(json_file.read_text()))
    return steps


@app.get("/api/runs/{run_id}/{agent}/screenshots/{filename}")
async def get_screenshot(run_id: str, agent: str, filename: str):
    """Serve a screenshot image."""
    path = RUNS_DIR / run_id / "agents" / agent / "steps" / filename
    if not path.exists():
        return HTMLResponse("Not found", status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/runs/{run_id}/{agent}/result")
async def get_result(run_id: str, agent: str):
    """Get final result for an agent."""
    result_path = RUNS_DIR / run_id / "agents" / agent / "result.md"
    if not result_path.exists():
        return {"result": None}
    return {"result": result_path.read_text()}
