"""The HTTP and WebSocket API, and the built UI.

Routes (all JSON; see ``server.engine`` for what the results contain):

- ``GET  /api/players``: checkpoints by run, the plain engines, and a
  default model (the newest checkpoint).
- ``POST /api/state``: a line's current position, legal hops, turns and
  result.
- ``POST /api/analyse``: search a line's position with a model for a number
  of simulations; returns the analysis when done.
- ``POST /api/move``: the hop a player (model, alpha-beta, random) plays.
- ``POST /api/evals``: White's chances at the start and after every turn
  of a line (the evaluation graph), optionally refined by search.
- ``POST /api/review``: every turn judged (best, mistake, ...) and each
  side's accuracy (``server.review``), by ``model`` or else the strongest
  checkpoint (``ModelPool.strongest``).
- ``WS   /api/live``: send ``{"id", "line", "model", "sims"}`` and receive a
  snapshot about ten times a second until ``done``. Sending a new request
  replaces the current one; ``{"stop": true}`` stops it.

A line is ``{"start": FEN or null, "actions": [hop, ...]}``. Bad input gives
HTTP 400 with the reason in ``detail``.

Searches share one GPU, so they run one at a time (``ModelPool.lock``) in
worker threads, keeping the event loop free. If ``ui/dist`` exists (``npm
run build`` in ``ui/``), it's served at ``/``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server.engine import (
    ENGINES,
    MAX_SIMS,
    Analysis,
    ModelPool,
    engine_move,
    hop,
    line_values,
    replay,
    state,
)
from server.review import REVIEW_SIMS, review

ROOT = Path(__file__).resolve().parent.parent
LIVE_INTERVAL = 0.1  # seconds of search between live snapshots
# searches per graph point: enough to see through exchanges (the network
# alone misjudges positions mid-capture), cheap for a whole game
GRAPH_SIMS = 100


class Line(BaseModel):
    """A starting position (FEN, or None for the standard start) and the
    hops played since."""

    start: str | None = None
    actions: list[int] = Field(default_factory=list)


class AnalyseRequest(BaseModel):
    """Search ``line``'s position with ``model`` for ``sims`` simulations."""

    line: Line
    model: str
    sims: int = Field(800, ge=0, le=MAX_SIMS)


class EvalsRequest(BaseModel):
    """Evaluate every turn of ``line`` with ``model``; ``sims`` > 0 also
    searches each position."""

    line: Line
    model: str
    sims: int = Field(GRAPH_SIMS, ge=0, le=MAX_SIMS)


class ReviewRequest(BaseModel):
    """Review ``line``'s turns; the judge is ``model`` or, if None, the
    strongest checkpoint."""

    line: Line
    model: str | None = None
    sims: int = Field(REVIEW_SIMS, ge=1, le=MAX_SIMS)


class MoveRequest(BaseModel):
    """Ask ``player`` (see ``server.engine.engine_move``) for its hop."""

    line: Line
    player: str


def _checked[T](call: Callable[[], T]) -> T:
    """Run ``call``, turning a ``ValueError`` into an HTTP 400."""
    try:
        return call()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def create_app(
    runs: Path = ROOT / 'runs',
    device: str | None = None,
    ui: Path | None = ROOT / 'ui' / 'dist',
) -> FastAPI:
    """The app, serving checkpoints from ``runs`` on ``device``."""
    app = FastAPI(title='Checkers Alpha')
    pool = ModelPool(runs, device)
    app.state.pool = pool

    def locked[T](call: Callable[[], T]) -> T:
        """Run ``call`` holding the GPU lock (in a worker thread)."""
        with pool.lock:
            return _checked(call)

    def newest() -> str | None:
        """The newest run's last checkpoint."""
        runs = pool.catalog()
        return runs[0]['checkpoints'][-1]['model'] if runs else None

    @app.get('/api/players')
    def players() -> dict[str, Any]:
        """Models by run (newest first), engines, a default model and the
        strongest one (the review judge)."""
        runs = pool.catalog()
        default = runs[0]['checkpoints'][-1]['model'] if runs else None
        return {
            'runs': runs,
            'engines': ENGINES,
            'default_model': default,
            'strongest_model': pool.strongest() or default,
            'device': pool.device,
        }

    @app.post('/api/state')
    def get_state(line: Line) -> dict[str, Any]:
        """The position at the end of ``line``."""
        return _checked(lambda: state(line.start, line.actions))

    @app.post('/api/analyse')
    def analyse(request: AnalyseRequest) -> dict[str, Any]:
        """Search the line's position; return the finished analysis."""

        def run() -> dict[str, Any]:
            """Build, search and summarise."""
            line = request.line
            pos = replay(line.start, line.actions)[-1]
            analysis = Analysis(pos, pool.evaluator(request.model))
            analysis.step(request.sims)
            return analysis.snapshot()

        return locked(run)

    @app.post('/api/evals')
    def evals(request: EvalsRequest) -> dict[str, Any]:
        """The evaluation graph of a line."""

        def run() -> dict[str, Any]:
            """Evaluate (and maybe search) every turn's position."""
            line = request.line
            points = line_values(
                line.start,
                line.actions,
                pool.evaluator(request.model),
                request.sims,
            )
            return {'points': points}

        return locked(run)

    @app.post('/api/review')
    def review_line(request: ReviewRequest) -> dict[str, Any]:
        """Judge every turn of a line."""
        judge = request.model or pool.strongest() or newest()
        if judge is None:
            raise HTTPException(400, 'no checkpoints to judge with')

        def run() -> dict[str, Any]:
            """Search every turn's positions and classify the moves."""
            line = request.line
            evaluator = pool.evaluator(judge)
            result = review(line.start, line.actions, evaluator, request.sims)
            return {'judge': judge, **result}

        return locked(run)

    @app.post('/api/move')
    def move(request: MoveRequest) -> dict[str, Any]:
        """The hop the requested player picks."""

        def run() -> dict[str, Any]:
            """Rebuild the position and ask the player."""
            line = request.line
            pos = replay(line.start, line.actions)[-1]
            action, analysis = engine_move(pos, request.player, pool)
            return {'move': hop(action), 'analysis': analysis}

        return locked(run)

    @app.websocket('/api/live')
    async def live(socket: WebSocket) -> None:
        """Stream snapshots of a growing search (see the module docstring)."""
        await socket.accept()
        requests: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        async def read() -> None:
            """Queue incoming requests until the client goes away."""
            with contextlib.suppress(WebSocketDisconnect):
                while True:
                    await requests.put(await socket.receive_json())
            await requests.put({'closed': True})

        reader = asyncio.create_task(read())
        try:
            request = await requests.get()
            while not request.get('closed'):
                request = await _serve_live(socket, request, requests, locked)
        finally:
            reader.cancel()

    if ui is not None and ui.is_dir():
        app.mount('/', StaticFiles(directory=ui, html=True), name='ui')
    return app


async def _serve_live(
    socket: WebSocket,
    request: dict[str, Any],
    requests: asyncio.Queue[dict[str, Any]],
    locked: Callable[[Callable[[], Any]], Any],
) -> dict[str, Any]:
    """Run one live request until done or replaced; return the next
    request (waiting for one if this one finished)."""
    ident = request.get('id')
    if request.get('stop'):
        return await requests.get()
    try:
        line = request.get('line') or {}
        target = min(int(request.get('sims') or MAX_SIMS), MAX_SIMS)

        pool: ModelPool = socket.app.state.pool

        def start() -> Analysis:
            """Rebuild the position and load the model."""
            pos = replay(line.get('start'), list(line.get('actions', [])))[-1]
            return Analysis(pos, pool.evaluator(request['model']))

        analysis: Analysis = await asyncio.to_thread(locked, start)
    except Exception as error:  # report anything wrong with the request
        detail = getattr(error, 'detail', str(error))
        await socket.send_json({'id': ident, 'error': detail})
        return await requests.get()
    while True:
        await asyncio.to_thread(
            locked, lambda: analysis.step(target, LIVE_INTERVAL)
        )
        done = analysis.sims >= target or analysis.pos.is_terminal
        snapshot = await asyncio.to_thread(locked, analysis.snapshot)
        await socket.send_json({'id': ident, 'done': done, **snapshot})
        if done:
            return await requests.get()
        if not requests.empty():
            return requests.get_nowait()
