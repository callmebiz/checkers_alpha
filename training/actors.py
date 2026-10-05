"""Self-play in CPU worker processes, served by one GPU inference server.

Why this design
---------------
Self-play is limited by single-threaded Python work: walking search trees
and generating moves. Worker processes spread that work over CPU cores.

If every worker had its own copy of the network on the GPU, adding workers
would soon make things slower. Each process has its own CUDA context, and on
Windows the GPU switches between contexts rather than running them side by
side. Each worker's batches would also shrink, since the games are divided
between them.

So the workers don't touch the GPU at all. The trainer process, which owns
the network and is otherwise idle during self-play, acts as the **inference
server**: it gathers the positions every worker is waiting on into one large
batch, runs the network once, and hands each worker its share of the
results. More workers mean more CPU cores searching and bigger GPU batches.

How a worker and the server talk
--------------------------------
Everything on the hot path goes through shared memory, which both processes
read and write directly. Pipes (``multiprocessing`` connections) are far
too slow on Windows for thousands of round trips per second, so they only
carry low-volume control messages.

Each worker has two *slots*, one per pipelined group of games (see
``training.self_play``). A slot holds up to ``capacity`` input positions (as
compact features, expanded to planes on the GPU) and room for their
outputs. A shared *mailbox* holds one state per slot::

    FREE ----(worker writes inputs, sets)----> REQUEST
    REQUEST --(server picks it up, sets)-----> BUSY
    BUSY ----(server writes outputs, sets)---> READY
    READY ---(worker reads outputs, sets)----> FREE

The server finds every ``REQUEST`` across all workers with one numpy
comparison, runs them as one batch, and marks them ``READY``. Each state is
written only after the data it announces, so a reader that sees the new
state also sees the data (x86 stores become visible in order).

**Dynamic batching**: while one batch runs on the GPU, new requests pile
up; when the GPU is free they all go as the next batch. The busier the
workers, the bigger the batches, so the fixed cost per batch (launch, sync)
is paid less often.

Control messages (pipes): ``("play", n)`` and ``("stop",)`` to a worker;
``("game",)`` per finished game (for the progress bar), ``("done", samples,
logged games, stats)`` and ``("error", traceback)`` back. Workers keep their
unfinished games between iterations, and the server always holds the latest
weights, so nothing needs reloading.

CPU placement (``pin_cpus``): everything waits on the server, so it gets a
core of its own. On hybrid CPUs such as the i9-12900K (8 hyperthreaded
performance cores, then 8 slower efficiency cores), workers go on the other
performance cores first, then the efficiency cores, and only then on the
performance cores' second threads. See ``cpu_plan``.

Failures: if a worker dies, the trainer raises an error naming it instead of
waiting forever. If the trainer goes away, workers notice and exit.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import fields
from multiprocessing.shared_memory import SharedMemory
from typing import Any, NoReturn

import numpy as np
import torch

try:
    import psutil
except ImportError:  # optional: without it, processes aren't pinned
    psutil = None

from core.encoding import NUM_BITBOARDS, features
from core.game import Position
from core.inference import (
    OUTPUT_SIZE,
    Evaluation,
    NetEvaluator,
    RawPending,
    to_evaluation,
)
from core.players import make_player
from training.arena import Contestant, play_games
from training.config import RunConfig
from training.replay_buffer import Samples
from training.self_play import SelfPlay, SelfPlayStats

# A pipe endpoint: ``Connection``, or ``PipeConnection`` on Windows (which
# isn't a subclass), so typed loosely.
type Conn = Any
# An evaluation request: (worker index, slot, number of positions)
type Request = tuple[int, int, int]

SLOTS = 2  # per worker: one per pipelined self-play group
FREE, REQUEST, BUSY, READY = 0, 1, 2, 3  # slot states in the mailbox
CONTROL_SECONDS = 0.002  # how often the server reads the control pipes
ALIVE_SECONDS = 1.0  # how often each side checks the other is still alive
IDLE_SLEEP = 0.00005  # pause when there is nothing to do, in seconds
QUEUED_BATCHES = 2  # batches the GPU may have queued at once
# Worker processes run single-threaded numeric libraries: many workers each
# starting a full thread pool would fight over the cores.
SINGLE_THREAD_ENV = {
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
}


def cpu_plan(workers: int) -> tuple[int, list[int]] | None:
    """The CPU for the server and one for each worker, or None if unknown.

    Windows numbers the hyperthreaded cores' threads first, in pairs, then
    any cores without hyperthreading (on hybrid Intel CPUs, the efficiency
    cores). The server takes the first core and leaves its second thread
    free; workers fill the other cores' first threads, then the
    single-thread cores, then the second threads, wrapping if needed.
    """
    if psutil is None:
        return None
    logical = psutil.cpu_count() or 0
    physical = psutil.cpu_count(logical=False) or 0
    if logical < 2 or physical < 1:
        return None
    paired = logical - physical  # cores with two threads
    first = list(range(0, 2 * paired, 2))
    second = list(range(1, 2 * paired, 2))
    single = list(range(2 * paired, logical))
    cores = first + single
    server = cores[0]
    order = [c for c in first[1:] + single + second[1:] if c != server]
    return server, [order[i % len(order)] for i in range(workers)]


def _pin(pid: int, cpu: int) -> None:
    """Restrict process ``pid`` to CPU ``cpu`` (best effort)."""
    if psutil is not None:
        try:
            psutil.Process(pid).cpu_affinity([cpu])
        except (psutil.Error, OSError, ValueError):
            pass


class Shared:
    """A block of shared memory, created here or attached to by name."""

    def __init__(self, size: int, name: str | None = None) -> None:
        """Create a ``size``-byte block (``name`` None) or attach to one."""
        if name is None:
            self.memory = SharedMemory(create=True, size=size)
        else:
            self.memory = SharedMemory(name=name)
        self.owner = name is None

    @property
    def name(self) -> str:
        """The block's name, for attaching from another process."""
        return self.memory.name

    def array(
        self, shape: tuple[int, ...], dtype: Any, offset: int = 0
    ) -> np.ndarray:
        """A numpy view of part of the block."""
        return np.ndarray(shape, dtype, buffer=self.memory.buf, offset=offset)

    def close(self) -> None:
        """Detach; the creating process also frees the block."""
        self.memory.close()
        if self.owner:
            self.memory.unlink()


class Slots(Shared):
    """A worker's input and output slots.

    Inputs are compact features (see ``core.encoding.features``): 40 bytes
    per position instead of 2 KB of planes. The server expands them on the
    GPU.
    """

    def __init__(self, capacity: int, name: str | None = None) -> None:
        """Room for ``capacity`` positions per slot."""
        self.capacity = capacity
        bits_bytes = capacity * NUM_BITBOARDS * 8
        scalar_bytes = capacity * 2 * 4
        out_bytes = capacity * OUTPUT_SIZE * 4
        per_slot = bits_bytes + scalar_bytes + out_bytes
        super().__init__(SLOTS * per_slot, name)
        self.bits = [
            self.array((capacity, NUM_BITBOARDS), np.int64, k * per_slot)
            for k in range(SLOTS)
        ]
        self.scalars = [
            self.array((capacity, 2), np.float32, k * per_slot + bits_bytes)
            for k in range(SLOTS)
        ]
        self.outputs = [
            self.array(
                (capacity, OUTPUT_SIZE),
                np.float32,
                k * per_slot + bits_bytes + scalar_bytes,
            )
            for k in range(SLOTS)
        ]

    def close(self) -> None:
        """Drop the views, then the block."""
        self.bits.clear()
        self.scalars.clear()
        self.outputs.clear()
        super().close()


class Mailbox(Shared):
    """Per worker and slot: state, position count and which model to use
    (0 = the current network, 1 = the previous checkpoint in an arena
    match); plus a stop flag."""

    def __init__(self, workers: int, name: str | None = None) -> None:
        """One ``(state, count, model)`` triple per worker and slot."""
        super().__init__((workers * SLOTS * 3 + 1) * 8, name)
        cells = self.array((workers, SLOTS, 3), np.int64)
        self.state = cells[:, :, 0]  # (workers, SLOTS)
        self.count = cells[:, :, 1]
        self.model = cells[:, :, 2]
        self.stop = self.array((1,), np.int64, workers * SLOTS * 3 * 8)

    def close(self) -> None:
        """Drop the views, then the block."""
        del self.state, self.count, self.model, self.stop
        super().close()


class _Stopped(Exception):
    """The trainer asked the workers to stop (or went away)."""


# -- worker side ---------------------------------------------------------------


class _RemotePending:
    """An evaluation requested from the server, not yet collected."""

    def __init__(
        self,
        client: RemoteEvaluator,
        slot: int,
        positions: Sequence[Position],
    ) -> None:
        """Remember which slot holds the request for ``positions``."""
        self.client = client
        self.slot = slot
        self.positions = positions

    def result(self) -> Evaluation:
        """Wait for the server's answer, then read the output slot."""
        return self.client.collect(self)


class RemoteEvaluator:
    """Worker-side evaluator that asks the inference server.

    Supports up to ``SLOTS`` requests in flight (which is how ``SelfPlay``
    uses it).
    """

    def __init__(
        self,
        index: int,
        slots: Slots,
        mailbox: Mailbox,
        model: int = 0,
        free: list[int] | None = None,
    ) -> None:
        """Worker ``index``'s view of its slots and the mailbox, asking
        for network ``model``. Evaluators of one worker share ``free``, its
        list of unused slots."""
        self.slots = slots
        self.state = mailbox.state[index]
        self.count = mailbox.count[index]
        self.model_cell = mailbox.model[index]
        self.stop = mailbox.stop
        self.model = model
        self.free = free if free is not None else list(range(SLOTS))

    def submit(self, positions: Sequence[Position]) -> _RemotePending:
        """Encode ``positions`` into a free slot and post the request."""
        n = len(positions)
        if n > self.slots.capacity:
            raise ValueError(f'{n} positions exceed slot capacity')
        slot = self.free.pop(0)
        bits, scalars = features(positions)
        self.slots.bits[slot][:n] = bits
        self.slots.scalars[slot][:n] = scalars
        self.count[slot] = n
        self.model_cell[slot] = self.model
        self.state[slot] = REQUEST  # last: the data above is now complete
        return _RemotePending(self, slot, positions)

    def collect(self, pending: _RemotePending) -> Evaluation:
        """Wait until the slot is ``READY``, then read its output."""
        slot = pending.slot
        spins = 0
        next_check = time.monotonic() + ALIVE_SECONDS
        while self.state[slot] != READY:
            spins += 1
            if spins > 100:
                time.sleep(0)  # give the core away while the server works
                if time.monotonic() > next_check:
                    parent = mp.parent_process()
                    if self.stop[0] or (parent and not parent.is_alive()):
                        raise _Stopped
                    next_check = time.monotonic() + ALIVE_SECONDS
        n = len(pending.positions)
        evaluation = to_evaluation(
            self.slots.outputs[slot][:n], pending.positions
        )
        self.state[slot] = FREE
        self.free.append(slot)
        return evaluation

    def evaluate(self, positions: Sequence[Position]) -> Evaluation:
        """Evaluate and wait, in slot-sized chunks if there are many."""
        size = self.slots.capacity
        if len(positions) <= size:
            return self.submit(positions).result()
        parts = [
            self.submit(positions[i : i + size]).result()
            for i in range(0, len(positions), size)
        ]
        return Evaluation(
            [p for part in parts for p in part.priors],
            np.concatenate([part.values for part in parts]),
            np.concatenate([part.wdl for part in parts]),
        )


def _play_arena(
    index: int,
    cfg: RunConfig,
    conn: Conn,
    slots: Slots,
    mailbox: Mailbox,
    free: list[int],
    settings: dict[str, Any],
    tasks: list[tuple[int, str, str, str]],
) -> list[dict[str, Any]]:
    """Play this worker's share of an arena: ``tasks`` are ``(id, opening
    FEN, Black, White)``. Each side is ``'net'`` (model 0, the current
    network), ``'prev'`` (model 1, the previous checkpoint), ``'model:K'``
    or ``'model:K:SIMS'`` (model K, with its own search budget; 1 means
    the raw network), or a player spec such as ``'ab:4'``. Network sides
    are named by ``settings['names'][spec]``."""
    contestants: dict[str, Contestant] = {}

    def contestant(spec: str) -> Contestant:
        """One contestant per spec, so their searches batch together."""
        if spec not in contestants:
            model = sims = None
            if spec in ('net', 'prev'):
                model, sims = (0 if spec == 'net' else 1), settings['sims']
            elif spec.startswith('model:'):
                parts = spec.split(':')
                model = int(parts[1])
                sims = int(parts[2]) if len(parts) > 2 else settings['sims']
            if model is not None and sims is not None:
                remote = RemoteEvaluator(index, slots, mailbox, model, free)
                contestants[spec] = Contestant.network(
                    settings['names'][spec], remote, sims
                )
            else:
                player = make_player(spec, seed=settings['seed'])
                contestants[spec] = Contestant.of(player)
        return contestants[spec]

    pairings = [
        (Position.from_fen(fen, cfg.rules), contestant(b), contestant(w))
        for _, fen, b, w in tasks
    ]
    games = play_games(
        pairings,
        settings['max_plies'],
        cfg.search,
        lambda: conn.send(('game',)),
    )
    for (task_id, *_), game in zip(tasks, games, strict=True):
        game['id'] = task_id
    return games


def _worker_main(
    index: int,
    config: dict[str, Any],
    conn: Conn,
    names: tuple[str, str],
    capacity: int,
) -> None:
    """Entry point of a worker process: play games when asked, until told
    to stop."""
    slots = mailbox = None
    try:
        cfg = RunConfig.from_dict(config)
        workers = cfg.self_play.workers
        # concurrent_games and saved_games are totals across all workers
        cfg.self_play.concurrent_games = -(
            -cfg.self_play.concurrent_games // workers
        )
        cfg.self_play.saved_games = cfg.self_play.saved_games // workers + (
            index < cfg.self_play.saved_games % workers
        )
        slots = Slots(capacity, name=names[0])
        mailbox = Mailbox(workers, name=names[1])
        # Each worker gets its own random stream, reproducible from the seed
        seed = None if cfg.seed is None else [cfg.seed, index]
        free = list(range(SLOTS))  # shared by all of this worker's evaluators
        self_play = SelfPlay(
            RemoteEvaluator(index, slots, mailbox, 0, free),
            cfg.self_play,
            cfg.search,
            cfg.rules,
            np.random.default_rng(seed),
        )
        while True:
            task = conn.recv()
            if task[0] == 'stop':
                return
            if task[0] == 'arena':
                games = _play_arena(
                    index, cfg, conn, slots, mailbox, free, task[1], task[2]
                )
                conn.send(('arena_done', games))
                continue
            records, stats = self_play.play(
                task[1], lambda _: conn.send(('game',))
            )
            samples = Samples.concat([r.samples for r in records])
            logged = [r.to_json() for r in records if r.logged]
            conn.send(('done', samples, logged, stats.to_dict()))
    except (KeyboardInterrupt, EOFError, BrokenPipeError, _Stopped):
        return  # Ctrl+C reaches every process, or the trainer went away
    except Exception:
        conn.send(('error', traceback.format_exc()))
    finally:
        for block in (slots, mailbox):
            if block is not None:
                block.close()


# -- server side ---------------------------------------------------------------


class ActorPool:
    """Self-play worker processes, and the server loop that feeds them."""

    def __init__(self, config: RunConfig, workers: int) -> None:
        """Start ``workers`` processes with their shared memory."""
        per_worker = -(-config.self_play.concurrent_games // workers)
        # a worker's biggest request is one of its two groups of games
        capacity = -(-per_worker // SLOTS)
        context = mp.get_context('spawn')
        self.mailbox = Mailbox(workers)
        self.slots = [Slots(capacity) for _ in range(workers)]
        self.conns: list[Conn] = []
        self.processes = []
        plan = cpu_plan(workers) if config.self_play.pin_cpus else None
        self.server_cpu = plan[0] if plan else None
        saved = {key: os.environ.get(key) for key in SINGLE_THREAD_ENV}
        os.environ.update(SINGLE_THREAD_ENV)  # inherited by the workers
        try:
            for i in range(workers):
                ours, theirs = context.Pipe()
                names = (self.slots[i].name, self.mailbox.name)
                process = context.Process(
                    target=_worker_main,
                    args=(i, config.to_dict(), theirs, names, capacity),
                    name=f'worker-{i}',
                    daemon=True,
                )
                process.start()
                if plan:
                    _pin(process.pid or 0, plan[1][i])
                theirs.close()
                self.conns.append(ours)
                self.processes.append(process)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def play(
        self,
        evaluator: NetEvaluator,
        num_games: int,
        on_game_end: Callable[[], None] | None = None,
    ) -> tuple[Samples, list[dict[str, Any]], SelfPlayStats]:
        """Have the workers play ``num_games``, serving their evaluations.

        Returns the samples, the logged games and the combined statistics.
        ``batches`` and ``evaluated`` in the statistics count the server's
        GPU batches, so their ratio is the real batch size.
        """
        start = time.perf_counter()
        workers = len(self.processes)
        shares = [
            num_games // workers + (i < num_games % workers)
            for i in range(workers)
        ]
        self._parts: list[Samples] = []
        self._logged: list[dict[str, Any]] = []
        self._stats = SelfPlayStats()
        busy = [i for i, share in enumerate(shares) if share > 0]
        for i in busy:
            self._send(i, ('play', shares[i]))
        with self._pinned(evaluator):
            batches, evaluated = self._serve([evaluator], busy, on_game_end)
        stats = self._stats
        stats.batches, stats.evaluated = batches, evaluated
        stats.seconds = time.perf_counter() - start
        return Samples.concat(self._parts), self._logged, stats

    def play_arena(
        self,
        evaluators: Sequence[NetEvaluator],
        settings: dict[str, Any],
        tasks: Sequence[tuple[int, str, str, str]],
        on_game_end: Callable[[], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Play arena games on the workers; return their records by id.

        ``tasks`` are ``(id, opening FEN, Black, White)`` with each side
        ``'net'`` (served by ``evaluators[0]``), ``'prev'``
        (``evaluators[1]``) or a player spec. ``settings`` holds ``names``
        (for ``'net'``/``'prev'``), ``sims``, ``max_plies`` and ``seed``.
        Alpha-beta and other plain players run on the workers' CPUs.
        """
        workers = len(self.processes)
        shares = [list(tasks[i::workers]) for i in range(workers)]
        self._arena_games: list[dict[str, Any]] = []
        busy = [i for i, share in enumerate(shares) if share]
        for i in busy:
            self._send(i, ('arena', settings, shares[i]))
        with self._pinned(evaluators[0]):
            self._serve(evaluators, busy, on_game_end)
        return sorted(self._arena_games, key=lambda g: g['id'])

    def _send(self, i: int, message: tuple[Any, ...]) -> None:
        """Send a control message to worker ``i``."""
        try:
            self.conns[i].send(message)
        except OSError:
            self._died(i)

    @contextmanager
    def _pinned(self, evaluator: NetEvaluator) -> Iterator[None]:
        """Pin this process to the server CPU while serving (if
        ``pin_cpus``); afterwards, training may use any core again.

        Only a GPU server is pinned: one on the CPU needs its threads for
        the network itself.
        """
        cpu = self.server_cpu if evaluator.device.type == 'cuda' else None
        me = (
            psutil.Process() if psutil is not None and cpu is not None else None
        )
        before = me.cpu_affinity() if me else None
        threads = torch.get_num_threads()
        if me and cpu is not None:
            me.cpu_affinity([cpu])
            torch.set_num_threads(1)  # no thread pool crammed onto one core
        try:
            yield
        finally:
            if me and before:
                me.cpu_affinity(before)
                torch.set_num_threads(threads)

    def _serve(
        self,
        evaluators: Sequence[NetEvaluator],
        busy: list[int],
        on_game_end: Callable[[], None] | None,
    ) -> tuple[int, int]:
        """Serve evaluations until every worker in ``busy`` reports done.

        Requests for different models (``evaluators[k]``) go in separate
        batches. Returns the number of batches and positions evaluated.
        """
        self._remaining = len(busy)
        self._on_game_end = on_game_end
        batches = evaluated = 0
        mailbox = self.mailbox
        state, count, model = mailbox.state, mailbox.count, mailbox.model
        # posted requests not yet sent to the GPU, per model
        waiting: dict[int, list[Request]] = {}
        # Batches on the GPU, oldest first. Up to QUEUED_BATCHES at once, so
        # the GPU starts the next batch the moment one finishes, instead of
        # idling while this process replies and launches.
        in_flight: deque[tuple[list[Request], RawPending]] = deque()
        next_control = next_alive = 0.0
        while self._remaining or waiting or in_flight:
            now = time.monotonic()
            if now >= next_control:
                self._read_control()
                next_control = now + CONTROL_SECONDS
            if now >= next_alive:
                self._check_alive()
                next_alive = now + ALIVE_SECONDS
            posted = np.argwhere(state == REQUEST)
            for w, s in posted:
                state[w, s] = BUSY
                request = (int(w), int(s), int(count[w, s]))
                waiting.setdefault(int(model[w, s]), []).append(request)
            while in_flight and in_flight[0][1].done():
                self._reply(*in_flight.popleft())
            if waiting and len(in_flight) < QUEUED_BATCHES:
                for k, requests in waiting.items():
                    pending = self._launch(evaluators[k], requests)
                    in_flight.append((requests, pending))
                    batches += 1
                    evaluated += sum(n for _, _, n in requests)
                waiting = {}
            elif not len(posted):
                if in_flight:
                    # nothing new to gather: wait for the oldest batch
                    self._reply(*in_flight.popleft())
                else:
                    time.sleep(IDLE_SLEEP)
        return batches, evaluated

    def _read_control(self) -> None:
        """Handle every waiting control message from the workers."""
        for i, conn in enumerate(self.conns):
            while self._poll(conn, i):
                message = self._receive(conn, i)
                kind = message[0]
                if kind == 'game':
                    if self._on_game_end is not None:
                        self._on_game_end()
                elif kind == 'done':
                    _, samples, games, worker_stats = message
                    self._parts.append(samples)
                    self._logged += games
                    _add_stats(self._stats, worker_stats)
                    self._remaining -= 1
                elif kind == 'arena_done':
                    self._arena_games += message[1]
                    self._remaining -= 1
                else:  # 'error'
                    self.close()
                    raise RuntimeError(f'worker-{i} failed:\n{message[1]}')

    def _launch(
        self, evaluator: NetEvaluator, requests: list[Request]
    ) -> RawPending:
        """Queue the network on every request at once (doesn't wait)."""
        bits = np.concatenate(
            [self.slots[i].bits[slot][:n] for i, slot, n in requests]
        )
        scalars = np.concatenate(
            [self.slots[i].scalars[slot][:n] for i, slot, n in requests]
        )
        return evaluator.submit_features(bits, scalars)

    def _reply(self, requests: list[Request], pending: RawPending) -> None:
        """Wait for a batch's output, then hand each worker its share."""
        out = pending.result()
        offset = 0
        for i, slot, n in requests:
            self.slots[i].outputs[slot][:n] = out[offset : offset + n]
            offset += n
            self.mailbox.state[i, slot] = READY  # last: outputs are complete

    def _poll(self, conn: Conn, i: int) -> bool:
        """Whether worker ``i`` has sent a message (errors if it's gone)."""
        try:
            return conn.poll()
        except (EOFError, OSError):
            self._died(i)

    def _receive(self, conn: Conn, i: int) -> tuple[Any, ...]:
        """Read one message, turning a vanished worker into an error."""
        try:
            return conn.recv()
        except (EOFError, OSError):
            self._died(i)

    def _died(self, i: int) -> NoReturn:
        """Shut everything down and report that worker ``i`` died."""
        process = self.processes[i]
        process.join(1)
        code = process.exitcode
        self.close()
        raise RuntimeError(f'worker-{i} died (exit code {code})') from None

    def _check_alive(self) -> None:
        """Raise if any worker process has died."""
        for i, process in enumerate(self.processes):
            if not process.is_alive():
                self._died(i)

    def close(self) -> None:
        """Stop the workers (terminating stragglers) and free the memory."""
        if not self.processes:
            return  # already closed
        self.mailbox.stop[0] = 1  # releases workers waiting on a reply
        for conn, process in zip(self.conns, self.processes, strict=True):
            if process.is_alive():
                try:
                    conn.send(('stop',))
                except (OSError, ValueError):
                    pass
        deadline = time.monotonic() + 10
        for process in self.processes:
            process.join(max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                process.terminate()
                process.join(1)
        for conn in self.conns:
            conn.close()
        for slots in self.slots:
            slots.close()
        self.mailbox.close()
        self.slots.clear()
        self.conns.clear()
        self.processes.clear()


def _add_stats(total: SelfPlayStats, part: dict[str, Any]) -> None:
    """Add a worker's statistics into ``total`` (except the timing)."""
    for f in fields(SelfPlayStats):
        if f.name != 'seconds':
            setattr(total, f.name, getattr(total, f.name) + part[f.name])
