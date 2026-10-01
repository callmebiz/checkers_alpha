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
network planes) and room for their outputs. A shared *mailbox* holds one
state per slot::

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
from collections.abc import Callable, Sequence
from dataclasses import fields
from multiprocessing.shared_memory import SharedMemory
from typing import Any, NoReturn

import numpy as np

try:
    import psutil
except ImportError:  # optional: without it, processes aren't pinned
    psutil = None

from core.encoding import NUM_PLANES, encode
from core.game import Position
from core.inference import (
    OUTPUT_SIZE,
    Evaluation,
    NetEvaluator,
    RawPending,
    to_evaluation,
)
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
    """A worker's input and output slots."""

    def __init__(self, capacity: int, name: str | None = None) -> None:
        """Room for ``capacity`` positions per slot."""
        self.capacity = capacity
        in_bytes = capacity * NUM_PLANES * 64 * 4
        out_bytes = capacity * OUTPUT_SIZE * 4
        super().__init__(SLOTS * (in_bytes + out_bytes), name)
        self.inputs = [
            self.array(
                (capacity, NUM_PLANES, 8, 8),
                np.float32,
                k * (in_bytes + out_bytes),
            )
            for k in range(SLOTS)
        ]
        self.outputs = [
            self.array(
                (capacity, OUTPUT_SIZE),
                np.float32,
                k * (in_bytes + out_bytes) + in_bytes,
            )
            for k in range(SLOTS)
        ]

    def close(self) -> None:
        """Drop the views, then the block."""
        self.inputs.clear()
        self.outputs.clear()
        super().close()


class Mailbox(Shared):
    """Slot states and position counts for all workers, plus a stop flag."""

    def __init__(self, workers: int, name: str | None = None) -> None:
        """One ``(state, count)`` pair per worker and slot."""
        super().__init__((workers * SLOTS * 2 + 1) * 8, name)
        cells = self.array((workers, SLOTS, 2), np.int64)
        self.state = cells[:, :, 0]  # (workers, SLOTS)
        self.count = cells[:, :, 1]
        self.stop = self.array((1,), np.int64, workers * SLOTS * 2 * 8)

    def close(self) -> None:
        """Drop the views, then the block."""
        del self.state, self.count, self.stop
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

    def __init__(self, index: int, slots: Slots, mailbox: Mailbox) -> None:
        """Worker ``index``'s view of its slots and the mailbox."""
        self.slots = slots
        self.state = mailbox.state[index]
        self.count = mailbox.count[index]
        self.stop = mailbox.stop
        self.free = list(range(SLOTS))

    def submit(self, positions: Sequence[Position]) -> _RemotePending:
        """Encode ``positions`` into a free slot and post the request."""
        n = len(positions)
        if n > self.slots.capacity:
            raise ValueError(f'{n} positions exceed slot capacity')
        slot = self.free.pop(0)
        self.slots.inputs[slot][:n] = encode(positions)
        self.count[slot] = n
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
        """Evaluate and wait for the result."""
        return self.submit(positions).result()


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
        self_play = SelfPlay(
            RemoteEvaluator(index, slots, mailbox),
            cfg.self_play,
            cfg.search,
            cfg.rules,
            np.random.default_rng(seed),
        )
        while True:
            task = conn.recv()
            if task[0] == 'stop':
                return
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
        GPU batches, so their ratio is the real batch size. While serving,
        this process is pinned to its CPU (if ``pin_cpus``); afterwards,
        training and the arena may use any core again.
        """
        cpu = self.server_cpu
        me = (
            psutil.Process() if psutil is not None and cpu is not None else None
        )
        before = me.cpu_affinity() if me else None
        if me and cpu is not None:
            me.cpu_affinity([cpu])
        try:
            return self._serve(evaluator, num_games, on_game_end)
        finally:
            if me and before:
                me.cpu_affinity(before)

    def _serve(
        self,
        evaluator: NetEvaluator,
        num_games: int,
        on_game_end: Callable[[], None] | None,
    ) -> tuple[Samples, list[dict[str, Any]], SelfPlayStats]:
        """The server loop behind ``play``."""
        start = time.perf_counter()
        workers = len(self.processes)
        shares = [
            num_games // workers + (i < num_games % workers)
            for i in range(workers)
        ]
        for i, (conn, share) in enumerate(zip(self.conns, shares, strict=True)):
            try:
                conn.send(('play', share))
            except OSError:
                self._died(i)

        self._parts: list[Samples] = []
        self._logged: list[dict[str, Any]] = []
        self._stats = SelfPlayStats()
        self._remaining = sum(share > 0 for share in shares)
        self._on_game_end = on_game_end
        batches = evaluated = 0
        state, count = self.mailbox.state, self.mailbox.count
        waiting: list[Request] = []  # posted, not yet sent to the GPU
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
                waiting.append((int(w), int(s), int(count[w, s])))
            while in_flight and in_flight[0][1].done():
                self._reply(*in_flight.popleft())
            if waiting and len(in_flight) < QUEUED_BATCHES:
                in_flight.append((waiting, self._launch(evaluator, waiting)))
                batches += 1
                evaluated += sum(n for _, _, n in waiting)
                waiting = []
            elif not len(posted):
                if in_flight:
                    # nothing new to gather: wait for the oldest batch
                    self._reply(*in_flight.popleft())
                else:
                    time.sleep(IDLE_SLEEP)
        stats = self._stats
        stats.batches, stats.evaluated = batches, evaluated
        stats.seconds = time.perf_counter() - start
        return Samples.concat(self._parts), self._logged, stats

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
                else:  # 'error'
                    self.close()
                    raise RuntimeError(f'worker-{i} failed:\n{message[1]}')

    def _launch(
        self, evaluator: NetEvaluator, requests: list[Request]
    ) -> RawPending:
        """Queue the network on every request at once (doesn't wait)."""
        x = np.concatenate(
            [self.slots[i].inputs[slot][:n] for i, slot, n in requests]
        )
        return evaluator.submit_planes(x)

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
