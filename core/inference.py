"""Evaluators: score positions with move priors and values for search.

The search (``core.mcts``) only talks to this interface, so the neural network
can be swapped for a stand-in: uniform priors for tests, or random playouts as a
network-free baseline.

Batching matters. A network call costs about the same for 1 position as for 256
on a GPU, so callers collect many positions and evaluate them together (see
``core.mcts.run_search``).

Pipelining
----------
``evaluate`` waits for the answer. ``submit(evaluator, positions)`` instead
returns a ``Pending`` handle: for a network on the GPU, the work is queued and
``submit`` returns at once, so the CPU can prepare the next batch while the
GPU computes. ``Pending.result()`` then waits (if needed) and returns the
``Evaluation``. Evaluators without a ``submit`` method simply compute
immediately. Self-play uses this to keep both CPU and GPU busy.
"""

from __future__ import annotations

import io
import itertools
import logging
import random
import time
import warnings
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, redirect_stdout
from typing import Any, NamedTuple, Protocol

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from core.encoding import (
    NUM_PLANES,
    PlaneExpander,
    canonical_actions,
    features,
)
from core.game import Position
from core.model import CheckersNet

type FloatArray = npt.NDArray[np.floating]


class Evaluation(NamedTuple):
    """Evaluation of N non-terminal positions.

    Everything is from each position's side to move.
    """

    # per position: probabilities aligned with pos.legal_actions()
    priors: list[FloatArray]
    values: FloatArray  # (N,) expected outcome in [-1, 1]
    wdl: FloatArray  # (N, 3) win / draw / loss probabilities


class Evaluator(Protocol):
    def evaluate(self, positions: Sequence[Position]) -> Evaluation:
        """Score ``positions`` (all non-terminal) for their sides to move."""
        ...


class Pending(Protocol):
    def result(self) -> Evaluation:
        """Wait if needed, then return the evaluation."""
        ...


class _Ready:
    """A ``Pending`` whose result is already known."""

    def __init__(self, evaluation: Evaluation) -> None:
        """Wrap an evaluation that is already finished."""
        self.evaluation = evaluation

    def result(self) -> Evaluation:
        """Return the stored evaluation."""
        return self.evaluation


def submit(evaluator: Evaluator, positions: Sequence[Position]) -> Pending:
    """Start evaluating ``positions``; collect with ``.result()``."""
    start = getattr(evaluator, 'submit', None)
    if start is not None:
        return start(positions)
    return _Ready(evaluator.evaluate(positions))


def legal_softmax(
    logits: npt.NDArray[np.float32], positions: Sequence[Position]
) -> list[FloatArray]:
    """Per position, the softmax of its legal moves' logits only.

    Done for the whole batch at once: the legal logits are gathered into one
    flat array, and each position's segment is normalised with ``reduceat``.
    """
    index = [canonical_actions(p) for p in positions]
    lengths = np.fromiter((len(i) for i in index), np.intp, len(index))
    rows = np.repeat(np.arange(len(index)), lengths)
    cols = np.fromiter(itertools.chain.from_iterable(index), np.intp)
    legal = logits[rows, cols]
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    # subtract each segment's max first so exp can't overflow
    e = np.exp(legal - np.repeat(np.maximum.reduceat(legal, starts), lengths))
    p = e / np.repeat(np.add.reduceat(e, starts), lengths)
    return np.split(p, starts[1:])


# Width of the network's combined output: 256 policy logits, then win / draw /
# loss probabilities
OUTPUT_SIZE = 256 + 3


def to_evaluation(
    out: npt.NDArray[np.float32], positions: Sequence[Position]
) -> Evaluation:
    """Turn raw network output ``(N, OUTPUT_SIZE)`` into an ``Evaluation``."""
    policy, wdl = out[:, :-3], np.array(out[:, -3:])
    priors = legal_softmax(policy, positions)
    return Evaluation(priors, wdl[:, 0] - wdl[:, 2], wdl)


_CAPTURE_STREAMS: dict[torch.device, torch.cuda.Stream] = {}


def _capture_stream(device: torch.device) -> torch.cuda.Stream:
    """The one side stream per device used for all CUDA graph captures."""
    if device not in _CAPTURE_STREAMS:
        _CAPTURE_STREAMS[device] = torch.cuda.Stream(device)
    return _CAPTURE_STREAMS[device]


def graph_batch_size(n: int) -> int:
    """Batch size of the CUDA graph used for ``n`` positions.

    Powers of two up to 32, then multiples of 32: fine enough that padding
    wastes little (96 positions use a 96 graph, not 128), coarse enough that
    only a handful of graphs are ever captured.
    """
    if n <= 32:
        return 1 << (n - 1).bit_length()
    return -(-n // 32) * 32


class _Head(nn.Module):
    """The network plus output packing: planes in, ``(N, OUTPUT_SIZE)`` out.

    Policy logits, then win/draw/loss probabilities, in one float32 array so
    a single copy brings everything back from the GPU.
    """

    def __init__(self, model: CheckersNet) -> None:
        """Wrap ``model`` (shared, not copied)."""
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network and pack its two outputs side by side."""
        policy, wdl_logits = self.model(x)
        wdl = torch.softmax(wdl_logits.float(), dim=1)
        return torch.cat([policy.float(), wdl], dim=1)


class RawPending:
    """Raw network output on its way from the GPU."""

    def __init__(
        self,
        host: torch.Tensor,
        ready: torch.cuda.Event | None,
        keep: tuple[torch.Tensor, ...] = (),
    ) -> None:
        """Hold the output buffer, its ready event, and any buffers the GPU
        still reads from (``keep``)."""
        self.host = host
        self.ready = ready
        self.keep = keep

    def done(self) -> bool:
        """Whether the output has arrived (never waits)."""
        return self.ready is None or self.ready.query()

    def result(self) -> npt.NDArray[np.float32]:
        """Wait for the output, then return it as ``(N, OUTPUT_SIZE)``."""
        if self.ready is not None:
            self.ready.synchronize()
        self.keep = ()
        return self.host.numpy()


BACKENDS = ('auto', 'tensorrt', 'graphs', 'eager')


class NetEvaluator:
    """Evaluates positions with a ``CheckersNet``.

    Input is compact features (see ``core.encoding.features``); a
    ``PlaneExpander`` turns them into planes on the GPU. Three ways to run
    the network:

    - ``tensorrt``: NVIDIA TensorRT compiles the network into one optimised
      fp16 engine that accepts any batch size up to ``max_batch``: 1.5-2.4x
      the speed of the CUDA graphs below. Building it takes ~10 s, once.
      TensorRT bakes the weights into the engine, so after training changes
      them, call ``weights_changed()``: the engine is then *refitted* (new
      weights swapped in, ~1 s) before its next use.
    - ``graphs``: launching the ~35 small GPU kernels of a forward pass costs
      more CPU time than running them, so each batch is padded to a fixed
      size (see ``graph_batch_size``) and the expansion plus forward pass for
      that size is recorded once as a CUDA graph, then replayed with a single
      launch. Graphs read the model's weights where they live, and the
      optimiser updates them in place, so they always see current weights.
      Runs in bf16.
    - ``eager``: plain PyTorch; used on the CPU.

    ``auto`` lets the trainer try both on real iterations and keep the
    faster (see ``Trainer``; ``candidates``, ``use`` and ``settle``). It's
    CUDA graphs otherwise, including when ``torch_tensorrt`` isn't
    installed (``fallback_reason`` says why). Isolated timings mislead
    here: what decides is the real server (one pinned core, workers busy,
    a refit per iteration). On an RTX 3070, TensorRT wins for the 10x128
    network (GPU-bound) and CUDA graphs for the 6x64 one (whose server is
    CPU-bound, and TensorRT calls cost more CPU).
    """

    def __init__(
        self,
        model: CheckersNet,
        backend: str = 'auto',
        max_batch: int = 512,
    ) -> None:
        """Wrap ``model`` (shared with the trainer, not copied)."""
        if backend not in BACKENDS:
            raise ValueError(f'backend must be one of {BACKENDS}')
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.max_batch = max_batch
        self.expander = PlaneExpander().to(self.device)
        self.head = _Head(model)
        self.fallback_reason: str | None = None
        cuda = self.device.type == 'cuda'
        if cuda:
            # channels-last layout: 20-45% faster convolutions on the GPU.
            # Converts the model in place (the trainer shares it; training
            # benefits too, and the optimiser keeps the same parameters).
            # (torch's type stubs omit the memory_format keyword of to())
            model.to(memory_format=torch.channels_last)  # type: ignore
        # 'auto' starts on CUDA graphs; prepare() may switch to TensorRT
        self.auto = cuda and backend == 'auto'
        self.choice_note = ''  # why 'auto' picked what it did
        if not cuda:
            self.backend = 'eager'
        elif backend == 'auto':
            self.backend = 'graphs'
            if not _tensorrt_available():
                self.fallback_reason = _TENSORRT_ERROR
        else:
            self.backend = backend
        # batch size -> (graph, static bits, static scalars, static output)
        self._graphs: dict[int, tuple[Any, ...]] = {}
        self._engine: Any = None  # the TensorRT engine module
        self._graphed: Any = None  # the same, run as CUDA graphs
        self._graphed_context: Any = None
        self._weights_version = 0  # bumped by weights_changed()
        self._engine_version = -1  # weights the engine was built/refit with
        # batch size -> padded (bits, scalars) inputs for the TensorRT path
        self._inputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    # -- public ---------------------------------------------------------------

    def describe(self) -> str:
        """The backend in use, and why if it isn't the one asked for."""
        if self.fallback_reason:
            return (
                f'{self.backend} (TensorRT unavailable: {self.fallback_reason})'
            )
        return self.backend + (
            f' ({self.choice_note})' if self.choice_note else ''
        )

    def prepare(self) -> float:
        """Build the TensorRT engine now if it may be used (else it's built
        on first use); returns the seconds it took."""
        start = time.perf_counter()
        if self.backend == 'tensorrt' or 'tensorrt' in self.candidates():
            self._tensorrt_engine()
        return time.perf_counter() - start

    def candidates(self) -> list[str]:
        """Backends ``auto`` still has to choose between (one if settled)."""
        if self.auto and self.fallback_reason is None:
            return ['graphs', 'tensorrt']
        return [self.backend]

    def use(self, backend: str) -> None:
        """Run on ``backend`` for now (while ``auto`` tries them out)."""
        self.backend = backend

    def settle(self, backend: str, note: str) -> None:
        """End ``auto``'s trial: keep ``backend`` from now on."""
        self.auto = False
        self.backend = backend
        self.choice_note = note
        if backend != 'tensorrt':
            self._drop_engine()

    def _drop_engine(self) -> None:
        """Free the TensorRT engine (not chosen)."""
        if self._graphed_context is not None:
            self._graphed_context.__exit__(None, None, None)
        self._engine = self._graphed = self._graphed_context = None
        self._engine_version = -1
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def weights_changed(self) -> None:
        """Note that the model's weights changed (TensorRT then refits)."""
        self._weights_version += 1

    @torch.no_grad()
    def submit(self, positions: Sequence[Position]) -> _NetPending:
        """Queue the network on ``positions`` and return without waiting."""
        bits, scalars = features(positions)
        return _NetPending(self.submit_features(bits, scalars), positions)

    def evaluate(self, positions: Sequence[Position]) -> Evaluation:
        """Evaluate ``positions`` and wait for the result."""
        return self.submit(positions).result()

    @torch.no_grad()
    def submit_features(
        self, bits: npt.NDArray[np.integer], scalars: npt.NDArray[np.float32]
    ) -> RawPending:
        """Queue the network on compact features; return at once.

        ``bits (N, 5)`` and ``scalars (N, 2)`` as from
        ``core.encoding.features``. Used directly by the self-play inference
        server (``training.actors``), whose workers send features. Inputs go
        through pinned memory and outputs come back asynchronously, so the
        CPU can prepare the next batch while this one runs.
        """
        n = len(bits)
        if self.device.type != 'cuda':
            out = self._eager(
                torch.from_numpy(np.asarray(bits, np.int64)),
                torch.from_numpy(scalars),
            )
            return RawPending(out, None)
        staged_bits = torch.empty((n, 5), dtype=torch.int64, pin_memory=True)
        staged_bits.numpy()[:] = bits
        staged_scalars = torch.empty(
            (n, 2), dtype=torch.float32, pin_memory=True
        )
        staged_scalars.numpy()[:] = scalars
        host = torch.empty(
            (n, OUTPUT_SIZE), dtype=torch.float32, pin_memory=True
        )
        for start in range(0, n, self.max_batch):
            end = min(n, start + self.max_batch)
            chunk_bits = staged_bits[start:end]
            chunk_scalars = staged_scalars[start:end]
            out = self._run(chunk_bits, chunk_scalars)
            # stream-ordered: later batches can't overwrite the buffers
            # read here before these copies have finished
            host[start:end].copy_(out, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record()
        # the staged inputs must live until the GPU has copied them
        return RawPending(host, ready, (staged_bits, staged_scalars))

    # -- backends -------------------------------------------------------------

    def _run(self, bits: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """Output for one chunk of pinned features, still on the GPU."""
        n = len(bits)
        if self.backend == 'tensorrt':
            # Pad to a bucket size (as for CUDA graphs below): the engine
            # runs in TensorRT's CUDA-graph mode, which records one graph
            # per input shape, so this keeps the number of shapes small.
            size = graph_batch_size(n)
            static_bits, static_scalars = self._padded(size)
            static_bits[:n].copy_(bits, non_blocking=True)
            static_scalars[:n].copy_(scalars, non_blocking=True)
            x = self.expander(static_bits, static_scalars)
            return self._tensorrt_engine()(x)[:n]
        if self.backend == 'graphs':
            graph, static_bits, static_scalars, static_out = self._graph(
                graph_batch_size(n)
            )
            # rows past n keep stale inputs; their outputs are ignored
            static_bits[:n].copy_(bits, non_blocking=True)
            static_scalars[:n].copy_(scalars, non_blocking=True)
            graph.replay()
            return static_out[:n]
        return self._eager(
            bits.to(self.device, non_blocking=True),
            scalars.to(self.device, non_blocking=True),
        )

    def _eager(self, bits: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """Plain PyTorch forward pass (bf16 autocast on the GPU)."""
        x = self.expander(bits, scalars)
        if self.device.type != 'cuda':
            return self.head(x)
        x = x.contiguous(memory_format=torch.channels_last)
        # cache off: weight casts must be part of a captured graph, so each
        # replay casts the current weights
        with torch.autocast('cuda', torch.bfloat16, cache_enabled=False):
            return self.head(x)

    def _graph(self, size: int) -> tuple[Any, ...]:
        """The CUDA graph for batches of ``size``, captured on first use."""
        if size not in self._graphs:
            static_bits = torch.zeros(
                (size, 5), dtype=torch.int64, device=self.device
            )
            static_scalars = torch.zeros((size, 2), device=self.device)
            # Warm up on a side stream first, as CUDA graph capture requires.
            # Always the same stream: cuBLAS keeps a workspace (~8 MB) for
            # every stream it has seen and never frees it, so a new stream
            # per capture leaked GPU memory with every evaluator created.
            stream = _capture_stream(self.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._eager(static_bits, static_scalars)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                static_out = self._eager(static_bits, static_scalars)
            self._graphs[size] = (
                graph,
                static_bits,
                static_scalars,
                static_out,
            )
        return self._graphs[size]

    def _padded(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reusable device inputs of batch ``size`` for the TensorRT path."""
        if size not in self._inputs:
            self._inputs[size] = (
                torch.zeros((size, 5), dtype=torch.int64, device=self.device),
                torch.zeros((size, 2), device=self.device),
            )
        return self._inputs[size]

    def _tensorrt_engine(self) -> Any:
        """The TensorRT engine in CUDA-graph mode, built on first use and
        refitted whenever the weights have changed since.

        CUDA-graph mode replays each input shape's recorded launches in one
        go, cutting the CPU cost per call. After a refit the wrapper is
        recreated, so graphs are re-recorded against the new weights.
        """
        if self._engine_version == self._weights_version:
            return self._graphed
        import torch_tensorrt
        import torch_tensorrt.runtime

        example = torch.zeros(
            (min(256, self.max_batch), NUM_PLANES, 8, 8), device=self.device
        )
        self.model.eval()
        if self._engine is None:
            spec = torch_tensorrt.Input(
                min_shape=(1, NUM_PLANES, 8, 8),
                opt_shape=tuple(example.shape),
                max_shape=(self.max_batch, NUM_PLANES, 8, 8),
                dtype=torch.float32,
            )
            with _quiet_tensorrt():
                self._engine = torch_tensorrt.compile(
                    self.head,
                    ir='dynamo',
                    inputs=[spec],
                    enabled_precisions={torch.float16},
                    immutable_weights=False,  # so it can be refitted
                )
        else:
            with _quiet_tensorrt():
                exported = torch.export.export(self.head, (example,))
                self._engine = torch_tensorrt.dynamo.refit_module_weights(
                    self._engine, exported, arg_inputs=(example,)
                )
        if self._graphed_context is not None:
            self._graphed_context.__exit__(None, None, None)
        with _quiet_tensorrt():
            context = torch_tensorrt.runtime.enable_cudagraphs(self._engine)
            self._graphed = context.__enter__()
        self._graphed_context = context
        self._engine_version = self._weights_version
        return self._graphed


class _NetPending:
    """Output for known positions on its way from the GPU."""

    def __init__(self, raw: RawPending, positions: Sequence[Position]) -> None:
        """Pair the raw output with the positions it belongs to."""
        self.raw = raw
        self.positions = positions

    def result(self) -> Evaluation:
        """Wait for the output, then turn logits into per-move priors."""
        return to_evaluation(self.raw.result(), self.positions)


_TENSORRT_ERROR: str | None = None
_TENSORRT_CHECKED = False


def _tensorrt_available() -> bool:
    """Whether ``torch_tensorrt`` imports (checked once; the reason for a
    failure is kept in ``_TENSORRT_ERROR``)."""
    global _TENSORRT_ERROR, _TENSORRT_CHECKED
    if not _TENSORRT_CHECKED:
        _TENSORRT_CHECKED = True
        try:
            with _quiet_tensorrt():
                import torch_tensorrt  # noqa: F401
        except Exception as e:  # ImportError, or a broken install
            _TENSORRT_ERROR = f'{type(e).__name__}: {e}'
    return _TENSORRT_ERROR is None


@contextmanager
def _quiet_tensorrt() -> Iterator[None]:
    """Silence torch_tensorrt's import and build chatter (warnings about
    optional extras, progress logs)."""
    loggers = [logging.getLogger(n) for n in ('torch_tensorrt', 'torch')]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
            warnings.simplefilter('ignore')
            yield
    finally:
        for lg, level in zip(loggers, levels, strict=True):
            lg.setLevel(level)


class EvalCache:
    """Remembers network results for recently seen positions.

    The same position often comes up again: in another branch of a search
    tree (a transposition, e.g. two independent moves played in either
    order), in the next move's search, or in another game. Self-play checks
    the cache when a search reaches a leaf; on a hit, the leaf is expanded
    at once and that game's search carries on without waiting for the GPU.

    The key is the position *and* its draw counters, since those are part
    of the network's input too. Entries are only valid for one set of
    weights: call ``clear()`` whenever they change. The oldest entries are
    dropped first once ``capacity`` is reached.
    """

    def __init__(self, capacity: int) -> None:
        """Hold up to ``capacity`` results."""
        self.capacity = capacity
        # insertion-ordered, so the oldest entry is first; OrderedDict drops
        # it in constant time (a plain dict slows down as its front fills
        # with deleted slots, which makes eviction quadratic)
        self.entries: OrderedDict[tuple[int, ...], tuple[Any, ...]] = (
            OrderedDict()
        )
        self.lookups = 0
        self.hits = 0

    @staticmethod
    def key(pos: Position) -> tuple[int, ...]:
        """What identifies a position's network input."""
        return (
            pos.black, pos.white, pos.kings, pos.turn, pos.jump_sq,
            pos.no_progress, pos.reps,
        )  # fmt: skip

    def get(self, pos: Position) -> tuple[Any, float, Any] | None:
        """``(priors, value, wdl)`` for ``pos`` if remembered, else None."""
        self.lookups += 1
        entry = self.entries.get(self.key(pos))
        if entry is not None:
            self.hits += 1
        return entry

    def put(self, pos: Position, priors: Any, value: float, wdl: Any) -> None:
        """Remember ``pos``'s result, dropping the oldest entry if full."""
        if len(self.entries) >= self.capacity:
            self.entries.popitem(last=False)
        self.entries[self.key(pos)] = (priors, value, wdl)

    def clear(self) -> None:
        """Forget everything (the network's weights changed)."""
        self.entries.clear()
        self.lookups = self.hits = 0


class UniformEvaluator:
    """Uniform priors and a neutral value.

    Search is then guided only by the game endings it reaches, which makes it a
    handy reference for testing search logic.
    """

    def evaluate(self, positions: Sequence[Position]) -> Evaluation:
        """Equal priors for every legal move and a value of 0 (unknown)."""
        priors = [
            np.full(len(p.legal_actions()), 1.0 / len(p.legal_actions()))
            for p in positions
        ]
        n = len(positions)
        return Evaluation(priors, np.zeros(n), np.full((n, 3), 1.0 / 3))


class RolloutEvaluator(UniformEvaluator):
    """Uniform priors, values from random playouts.

    Classic MCTS without a network.
    """

    def __init__(
        self,
        rollouts: int = 1,
        max_plies: int = 200,
        rng: random.Random | None = None,
    ) -> None:
        """Set playouts per position and the ply cap of each playout."""
        self.rollouts = rollouts
        self.max_plies = max_plies
        self.rng = rng or random.Random()

    def evaluate(self, positions: Sequence[Position]) -> Evaluation:
        """Uniform priors; the value averages random playouts."""
        priors = super().evaluate(positions).priors
        values = np.array(
            [
                np.mean([self._playout(p) for _ in range(self.rollouts)])
                for p in positions
            ]
        )
        wins, losses = np.clip(values, 0, 1), np.clip(-values, 0, 1)
        wdl = np.stack([wins, 1 - wins - losses, losses], axis=1)
        return Evaluation(priors, values, wdl)

    def _playout(self, pos: Position) -> float:
        """Play random moves to the end of the game.

        Returns the outcome for the side to move in ``pos``.
        """
        turn = pos.turn
        for _ in range(self.max_plies):
            result = pos.result()
            if result is not None:
                return result if pos.turn == turn else -result
            pos = pos.play(self.rng.choice(pos.legal_actions()))
        return 0.0
