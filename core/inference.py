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

import itertools
import random
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, NamedTuple, Protocol

import numpy as np
import numpy.typing as npt
import torch

from core.encoding import NUM_PLANES, canonical_actions, encode
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


class _NetPending:
    """Network output on its way from the GPU (see ``NetEvaluator.submit``)."""

    def __init__(
        self,
        host: torch.Tensor,
        ready: torch.cuda.Event | None,
        positions: Sequence[Position],
    ) -> None:
        """Hold the output buffer, the event marking it ready, the positions."""
        self.host = host
        self.ready = ready
        self.positions = positions

    def result(self) -> Evaluation:
        """Wait for the copy from the GPU, then turn logits into priors."""
        if self.ready is not None:
            self.ready.synchronize()  # wait for the copy to host memory
        out = self.host.numpy()
        policy, wdl = out[:, :-3], out[:, -3:]
        priors = legal_softmax(policy, self.positions)
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


class NetEvaluator:
    """Evaluates positions with a ``CheckersNet`` (bf16 autocast on CUDA).

    CUDA graphs: launching the ~35 small GPU kernels of a forward pass costs
    more CPU time than running them. On CUDA, each batch is padded up to a
    fixed size (see ``graph_batch_size``; up to ``max_graph_batch``), and the
    forward pass for that size is recorded once as a CUDA graph, then
    replayed with a single launch. The graph reads the model's weights where
    they live, and the optimiser updates them in place, so replays always use
    the current weights. Larger batches, and CPU models, run normally.
    """

    def __init__(
        self,
        model: CheckersNet,
        use_amp: bool = True,
        use_cuda_graphs: bool = True,
        max_graph_batch: int = 512,
    ) -> None:
        """Wrap ``model``; bf16 and CUDA graphs are only used on a GPU."""
        self.model = model.eval()
        self.device = next(model.parameters()).device
        cuda = self.device.type == 'cuda'
        self.use_amp = use_amp and cuda
        self.use_cuda_graphs = use_cuda_graphs and cuda
        self.max_graph_batch = max_graph_batch
        # batch size -> (graph, static input, static output)
        self._graphs: dict[
            int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]
        ] = {}

    def _autocast(self) -> Any:
        """The mixed-precision context for a forward pass (or a no-op)."""
        if not self.use_amp:
            return nullcontext()
        # cache off: weight casts must be part of a captured graph, so each
        # replay casts the current weights
        return torch.autocast('cuda', torch.bfloat16, cache_enabled=False)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, 256 + 3)``: policy logits, then win/draw/loss probabilities
        (one array, so a single copy brings everything back)."""
        with self._autocast():
            policy, wdl_logits = self.model(x)
        wdl = torch.softmax(wdl_logits.float(), dim=1)
        return torch.cat([policy.float(), wdl], dim=1)

    def _graph(
        self, size: int
    ) -> tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]:
        """The CUDA graph for batches of ``size``, captured on first use."""
        if size not in self._graphs:
            static_in = torch.zeros(
                (size, NUM_PLANES, 8, 8), device=self.device
            )
            # Warm up on a side stream first, as CUDA graph capture requires.
            # Always the same stream: cuBLAS keeps a workspace (~8 MB) for
            # every stream it has seen and never frees it, so a new stream
            # per capture leaked GPU memory with every evaluator created.
            stream = _capture_stream(self.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._forward(static_in)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                static_out = self._forward(static_in)
            self._graphs[size] = (graph, static_in, static_out)
        return self._graphs[size]

    @torch.no_grad()
    def submit(self, positions: Sequence[Position]) -> _NetPending:
        """Queue the network on ``positions`` and return without waiting.

        On CUDA the output is copied into pinned host memory asynchronously
        and an event marks when it has arrived.
        """
        n = len(positions)
        x = torch.from_numpy(encode(positions))
        if self.device.type != 'cuda':
            return _NetPending(self._forward(x), None, positions)

        if self.use_cuda_graphs and n <= self.max_graph_batch:
            graph, static_in, static_out = self._graph(graph_batch_size(n))
            # rows past n keep stale inputs; their outputs are ignored
            static_in[:n].copy_(x)
            graph.replay()
            out = static_out[:n]
        else:
            out = self._forward(x.to(self.device))
        host = torch.empty(out.shape, dtype=out.dtype, pin_memory=True)
        # stream-ordered: this copy finishes before the next replay of the
        # same graph can overwrite static_out
        host.copy_(out, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record()
        return _NetPending(host, ready, positions)

    def evaluate(self, positions: Sequence[Position]) -> Evaluation:
        """Evaluate ``positions`` and wait for the result."""
        return self.submit(positions).result()


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
