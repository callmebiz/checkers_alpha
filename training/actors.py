"""Self-play in several worker processes (actors) sharing one GPU.

Why processes
-------------
After batching and CUDA graphs, self-play is limited by single-threaded
Python work: walking search trees and generating moves. One process keeps
the GPU only about half busy and uses one CPU core. Python threads can't
run that work in parallel (the GIL), but processes can.

How it works
------------
Each actor is a process with its own copy of the network on the GPU and its
own ``SelfPlay`` (pipelined batches, CUDA graphs, games carried over between
iterations). Each iteration:

1. The trainer writes the current weights to ``actor_weights.pt``.
2. It asks every actor for its share of ``games_per_iteration``.
3. Each actor loads the weights (in place, so its CUDA graphs stay valid),
   plays its games, and sends back its training samples, the move logs of
   games chosen for saving, and statistics. A short message per finished
   game drives the progress bar.

The GPU runs the actors' kernels side by side. Actors start once per run
(importing torch and creating a CUDA context takes a few seconds each) and
each needs about 1.5-2 GB of RAM.

Failures: if an actor process dies (for example, out of memory), the trainer
raises an error naming it instead of waiting forever. Actors are shut down
when the run ends, including on Ctrl+C.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.inference import NetEvaluator
from core.model import CheckersNet
from training import fileio
from training.config import RunConfig
from training.replay_buffer import Samples
from training.self_play import SelfPlay, SelfPlayStats

POLL_SECONDS = 1.0  # how often to check actors are alive while waiting


def _actor_main(
    index: int,
    config: dict[str, Any],
    tasks: mp.Queue,
    results: mp.Queue,
) -> None:
    """Entry point of an actor process: play games when asked, until told
    to stop."""
    try:
        cfg = RunConfig.from_dict(config)
        workers = cfg.self_play.workers
        # concurrent_games is the total across all actors
        cfg.self_play.concurrent_games = max(
            1, -(-cfg.self_play.concurrent_games // workers)
        )
        # each actor saves its share of the games chosen for saving
        cfg.self_play.saved_games = cfg.self_play.saved_games // workers + (
            index < cfg.self_play.saved_games % workers
        )
        device = torch.device(cfg.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            device = torch.device('cpu')
        model = CheckersNet(cfg.model).to(device).eval()
        # Each actor gets its own random stream, reproducible from the seed
        seed = None if cfg.seed is None else [cfg.seed, index]
        self_play = SelfPlay(
            NetEvaluator(model),
            cfg.self_play,
            cfg.search,
            cfg.rules,
            np.random.default_rng(seed),
        )
        loaded = None
        while True:
            task = tasks.get()
            if task[0] == 'stop':
                return
            _, weights_path, version, num_games = task
            if version != loaded:
                state = torch.load(weights_path, map_location=device)
                model.load_state_dict(state)  # copies in place
                loaded = version
            records, stats = self_play.play(
                num_games, lambda _: results.put(('game', index))
            )
            samples = Samples.concat([r.samples for r in records])
            logged = [r.to_json() for r in records if r.logged]
            results.put(('done', index, samples, logged, stats.to_dict()))
    except KeyboardInterrupt:
        return  # Ctrl+C reaches every process; the trainer handles it
    except Exception:
        results.put(('error', index, traceback.format_exc()))


class ActorPool:
    """A fixed set of self-play actor processes."""

    def __init__(self, config: RunConfig, workers: int, weights_path: Path):
        """Start ``workers`` actor processes sharing ``weights_path``."""
        self.weights_path = weights_path
        context = mp.get_context('spawn')
        self.tasks = [context.Queue() for _ in range(workers)]
        self.results: mp.Queue = context.Queue()
        self.processes = [
            context.Process(
                target=_actor_main,
                args=(i, config.to_dict(), self.tasks[i], self.results),
                name=f'actor-{i}',
                daemon=True,
            )
            for i in range(workers)
        ]
        for process in self.processes:
            process.start()

    def play(
        self,
        state: dict[str, torch.Tensor],
        version: int,
        num_games: int,
        on_game_end: Callable[[], None] | None = None,
    ) -> tuple[Samples, list[dict[str, Any]], SelfPlayStats]:
        """Play ``num_games`` split across the actors with weights ``state``.

        ``version`` identifies the weights: actors reload only when it
        changes. Returns the samples, logged games and summed statistics.
        """
        start = time.perf_counter()
        fileio.torch_save(state, self.weights_path)
        workers = len(self.processes)
        shares = [
            num_games // workers + (i < num_games % workers)
            for i in range(workers)
        ]
        for task_queue, share in zip(self.tasks, shares, strict=True):
            task_queue.put(('play', str(self.weights_path), version, share))

        parts: list[Samples] = []
        logged: list[dict[str, Any]] = []
        stats = SelfPlayStats()
        pending = sum(share > 0 for share in shares)
        while pending:
            message = self._next_message()
            if message[0] == 'game':
                if on_game_end is not None:
                    on_game_end()
            elif message[0] == 'done':
                _, _, samples, games, actor_stats = message
                parts.append(samples)
                logged += games
                for f in fields(SelfPlayStats):
                    if f.name != 'seconds':
                        setattr(
                            stats,
                            f.name,
                            getattr(stats, f.name) + actor_stats[f.name],
                        )
                pending -= 1
            else:
                self.close()
                raise RuntimeError(f'actor {message[1]} failed:\n{message[2]}')
        stats.seconds = time.perf_counter() - start
        return Samples.concat(parts), logged, stats

    def _next_message(self) -> tuple[Any, ...]:
        """Wait for an actor's message, checking that none has died."""
        while True:
            try:
                return self.results.get(timeout=POLL_SECONDS)
            except queue.Empty:
                for process in self.processes:
                    if not process.is_alive():
                        code = process.exitcode
                        self.close()
                        raise RuntimeError(
                            f'{process.name} died (exit code {code})'
                        ) from None

    def close(self) -> None:
        """Stop the actors: ask nicely, then terminate stragglers."""
        for task_queue, process in zip(self.tasks, self.processes, strict=True):
            if process.is_alive():
                try:
                    task_queue.put(('stop',))
                except (OSError, ValueError):
                    pass
        deadline = time.monotonic() + 10
        for process in self.processes:
            process.join(max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                process.terminate()
                process.join(1)
