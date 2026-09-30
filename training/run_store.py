"""The run directory: everything a training run writes, in one place.

The trainer only writes here and the UI only reads (except ``probes.json``,
which either may edit), so either can restart without the other. Layout::

    runs/<name>/
      config.json             the fully resolved RunConfig
      status.json             live heartbeat: phase, iteration, progress
      metrics.jsonl           one row per iteration: losses, outcomes, speed
      arena.jsonl             one row per evaluation: scores, Elo
      probes.json             probe positions (editable during the run)
      probes.jsonl            probe results, one row per evaluation
      train.log               detailed human-readable log
      games/
        iter_NNNN.jsonl       a few self-play games per iteration
        arena_NNNN.jsonl      every arena game of an evaluation
      checkpoints/
        iter_NNNN.pt          model weights + config (play, arena, probes)
        latest.pt             newest full state: weights, optimiser,
                              counters, RNG (for resuming)
        buffer.npz            replay buffer as of latest.pt

``.jsonl`` files hold one JSON object per line, so they can be appended to
while a reader follows along. Rows carry an ``iteration`` field: 0-based,
the iteration they describe. ``status.json`` is replaced atomically (write a
temporary file, then rename) so a reader never sees half a file. The same
goes for checkpoints and the buffer: an interrupted save leaves the previous
file intact rather than a corrupt one (see ``training.fileio``).
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from training import fileio
from training.config import RunConfig
from training.fileio import append_jsonl, read_jsonl, write_jsonl
from training.logs import BlockFormatter, Destination
from training.progress import TqdmHandler

DEFAULT_PROBES = Path(__file__).parent.parent / 'configs' / 'probes.json'


class RunStore:
    """Reads and writes the files of one run directory."""

    def __init__(self, root: str | Path) -> None:
        """Point at the run directory ``root`` (nothing is created)."""
        self.root = Path(root)
        self.checkpoints = self.root / 'checkpoints'
        self.games = self.root / 'games'
        self.config_path = self.root / 'config.json'
        self.metrics_path = self.root / 'metrics.jsonl'
        self.arena_path = self.root / 'arena.jsonl'
        self.probes_path = self.root / 'probes.json'
        self.probe_results_path = self.root / 'probes.jsonl'
        self.status_path = self.root / 'status.json'
        self.latest_path = self.checkpoints / 'latest.pt'
        self.buffer_path = self.checkpoints / 'buffer.npz'

    @classmethod
    def create(cls, root: str | Path, config: RunConfig) -> RunStore:
        """Start a new run. Refuses to reuse a directory that has a run."""
        store = cls(root)
        if store.config_path.exists():
            raise FileExistsError(
                f'{store.root} already holds a run; resume it or pick another'
            )
        store.checkpoints.mkdir(parents=True, exist_ok=True)
        store.games.mkdir(exist_ok=True)
        config.save(store.config_path)
        if DEFAULT_PROBES.exists():
            shutil.copyfile(DEFAULT_PROBES, store.probes_path)
        else:
            store.probes_path.write_text('[]\n', encoding='utf-8')
        return store

    @classmethod
    def open(cls, root: str | Path) -> RunStore:
        """Open an existing run; raises if there is none."""
        store = cls(root)
        if not store.config_path.exists():
            raise FileNotFoundError(f'No run found in {store.root}')
        return store

    def load_config(self) -> RunConfig:
        """The run's saved config."""
        return RunConfig.load(self.config_path)

    # -- logs and results ------------------------------------------------------

    def logger(self) -> logging.Logger:
        """Logger for this run (see ``training.logs`` for the layout).

        ``train.log`` gets the full reports; the console gets short
        one-line entries, printed above any progress bars.
        """
        log = logging.getLogger(f'run:{self.root.resolve()}')
        if not log.handlers:
            log.setLevel(logging.INFO)
            log.propagate = False
            file = logging.FileHandler(
                self.root / 'train.log', encoding='utf-8'
            )
            file.setFormatter(BlockFormatter('%Y-%m-%d %H:%M:%S'))
            file.addFilter(Destination('file'))
            console = TqdmHandler()
            console.setFormatter(BlockFormatter('%H:%M:%S'))
            console.addFilter(Destination('console'))
            log.addHandler(file)
            log.addHandler(console)
        return log

    def close_logger(self) -> None:
        """Close the log file (handlers are re-created on next use)."""
        log = logging.getLogger(f'run:{self.root.resolve()}')
        for handler in list(log.handlers):
            handler.close()
            log.removeHandler(handler)

    def append_metrics(self, row: dict[str, Any]) -> None:
        """Append one iteration's metrics row."""
        append_jsonl(self.metrics_path, row)

    def read_metrics(self) -> list[dict[str, Any]]:
        """All metrics rows so far."""
        return read_jsonl(self.metrics_path)

    def append_arena(self, row: dict[str, Any]) -> None:
        """Append one arena evaluation row."""
        append_jsonl(self.arena_path, row)

    def read_arena(self) -> list[dict[str, Any]]:
        """All arena rows so far."""
        return read_jsonl(self.arena_path)

    def append_probe_results(self, row: dict[str, Any]) -> None:
        """Append one row of probe results."""
        append_jsonl(self.probe_results_path, row)

    def read_probe_results(self) -> list[dict[str, Any]]:
        """All probe result rows, back-filled ones included."""
        return read_jsonl(self.probe_results_path)

    def truncate_after(self, last_iteration: int) -> None:
        """Drop result rows after ``last_iteration``.

        Used when resuming from a checkpoint older than the newest results:
        those iterations will be run again.
        """
        for path in (
            self.metrics_path,
            self.arena_path,
            self.probe_results_path,
        ):
            if path.exists():
                rows = read_jsonl(path)
                write_jsonl(
                    path, (r for r in rows if r['iteration'] <= last_iteration)
                )

    def write_status(self, status: dict[str, Any]) -> None:
        """Replace ``status.json``. Skipped (not retried forever) if a
        reader holds the file: the next refresh will catch up."""
        fileio.write_text(self.status_path, json.dumps(status), required=False)

    def save_games(self, name: str, games: Iterable[dict[str, Any]]) -> None:
        """Write games to ``games/<name>.jsonl``, replacing the file."""
        write_jsonl(self.games / f'{name}.jsonl', games)

    # -- checkpoints -----------------------------------------------------------

    def checkpoint_path(self, iteration: int) -> Path:
        """Path of the checkpoint for ``iteration``."""
        return self.checkpoints / f'iter_{iteration:04d}.pt'

    def save_checkpoint(
        self, iteration: int, weights: dict[str, Any], full: dict[str, Any]
    ) -> Path:
        """Save ``weights`` as ``iter_NNNN.pt`` and ``full`` (weights plus
        optimiser and counters, for resuming) as ``latest.pt``."""
        path = self.checkpoint_path(iteration)
        fileio.torch_save(weights, path)
        fileio.torch_save(full, self.latest_path)
        return path

    def load_latest(self) -> dict[str, Any] | None:
        """The newest full checkpoint, or None before the first save."""
        if not self.latest_path.exists():
            return None
        return torch.load(
            self.latest_path, map_location='cpu', weights_only=False
        )

    def list_checkpoints(self) -> list[Path]:
        """All per-iteration checkpoints, oldest first."""
        return sorted(self.checkpoints.glob('iter_*.pt'))
