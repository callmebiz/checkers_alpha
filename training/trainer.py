"""The training loop: self-play, then learn from it, repeated.

Each iteration
--------------
1. **Self-play** (``training.self_play``): the current network plays
   ``games_per_iteration`` games against itself. Recorded positions go into
   the replay buffer; a few games are saved for replay in the UI.
2. **Train**: once the buffer holds ``min_buffer`` samples, run
   ``new samples * sample_reuse / batch_size`` optimiser steps on random
   batches from the buffer.
3. **Probes** (``training.probes``): every ``probes.every`` iterations,
   evaluate the run's probe positions and append to ``probes.jsonl``.
4. **Log** one row of metrics to ``metrics.jsonl`` and a detailed report to
   ``train.log`` (see below).
5. **Checkpoint** every ``checkpoint_every`` iterations and at the end.
6. **Arena** (``training.arena``): every ``arena.every`` iterations (after a
   checkpoint), play matches against fixed opponents and the previous
   evaluated checkpoint; append scores and Elo to ``arena.jsonl``.

The loss
--------
- Policy: cross-entropy between the search's visit distribution and the
  network's policy, with the softmax taken over *legal moves only*. Illegal
  logits never enter the loss, so the network spends no effort on them.
- Value: cross-entropy between the win/draw/loss prediction and the actual
  game outcome, weighted by ``value_weight``.

Also logged: ``policy_kl`` (policy loss minus the target's own entropy: 0
means the network already matches the search), prediction entropy, value
accuracy and gradient norm.

Learning rate: linear warmup over ``warmup_steps``, then a cosine decay from
``lr`` to ``lr_min`` over the run's iterations.

Output
------
The console shows progress bars (``training.progress``) and only notable
events. ``train.log`` gets a full report per iteration and per arena
evaluation, including memory use, so a long run can be checked afterwards
for slowdowns or leaks. ``status.json`` is refreshed every few seconds with
the current phase and progress, for the UI.

Stopping and resuming: Ctrl+C saves a checkpoint and exits. ``resume`` reloads
the model, optimiser, replay buffer, counters and random state from
``latest.pt`` and continues with the next iteration.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from core.game import Position
from core.inference import NetEvaluator
from core.model import CheckersNet, load_model
from core.players import make_player
from training.actors import ActorPool
from training.arena import (
    Contestant,
    MatchResult,
    play_match,
    random_openings,
    tally,
)
from training.config import RunConfig
from training.logs import (
    arena_line,
    arena_report,
    event,
    iteration_line,
    iteration_report,
    postfix,
    probe_report,
    run_header,
)
from training.probes import evaluate_probes, load_probes
from training.progress import Progress, interactive, memory_snapshot
from training.replay_buffer import ReplayBuffer, Samples
from training.run_store import RunStore
from training.self_play import SelfPlay

STATUS_INTERVAL = 2.0  # seconds between status.json refreshes within a phase


def compute_loss(
    model: CheckersNet,
    x: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    wdl: torch.Tensor,
    value_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Training loss for one batch, plus statistics for logging.

    The statistics stay on the device as 0-d tensors: reading them (e.g.
    ``float()``) makes the CPU wait for the GPU, so the caller does that once
    after many steps rather than every step.

    Args:
        x: input planes ``(B, NUM_PLANES, 8, 8)``.
        target: search visit distributions ``(B, 256)``.
        mask: legal network actions ``(B, 256)`` bool.
        wdl: outcome class per row ``(B,)``: 0 win, 1 draw, 2 loss.
    """
    logits, wdl_logits = model(x)
    logits = logits.float().masked_fill(~mask, float('-inf'))
    log_p = torch.log_softmax(logits, dim=1).masked_fill(~mask, 0.0)
    policy_loss = -(target * log_p).sum(1).mean()
    value_loss = F.cross_entropy(wdl_logits.float(), wdl)
    loss = policy_loss + value_weight * value_loss

    with torch.no_grad():
        log_t = torch.log(target.clamp_min(1e-12))
        target_entropy = -(target * log_t).sum(1).mean()
        entropy = -(log_p.exp() * log_p).sum(1).mean()
        accuracy = (wdl_logits.argmax(1) == wdl).float().mean()
    stats = {
        'policy_loss': policy_loss.detach(),
        'value_loss': value_loss.detach(),
        'policy_kl': (policy_loss - target_entropy).detach(),
        'policy_entropy': entropy,
        'value_accuracy': accuracy,
    }
    return loss, stats


class Trainer:
    """Owns the model, optimiser and replay buffer of one run."""

    def __init__(self, config: RunConfig, store: RunStore) -> None:
        """Build the model, optimiser, buffer and self-play for ``config``."""
        self.config = config
        self.store = store
        self.log = store.logger()
        self.device = torch.device(config.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            event(self.log, 'WARN', 'CUDA unavailable, training on CPU')
            self.device = torch.device('cpu')
        if config.seed is not None:
            torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)

        self.model = CheckersNet(config.model).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )
        self.buffer = ReplayBuffer(config.train.buffer_size)
        # One evaluator for self-play, probes and the arena, so its CUDA
        # graphs are captured once. It always sees the latest weights.
        self.evaluator = NetEvaluator(self.model, backend=config.backend)
        # Kept for the whole run so unfinished games carry over (see
        # training.self_play).
        self.self_play = SelfPlay(
            self.evaluator,
            config.self_play,
            config.search,
            config.rules,
            self.rng,
        )
        # Worker processes for self-play, started by run() when configured
        self.actors: ActorPool | None = None
        # 'auto' backend trial: sims/s measured per backend (see _try_backend)
        self._trial: dict[str, float] = {}
        self._trial_step = 0
        self.iteration = 0  # next iteration to run
        self.step = 0  # optimiser steps taken
        self.saved_iteration = -1  # last iteration in a checkpoint
        self.last_elo: float | None = None
        self.progress = Progress(0, 0, enabled=False)
        self._phase = ''
        self._status_time = 0.0

    @classmethod
    def new(cls, config: RunConfig, run_dir: str | Path) -> Trainer:
        """Start a new run in ``run_dir``."""
        return cls(config, RunStore.create(run_dir, config))

    @classmethod
    def resume(cls, run_dir: str | Path) -> Trainer:
        """Continue the run in ``run_dir`` from its latest checkpoint."""
        store = RunStore.open(run_dir)
        trainer = cls(store.load_config(), store)
        ckpt = store.load_latest()
        if ckpt is not None:
            trainer._restore(ckpt)
        store.truncate_after(trainer.iteration - 1)
        arena = store.read_arena()
        trainer.last_elo = arena[-1]['elo'] if arena else None
        return trainer

    # -- main loop -------------------------------------------------------------

    def run(self, progress_bars: bool | None = None) -> None:
        """Train until ``train.iterations``. Bars are shown when
        ``progress_bars`` is True, or by default when stdout is a terminal."""
        cfg = self.config.train
        root = self.store.root
        header = run_header(
            self.config, str(root), str(self.device), self.iteration
        )
        event(self.log, 'RUN', header, to='file')
        verb = 'start' if self.iteration == 0 else 'resume'
        event(
            self.log,
            'RUN',
            f'{verb} {root}: {cfg.iterations} iterations on {self.device}; '
            f'full log in {root / "train.log"}',
            to='console',
        )
        if progress_bars is None:
            progress_bars = interactive()
        self.progress = Progress(cfg.iterations, self.iteration, progress_bars)
        workers = self.config.self_play.workers
        built = self.evaluator.prepare()  # e.g. the TensorRT engine
        note = f', engine built in {built:.0f} s' if built else ''
        what = self.evaluator.describe()
        if len(self.evaluator.candidates()) > 1:
            what = 'auto: trying CUDA graphs and TensorRT on iterations 2-3'
        event(self.log, 'RUN', f'inference backend: {what}{note}')
        try:
            if workers > 0 and self.iteration < cfg.iterations:
                event(self.log, 'RUN', f'starting {workers} self-play workers')
                self.actors = ActorPool(self.config, workers)
            while self.iteration < cfg.iterations:
                self.run_iteration()
                done = self.iteration - 1
                last = self.iteration == cfg.iterations
                every = self.config.arena.every
                arena_due = every > 0 and (self.iteration % every == 0 or last)
                checkpoint_due = self.iteration % cfg.checkpoint_every == 0
                if last or arena_due or checkpoint_due:
                    self.save()  # also the arena's next "previous" opponent
                if arena_due:
                    self.run_arena(done)
            self._status('finished', force=True)
            self.progress.close()
            event(self.log, 'RUN', f'finished {cfg.iterations} iterations')
        except KeyboardInterrupt:
            self.progress.close()
            event(self.log, 'RUN', 'interrupted')
            if self.iteration > 0 and self.saved_iteration < self.iteration - 1:
                self.save()
            if self.iteration > 0:
                event(
                    self.log,
                    'RUN',
                    f'stopped after iteration {self.saved_iteration + 1}; '
                    f'continue with: python train.py --resume '
                    f'{self.store.root}',
                )
            self._status('stopped', force=True)
        finally:
            if self.actors is not None:
                self.actors.close()
                self.actors = None
            self.progress.close()
            self.store.close_logger()

    def run_iteration(self) -> dict[str, Any]:
        """Self-play, train, and log one iteration. Returns its metrics."""
        i = self.iteration
        cfg = self.config
        t0 = time.perf_counter()

        self._phase_start(
            'self-play', cfg.self_play.games_per_iteration, 'game'
        )
        self.model.eval()
        games = cfg.self_play.games_per_iteration
        trying = self._try_backend()
        if self.actors is not None:
            # this process serves the workers' network evaluations
            new, saved, sp = self.actors.play(self.evaluator, games, self._tick)
        else:
            records, sp = self.self_play.play(games, lambda _: self._tick())
            new = Samples.concat([r.samples for r in records])
            saved = [r.to_json() for r in records if r.logged]
            del records
        self.buffer.add(new)
        self.store.save_games(f'iter_{i:04d}', saved)
        if trying is not None:
            self._trial_result(trying, sp.sims / max(sp.seconds, 1e-9))

        t1 = time.perf_counter()
        steps = 0
        if len(self.buffer) >= cfg.train.min_buffer:
            reuse = len(new) * cfg.train.sample_reuse
            steps = max(1, math.ceil(reuse / cfg.train.batch_size))
        self._phase_start('train', steps, 'step')
        train_stats = self.train_steps(steps)
        train_seconds = time.perf_counter() - t1

        t2 = time.perf_counter()
        probe_results = None
        if cfg.probes.every > 0 and i % cfg.probes.every == 0:
            probe_results = self.run_probes(i)
        probe_seconds = time.perf_counter() - t2

        row: dict[str, Any] = {
            'iteration': i,
            'time': time.time(),
            'seconds': round(time.perf_counter() - t0, 2),
            'self_play': sp.to_dict(),
            'sims_per_second': round(sp.sims / max(sp.seconds, 1e-9)),
            'avg_plies': round(sp.plies / max(sp.games, 1), 1),
            'new_samples': len(new),
            'buffer_size': len(self.buffer),
            'buffer_total': self.buffer.total_added,
            'train_steps': steps,
            'train_seconds': round(train_seconds, 2),
            'probe_seconds': round(probe_seconds, 2),
            'step': self.step,
            'lr': self._lr(),
            'memory': memory_snapshot(self.device),
            **train_stats,
        }
        self.store.append_metrics(row)
        iterations = cfg.train.iterations
        event(self.log, 'ITER', iteration_report(row, cfg), to='file')
        if probe_results is not None:
            report = probe_report(i, iterations, *probe_results)
            event(self.log, 'PROBES', report, to='file')
        if self.progress.enabled:
            self.progress.iteration_done(postfix(row, self.last_elo))
        else:
            event(
                self.log, 'ITER', iteration_line(row, iterations), to='console'
            )
        self.iteration += 1
        return row

    def train_steps(self, steps: int) -> dict[str, float]:
        """Run ``steps`` optimiser steps; return averaged statistics."""
        if steps == 0:
            return {}
        cfg = self.config.train
        self.model.train()
        use_amp = self.device.type == 'cuda'
        totals: dict[str, torch.Tensor] = {}
        for _ in range(steps):
            for group in self.optimizer.param_groups:
                group['lr'] = self._lr()
            batch = self.buffer.sample(cfg.batch_size, self.rng)
            x, target, mask, wdl = self._tensors(batch)
            with torch.autocast('cuda', torch.bfloat16, enabled=use_amp):
                loss, stats = compute_loss(
                    self.model, x, target, mask, wdl, cfg.value_weight
                )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            stats['grad_norm'] = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip
            )
            self.optimizer.step()
            self.step += 1
            for key, value in stats.items():
                value = value.detach()
                totals[key] = totals[key] + value if key in totals else value
            self._tick()
        self.model.eval()
        self.evaluator.weights_changed()  # TensorRT refits before next use
        # one GPU sync per statistic, for the whole iteration
        return {k: round(float(v) / steps, 4) for k, v in totals.items()}

    # -- evaluation ------------------------------------------------------------

    def run_probes(
        self, i: int
    ) -> tuple[dict[str, list[str]], dict[str, Any]] | None:
        """Evaluate the run's probe positions; append to ``probes.jsonl``.

        ``probes.json`` is re-read every time, so edits during the run count.
        A broken file is logged and skipped rather than stopping training.
        Returns each probe's accepted moves and its results, for the log.
        """
        cfg = self.config
        try:
            probes = load_probes(self.store.probes_path, cfg.rules)
        except (OSError, ValueError, TypeError) as e:
            event(self.log, 'WARN', f'skipping probes: {e}')
            return None
        if not probes:
            return None
        self._phase_start('probes', len(probes), 'probe')
        self.model.eval()
        results = evaluate_probes(
            probes, self.evaluator, cfg.probes.sims, cfg.rules
        )
        self.store.append_probe_results({'iteration': i, 'probes': results})
        self.progress.advance(len(probes))
        return {p.name: p.accepted() for p in probes}, results

    def run_arena(self, i: int) -> dict[str, Any]:
        """Play the arena matches for iteration ``i``; append to
        ``arena.jsonl`` and save the games. Returns the row."""
        cfg = self.config.arena
        t0 = time.perf_counter()
        self.model.eval()
        rng = np.random.default_rng(cfg.seed)
        openings = random_openings(
            max(cfg.games // 2, 1), cfg.opening_plies, rng, self.config.rules
        )
        history = [r for r in self.store.read_arena() if r['iteration'] < i]
        play_previous = bool(history) and cfg.vs_previous
        matches = len(cfg.opponents) + play_previous
        self._phase_start('arena', matches * 2 * len(openings), 'game')

        name = f'iter {i}'
        prev = history[-1] if play_previous else None
        prev_name = f'iter {prev["iteration"]}' if prev else ''
        prev_evaluator = None
        if prev is not None:
            path = self.store.checkpoint_path(prev['iteration'])
            prev_model, _ = load_model(path, self.device)
            # CUDA graphs: not worth a TensorRT build for one match
            prev_evaluator = NetEvaluator(prev_model, backend='graphs')

        if self.actors is not None:
            # Spread every game over the workers: alpha-beta runs on their
            # CPUs, both networks are served from here
            results = self._arena_on_workers(
                openings, name, prev_name, prev_evaluator
            )
        else:
            me = Contestant.network(name, self.evaluator, cfg.sims)
            opponents = [
                Contestant.of(make_player(spec, seed=cfg.seed))
                for spec in cfg.opponents
            ]
            if prev_evaluator is not None:
                opponents.append(
                    Contestant.network(prev_name, prev_evaluator, cfg.sims)
                )
            results = [
                play_match(
                    me,
                    opponent,
                    openings,
                    cfg.max_plies,
                    self.config.search,
                    on_game_end=self._tick,
                )
                for opponent in opponents
            ]

        # Elo chain: previous evaluation's rating + head-to-head difference
        elo: float | None = 0.0 if not history else None
        if prev is not None and prev.get('elo') is not None:
            elo = prev['elo'] + results[-1].elo_diff
        if prev_evaluator is not None:
            del prev_evaluator  # free its weights and CUDA graphs
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        row = {
            'iteration': i,
            'elo': None if elo is None else round(elo, 1),
            'seconds': round(time.perf_counter() - t0, 1),
            'results': [r.summary() for r in results],
        }
        self.store.append_arena(row)
        self.store.save_games(
            f'arena_{i:04d}', (g for r in results for g in r.games)
        )
        self.progress.end_phase()
        iterations = self.config.train.iterations
        report = arena_report(row, iterations, self.last_elo)
        event(self.log, 'ARENA', report, to='file')
        event(self.log, 'ARENA', arena_line(row, iterations), to='console')
        self.last_elo = row['elo']
        return row

    def _arena_on_workers(
        self,
        openings: list[Position],
        name: str,
        prev_name: str,
        prev_evaluator: NetEvaluator | None,
    ) -> list[MatchResult]:
        """Play the arena's games on the self-play workers; one result per
        opponent, the previous checkpoint last."""
        assert self.actors is not None
        cfg = self.config.arena
        sides = list(cfg.opponents) + (['prev'] if prev_evaluator else [])
        tasks = []
        for opponent in sides:
            for opening in openings:
                fen = opening.to_fen()
                tasks.append((len(tasks), fen, 'net', opponent))
                tasks.append((len(tasks), fen, opponent, 'net'))
        settings = {
            'names': {'net': name, 'prev': prev_name},
            'sims': cfg.sims,
            'max_plies': cfg.max_plies,
            'seed': cfg.seed,
        }
        evaluators = [self.evaluator]
        if prev_evaluator is not None:
            evaluators.append(prev_evaluator)
        games = self.actors.play_arena(evaluators, settings, tasks, self._tick)
        names = [
            prev_name
            if side == 'prev'
            else make_player(side, seed=cfg.seed).name
            for side in sides
        ]
        return [tally(name, other, games) for other in names]

    # -- helpers ---------------------------------------------------------------

    def _tensors(
        self, batch: Samples
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """A batch as device tensors: input, target, legal mask, outcome."""

        def to(array: np.ndarray) -> torch.Tensor:
            """Move a numpy array to the training device."""
            return torch.from_numpy(array).to(self.device, non_blocking=True)

        # planes are expanded on the GPU from the compact features
        bits = to(batch.bits.astype(np.int64))
        x = self.evaluator.expander(bits, to(batch.scalars))
        if self.device.type == 'cuda':
            x = x.contiguous(memory_format=torch.channels_last)  # as model
        target = to(batch.policy.astype(np.float32))
        mask = to(batch.legal_mask())
        wdl = to(batch.wdl.astype(np.int64))
        return x, target, mask, wdl

    def _lr(self) -> float:
        """Warmup by step, then cosine decay by iteration."""
        cfg = self.config.train
        progress = self.iteration / max(cfg.iterations, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        lr = cfg.lr_min + (cfg.lr - cfg.lr_min) * cosine
        return lr * min(1.0, (self.step + 1) / max(cfg.warmup_steps, 1))

    def _try_backend(self) -> str | None:
        """Pick this iteration's backend while ``auto`` is still deciding.

        Iteration 1 warms up on CUDA graphs (first-use costs would skew it),
        then TensorRT and CUDA graphs get one iteration each. Returns the
        backend to measure this iteration, or None.
        """
        options = self.evaluator.candidates()
        if len(options) < 2:
            return None
        order = ['graphs', 'tensorrt', 'graphs']
        backend = order[min(self._trial_step, 2)]
        self._trial_step += 1
        self.evaluator.use(backend)
        return None if self._trial_step == 1 else backend

    def _trial_result(self, backend: str, rate: float) -> None:
        """Record a trial iteration; once both are measured, settle."""
        self._trial[backend] = rate
        if len(self._trial) < 2:
            return
        graphs, tensorrt = self._trial['graphs'], self._trial['tensorrt']
        # TensorRT must clearly win: it also costs a refit per iteration
        best = 'tensorrt' if tensorrt > graphs * 1.05 else 'graphs'
        note = (
            f'auto: TensorRT {tensorrt / 1e3:.0f}k vs graphs '
            f'{graphs / 1e3:.0f}k sims/s'
        )
        self.evaluator.settle(best, note)
        event(
            self.log, 'RUN', f'inference backend: {self.evaluator.describe()}'
        )

    def _phase_start(self, name: str, total: int, unit: str) -> None:
        """Begin a phase: a fresh progress bar and a status update."""
        self._phase = name
        self.progress.phase(name, total, unit)
        self._status(name, force=True)

    def _tick(self) -> None:
        """One unit of the current phase is done."""
        self.progress.advance()
        self._status(self._phase)

    def _status(self, phase: str, force: bool = False) -> None:
        """Refresh ``status.json`` (at most every ``STATUS_INTERVAL``
        seconds unless forced)."""
        now = time.time()
        if not force and now - self._status_time < STATUS_INTERVAL:
            return
        self._status_time = now
        self.store.write_status(
            {
                'phase': phase,
                'iteration': self.iteration,
                'iterations': self.config.train.iterations,
                'done': self.progress.done,
                'total': self.progress.total,
                'updated': now,
            }
        )

    # -- checkpoints -----------------------------------------------------------

    def save(self) -> Path:
        """Checkpoint the last completed iteration, with the replay buffer.

        ``iter_NNNN.pt`` gets the weights only (enough for play, the arena
        and probes); ``latest.pt`` also gets everything needed to resume.
        """
        done = self.iteration - 1
        weights = {
            'model_config': asdict(self.config.model),
            'model_state': self.model.state_dict(),
            'iteration': done,
            'config': self.config.to_dict(),
        }
        full = {
            **weights,
            'optimizer_state': self.optimizer.state_dict(),
            'step': self.step,
            'rng': {
                'numpy': self.rng.bit_generator.state,
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
        }
        self._status('saving', force=True)
        t0 = time.perf_counter()
        path = self.store.save_checkpoint(done, weights, full)
        self.buffer.save(self.store.buffer_path)
        self.saved_iteration = done
        event(
            self.log,
            'SAVE',
            f'{path.name}, latest.pt, buffer.npz ({len(self.buffer):,} '
            f'samples) in {time.perf_counter() - t0:.1f} s',
            to='file',
        )
        return path

    def _restore(self, ckpt: dict[str, Any]) -> None:
        """Load model, optimiser, counters, RNG and buffer from ``ckpt``."""
        self.model.load_state_dict(ckpt['model_state'])
        self.evaluator.weights_changed()
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.iteration = ckpt['iteration'] + 1
        self.saved_iteration = ckpt['iteration']
        self.step = ckpt['step']
        rng = ckpt['rng']
        self.rng.bit_generator.state = rng['numpy']
        torch.set_rng_state(rng['torch'])
        if rng['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng['cuda'])
        if self.store.buffer_path.exists():
            self.buffer = ReplayBuffer.load(
                self.store.buffer_path, self.config.train.buffer_size
            )
