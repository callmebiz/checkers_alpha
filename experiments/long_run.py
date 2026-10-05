"""Long run: one network trained on and off for days or weeks.

Start, stop, come back a week later, carry on: training picks up exactly
where it stopped (``train.py --resume`` restores weights, optimiser,
counters, random state and the replay buffer). This module supervises
those sessions::

    python -m experiments.long_run start runs/long_10x128 \\
        --from runs/size_study/runs/10x128      # the first time
    python -m experiments.long_run start runs/long_10x128 --for-hours 48
    python -m experiments.long_run stop runs/long_10x128

(``train.bat`` and ``train_stop.bat`` in the project root run these.)

Segments
--------
A session trains in *segments* of a few hours, each a fresh trainer
process resumed from the last checkpoint. Between segments the supervisor
plays a gauntlet. Fresh processes also cap slow memory growth (TensorRT
leaks a little host memory per weight refit). A segment that crashes or
stalls (no progress for 30 minutes) is resumed, up to 3 times in a row.

Learning rate
-------------
Open-ended: ``lr`` halves after every 24 hours of training, down to
``lr_min`` (``TrainConfig.lr_schedule = 'step'``). Change it, or any other
setting but the network's shape, when starting a session:
``--set train.lr=0.0005`` (passed on to ``train.py --resume``).

Gauntlet
--------
The in-run arena's Elo chain drifts over days. So every few hours of
training (``gauntlet_hours``), the newest checkpoint plays fixed
opponents: alpha-beta at depths 4, 6 and 8 and the network-size study's
final networks, plus the previous gauntlet's checkpoint. All games go into
one tournament (``training.tournament``), so every checkpoint of the run's
life is rated on one scale, with alpha-beta depth 4 at 0 (as in the size
study's report).

Stopping
--------
``stop`` asks the trainer to finish its iteration, save and exit
(``control.json``), and the supervisor not to start anything new; a
gauntlet in progress is cut short (its finished games are kept). The GPU
is then free for the UI. ``start`` again later resumes.

Files (in the run directory, next to the trainer's)
------------------------------------------------------
- ``STATUS.md``: refreshed every 5 minutes while a session runs.
- ``REPORT.md`` and ``figures/``: rewritten every hour and after every
  gauntlet.
- ``long_run.log``: sessions, segments, gauntlets, restarts, resources.
- ``long_run.json``: the supervisor's settings and state.
- ``gauntlet/``: the tournament (plan, games, ratings).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from experiments.size_study import (
    checkpoints,
    commit_reading,
    gpu_reading,
    keep_awake,
    kill_tree,
    tree_rss_mb,
)
from training import fileio
from training.logs import BlockFormatter, event

# The supervisor runs for days: it imports nothing heavy (no torch; the
# trainer, gauntlets and reports run in their own processes), so its own
# memory stays small.

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / 'runs' / 'size_study' / 'runs'
SUITES = ROOT / 'configs' / 'suites'
STATUS_EVERY = 300.0  # seconds between STATUS.md refreshes
REPORT_EVERY = 3600.0  # seconds between REPORT.md refreshes (also
# after every gauntlet and at the end of a session)
STALL_LIMIT = 1800.0  # seconds without progress before a restart
MAX_FAILURES = 3  # crashed segments in a row before giving up

# settings for a new long run (kept in long_run.json; editable)
DEFAULTS: dict[str, Any] = {
    'segment_hours': 6.0,  # hours of training per trainer process
    'gauntlet_hours': 6.0,  # hours of training between gauntlets
    'gauntlet_openings': 20,  # per pairing, each with both colours
    'gauntlet_sims': 200,  # search per move for the networks
}
# config changes made when a run becomes a long run
LONG_CONFIG = [
    'train.iterations=100000000',
    'train.lr_schedule="step"',
    'train.lr=0.002',
    'train.lr_min=0.0001',
    'train.lr_step_hours=24',
    'train.lr_step_factor=0.5',
]


class LongRun:
    """A long run's supervisor (see the module docstring)."""

    def __init__(self, run: Path) -> None:
        """Open the long run in ``run`` (see ``init`` for a new one)."""
        self.run = run.resolve()
        self.settings_path = self.run / 'long_run.json'
        self.state = json.loads(self.settings_path.read_text(encoding='utf-8'))
        self.log = self._logger()
        self.current: subprocess.Popen[bytes] | None = None
        self.stage = 'idle'
        self.session_start = time.time()
        self.restarts: list[str] = []
        self._lock = threading.Lock()
        self._done = threading.Event()

    # -- setup -----------------------------------------------------------------

    @staticmethod
    def init(run: Path, source: Path) -> None:
        """Make ``run`` a long run continuing ``source`` (a copy: the source
        run is left as it was), with the open-ended learning rate."""
        if run.exists():
            raise SystemExit(f'{run} already exists')
        if not (source / 'checkpoints' / 'latest.pt').exists():
            raise SystemExit(f'{source} has no checkpoint to continue from')
        from training.config import RunConfig, changed_config  # torch

        shutil.copytree(source, run)
        for name in ('status.json', 'control.json'):
            (run / name).unlink(missing_ok=True)
        config_path = run / 'config.json'
        config, changes = changed_config(
            RunConfig.load(config_path),
            LONG_CONFIG + ['train.time_budget_hours=0'],
        )
        config.save(config_path)
        fileio.append_jsonl(
            run / 'config_changes.jsonl',
            {'time': time.time(), 'changes': changes, 'from': str(source)},
        )
        state = {**DEFAULTS, 'source': str(source), 'gauntlets': []}
        fileio.write_text(run / 'long_run.json', json.dumps(state, indent=2))
        print(f'{run}: continues {source}; ' + '; '.join(changes))

    def _logger(self) -> logging.Logger:
        """Log to ``long_run.log`` and the console."""
        log = logging.getLogger(f'long_run:{self.run}')
        if not log.handlers:
            log.setLevel(logging.INFO)
            log.propagate = False
            file = logging.FileHandler(
                self.run / 'long_run.log', encoding='utf-8'
            )
            file.setFormatter(BlockFormatter('%Y-%m-%d %H:%M:%S'))
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(BlockFormatter('%H:%M:%S'))
            log.addHandler(file)
            log.addHandler(console)
        return log

    def _save(self) -> None:
        """Write ``long_run.json``."""
        with self._lock:
            fileio.write_text(
                self.settings_path, json.dumps(self.state, indent=2)
            )

    # -- where the run is ------------------------------------------------------

    def trained_hours(self) -> float:
        """Hours of training so far (from the metrics)."""
        rows = fileio.read_jsonl(self.run / 'metrics.jsonl')
        return rows[-1].get('elapsed_hours', 0.0) if rows else 0.0

    def stop_flag(self) -> Path:
        """The file that tells the supervisor to stop."""
        return self.run / 'long_run.stop'

    def stopping(self) -> bool:
        """Whether a stop was asked for."""
        return self.stop_flag().exists()

    def phase(self) -> str:
        """The trainer's last phase (``finished``, ``stopped``, ...)."""
        path = self.run / 'status.json'
        if not path.exists():
            return ''
        return json.loads(path.read_text(encoding='utf-8')).get('phase', '')

    # -- a session -------------------------------------------------------------

    def session(self, for_hours: float, overrides: list[str]) -> None:
        """Train (in segments, with gauntlets) until stopped, or for
        ``for_hours`` of wall-clock time if given."""
        pid_path = self.run / 'long_run.pid'
        if pid_path.exists():
            other = int(pid_path.read_text() or 0)
            if other and psutil.pid_exists(other) and other != os.getpid():
                raise SystemExit(f'a session is already running (pid {other})')
        pid_path.write_text(str(os.getpid()))
        self.stop_flag().unlink(missing_ok=True)
        (self.run / 'control.json').unlink(missing_ok=True)
        keep_awake()
        end = self.session_start + for_hours * 3600 if for_hours else None
        event(
            self.log,
            'RUN',
            f'session start: {self.run.name} at {self.trained_hours():.2f} h '
            f'of training'
            + (f', for {for_hours:g} h' if for_hours else ', until stopped'),
        )
        watcher = threading.Thread(target=self._monitor, daemon=True)
        watcher.start()
        self.write_status()  # don't leave the last session's status up
        failures = 0
        try:
            while not self.stopping():
                if end and time.time() >= end:
                    event(self.log, 'RUN', 'session time is up')
                    break
                if self._gauntlet_due():
                    self.gauntlet()
                    continue
                if self._skills_due():
                    self.skills()
                    continue
                code, why = self.segment(end, overrides)
                overrides = []  # applied once, then saved in the config
                phase = self.phase()
                if phase in ('finished', 'stopped') and code == 0:
                    failures = 0
                    continue
                failures += 1
                note = f'segment ended badly ({why}, phase {phase!r})'
                self.restarts.append(f'{datetime.now():%m-%d %H:%M} {note}')
                event(self.log, 'WARN', note)
                if failures >= MAX_FAILURES:
                    event(
                        self.log,
                        'WARN',
                        f'{MAX_FAILURES} failures in a row: stopping the '
                        'session (see the logs, then start again)',
                    )
                    break
        except KeyboardInterrupt:
            event(self.log, 'RUN', 'interrupted (Ctrl+C)')
        finally:
            self._done.set()
            self.stage = 'stopped'
            self.write_status()
            self.write_report()
            pid_path.unlink(missing_ok=True)
            self.stop_flag().unlink(missing_ok=True)
            event(
                self.log,
                'RUN',
                f'session end at {self.trained_hours():.2f} h of training; '
                'resume with train.bat (or: python -m experiments.long_run '
                f'start {self.run})',
            )

    def segment(
        self, end: float | None, overrides: list[str]
    ) -> tuple[int, str]:
        """Train until the next gauntlet, the segment's length, or the
        session's end, whichever comes first."""
        hours = self.trained_hours()
        last = max((g['hours'] for g in self.state['gauntlets']), default=0)
        target = min(
            last + self.state['gauntlet_hours'],
            hours + self.state['segment_hours'],
        )
        if end:  # don't train past the session's end (roughly: wall time)
            target = min(target, hours + (end - time.time()) / 3600)
        target = max(target, hours + 0.05)
        args = [
            'train.py',
            '--resume',
            str(self.run),
            '--progress',
            'off',
            '--set',
            f'train.time_budget_hours={target:.3f}',
        ]
        for item in overrides:
            args += ['--set', item]
        event(
            self.log,
            'RUN',
            f'training: {hours:.2f} -> {target:.2f} h'
            + (f' (with {", ".join(overrides)})' if overrides else ''),
        )
        return self._supervise('training', args, self.run / 'status.json')

    def _supervise(
        self, stage: str, args: list[str], heartbeat: Path
    ) -> tuple[int, str]:
        """Run ``python args`` to the end; kill it if ``heartbeat`` stops
        changing, or (a gauntlet) when a stop is asked for."""
        console = self.run / 'logs' / f'{stage}.console.log'
        console.parent.mkdir(exist_ok=True)
        with console.open('ab') as sink:
            proc = subprocess.Popen(
                [sys.executable, *args],
                cwd=ROOT,
                stdout=sink,
                stderr=subprocess.STDOUT,
            )
            with self._lock:
                self.current, self.stage = proc, stage
            self.write_status()  # show the new stage at once
            why = 'exited'
            started = time.time()
            try:
                while proc.poll() is None:
                    time.sleep(5)
                    beat = (
                        heartbeat.stat().st_mtime if heartbeat.exists() else 0
                    )
                    if time.time() - max(beat, started) > STALL_LIMIT:
                        why = 'no progress for 30 min'
                        kill_tree(proc)
                    elif stage == 'gauntlet' and self.stopping():
                        why = 'stop requested'
                        kill_tree(proc)
            finally:
                with self._lock:
                    self.current, self.stage = None, 'idle'
        return proc.returncode, f'{why}, exit code {proc.returncode}'

    # -- gauntlet --------------------------------------------------------------

    def _gauntlet_due(self) -> bool:
        """Whether the newest checkpoint is due a gauntlet."""
        points = checkpoints(self.run)
        if not points:
            return False
        it, hours = points[-1]
        done = {g['iteration'] for g in self.state['gauntlets']}
        last = max((g['hours'] for g in self.state['gauntlets']), default=0)
        due = not done or hours >= last + self.state['gauntlet_hours'] - 0.05
        return it not in done and due

    def _anchors(self) -> list[dict[str, Any]]:
        """Fixed opponents: alpha-beta, and the size study's final networks."""
        entrants: list[dict[str, Any]] = [
            {'name': f'ab{d}', 'spec': f'ab:{d}'} for d in (4, 6, 8)
        ]
        for size in ('6x64', '8x96', '10x128'):
            points = checkpoints(STUDY / size)
            if points:
                path = (
                    STUDY
                    / size
                    / 'checkpoints'
                    / f'iter_{points[-1][0]:04d}.pt'
                )
                entrants.append(
                    {
                        'name': f'study {size}',
                        'checkpoint': str(path),
                        'sims': self.state['gauntlet_sims'],
                    }
                )
        return entrants

    def gauntlet(self) -> None:
        """Play the newest checkpoint against the fixed opponents and the
        previous gauntlet's checkpoint; refit everyone's ratings."""
        it, hours = checkpoints(self.run)[-1]
        name = f'{hours:.1f}h'
        record = {'iteration': it, 'hours': hours, 'name': name}
        gauntlets = [*self.state['gauntlets'], record]
        anchors = self._anchors()
        nets = [a['name'] for a in anchors if 'checkpoint' in a]
        entrants = anchors + [
            {
                'name': g['name'],
                'checkpoint': str(
                    self.run / 'checkpoints' / f'iter_{g["iteration"]:04d}.pt'
                ),
                'sims': self.state['gauntlet_sims'],
            }
            for g in gauntlets
        ]
        blocks = [
            {
                'name': 'anchors',
                'pairs': [['ab4', 'ab6'], ['ab6', 'ab8']]
                + [[n, ab] for n in nets for ab in ('ab6', 'ab8')],
            }
        ]
        for i, g in enumerate(gauntlets):
            pairs = [[g['name'], o] for o in ['ab6', 'ab8', *nets]]
            if i:
                pairs.append([g['name'], gauntlets[i - 1]['name']])
            blocks.append({'name': g['name'], 'pairs': pairs})
        plan = {
            'config': str(self.run / 'config.json'),
            'entrants': entrants,
            'blocks': blocks,
            'openings': self.state['gauntlet_openings'],
            'anchor': 'ab4',
            'chunk_games': 300,
            'seed': 11,
        }
        out = self.run / 'gauntlet'
        out.mkdir(exist_ok=True)
        fileio.write_text(out / 'plan.json', json.dumps(plan, indent=2))
        event(self.log, 'ARENA', f'gauntlet: {name} (iteration {it + 1})')
        code, why = self._supervise(
            'gauntlet',
            ['-m', 'training.tournament', str(out / 'plan.json'), str(out)],
            out / 'tournament.log',
        )
        if code != 0:
            event(self.log, 'WARN', f'gauntlet {name} cut short ({why})')
            return
        ratings = json.loads((out / 'ratings.json').read_text())['ratings']
        mine = ratings.get(name, {})
        record['elo'] = mine.get('elo')
        self.state['gauntlets'] = gauntlets
        self._save()
        table = ', '.join(
            f'{n} {r["elo"]:.0f}'
            for n, r in ratings.items()
            if not n[0].isdigit()
        )
        event(
            self.log,
            'ARENA',
            f'gauntlet {name}: Elo {mine.get("elo", 0):.0f} '
            f'[{mine.get("low", 0):.0f}, {mine.get("high", 0):.0f}] '
            f'(ab4 = 0; {table})',
        )
        self.write_report()

    # -- skills (the test suites) ----------------------------------------------

    def _skills_path(self) -> Path:
        """Where suite scores go (one row per player)."""
        return self.run / 'suites.jsonl'

    def _skills_todo(self) -> list[tuple[str, str, float | None]]:
        """``(name, target, hours)`` not yet scored on the suites:
        gauntlet checkpoints, the study's networks and alpha-beta."""
        if not (SUITES / 'tactics.json').exists():
            return []  # not built (python -m training.suites build)
        done = {r['name'] for r in fileio.read_jsonl(self._skills_path())}
        todo: list[tuple[str, str, float | None]] = []
        for g in self.state['gauntlets']:
            path = self.run / 'checkpoints' / f'iter_{g["iteration"]:04d}.pt'
            if g['name'] not in done and path.exists():
                todo.append((g['name'], str(path), g['hours']))
        for a in self._anchors():
            if a['name'] in done:
                continue
            target = a.get('checkpoint') or a['spec']
            todo.append((a['name'], target, None))
        return todo

    def _skills_due(self) -> bool:
        """Whether anything is waiting to be scored on the suites."""
        return bool(self._skills_todo())

    def skills(self) -> None:
        """Score whatever is waiting on the suites (each in its own
        process, which frees the GPU memory between them)."""
        for name, target, hours in self._skills_todo():
            if self.stopping():
                return
            args = [
                '-m', 'training.suites', 'score', target,
                '--out', str(self._skills_path()), '--name', name,
            ]  # fmt: skip
            if hours is not None:
                args += ['--hours', f'{hours:.3f}']
            code, why = self._supervise(
                'skills', args, self.run / 'logs' / 'skills.console.log'
            )
            if code != 0:
                event(self.log, 'WARN', f'skills {name} failed ({why})')
                return  # don't retry in a loop: next session tries again
            event(self.log, 'ARENA', f'skills scored: {name}')
        self.write_report()

    # -- reporting -------------------------------------------------------------

    def _monitor(self) -> None:
        """Refresh STATUS.md and log resources every ``STATUS_EVERY`` s,
        and REPORT.md every ``REPORT_EVERY`` s."""
        last_report = time.monotonic()
        while not self._done.wait(STATUS_EVERY):
            try:
                self.write_status()
                if time.monotonic() - last_report >= REPORT_EVERY:
                    last_report = time.monotonic()
                    self.write_report()
            except Exception as error:  # monitoring must never kill a run
                event(self.log, 'WARN', f'status update failed: {error!r}')

    def write_status(self) -> None:
        """Rewrite STATUS.md, and log one resource reading (called by the
        monitor, and whenever the stage changes)."""
        with self._lock:
            proc, stage = self.current, self.stage
        memory = psutil.virtual_memory()
        reading: dict[str, Any] = {
            'time': time.time(),
            'stage': stage,
            'system_used_mb': round(memory.used / 2**20),
            'stage_rss_mb': round(tree_rss_mb(proc.pid)) if proc else 0,
            'disk_free_gb': round(shutil.disk_usage(self.run).free / 2**30),
            **commit_reading(),
            **gpu_reading(),
        }
        fileio.append_jsonl(self.run / 'resources.jsonl', reading)
        commit = reading.get('commit_mb', 0) / max(
            reading.get('commit_limit_mb', 1), 1
        )
        event(
            self.log,
            'RUN',
            f'resources [{stage}]: process tree '
            f'{reading["stage_rss_mb"] / 1024:.1f} GB, commit {commit:.0%}, '
            f'GPU {reading.get("gpu_mb", 0) / 1024:.1f} GB '
            f'{reading.get("gpu_temp", 0):.0f}C, '
            f'disk {reading["disk_free_gb"]} GB free',
            to='file',
        )
        rows = fileio.read_jsonl(self.run / 'metrics.jsonl')
        arena = fileio.read_jsonl(self.run / 'arena.jsonl')
        last = rows[-1] if rows else {}
        lines = [
            f'# Long run: {self.run.name}',
            '',
            f'Updated {datetime.now():%Y-%m-%d %H:%M}. Now: **{stage}**. '
            f'Session running {(time.time() - self.session_start) / 3600:.1f} '
            'h.',
            '',
            f'- Training: {last.get("elapsed_hours", 0):.2f} h, '
            f'iteration {last.get("iteration", -1) + 1}, '
            f'lr {last.get("lr", 0):.2g}, '
            f'{last.get("seconds", 0):.0f} s/iteration, '
            f'{last.get("sims_per_second", 0) / 1e3:.0f}k sims/s',
        ]
        if arena:
            scores = ', '.join(
                f'{r["opponent"]} {r["wins"]}-{r["draws"]}-{r["losses"]}'
                for r in arena[-1]['results']
            )
            lines.append(f'- Last in-run evaluation: {scores}')
        gauntlets = [g for g in self.state['gauntlets'] if g.get('elo')]
        if gauntlets:
            lines.append(
                '- Gauntlet Elo (ab4 = 0): '
                + ', '.join(
                    f'{g["name"]} {g["elo"]:.0f}' for g in gauntlets[-6:]
                )
            )
        lines += [
            f'- Commit charge {commit:.0%}, GPU '
            f'{reading.get("gpu_temp", 0):.0f} C, disk '
            f'{reading["disk_free_gb"]} GB free',
            '',
            '## Problems',
            '',
        ]
        problems = list(self.restarts)
        if commit > 0.9:
            problems.append(f'commit charge at {commit:.0%}: close programs')
        lines += [f'- {p}' for p in problems] or ['- none']
        fileio.write_text(self.run / 'STATUS.md', '\n'.join(lines) + '\n')

    def write_report(self) -> None:
        """Rewrite REPORT.md and its figures, in a short-lived process
        (plotting loads matplotlib and torch); never fails the run."""
        console = self.run / 'logs' / 'report.console.log'
        console.parent.mkdir(exist_ok=True)
        with console.open('ab') as sink:
            code = subprocess.call(
                [
                    sys.executable,
                    '-m',
                    'experiments.long_run',
                    'report',
                    str(self.run),
                ],
                cwd=ROOT,
                stdout=sink,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        if code:
            event(self.log, 'WARN', f'report failed (see {console.name})')


def spread(values: list[float], gap: float) -> list[float]:
    """Label heights for lines at ``values`` (sorted ascending), nudged
    apart so neighbours are at least ``gap`` apart, staying centred on the
    lines they label."""
    out = list(values)
    for _ in range(50):  # settles in a few rounds
        moved = False
        for i in range(1, len(out)):
            short = gap - (out[i] - out[i - 1])
            if short > 1e-9:
                out[i - 1] -= short / 2
                out[i] += short / 2
                moved = True
        if not moved:
            break
    return out


def label_lines(ax: Any, refs: dict[str, float], fmt: str = '{:.0f}') -> None:
    """Name horizontal reference lines at the right edge, spreading labels
    that would overlap (lines a few Elo apart) and pointing each to its
    line."""
    if not refs:
        return
    names = sorted(refs, key=refs.__getitem__)
    low, high = ax.get_ylim()
    low, high = min(low, min(refs.values())), max(high, max(refs.values()))
    ax.set_ylim(low, high)
    gap = (high - low) * 0.055  # about one line of small text
    ys = spread([refs[n] for n in names], gap)
    for name, y in zip(names, ys, strict=True):
        ax.annotate(
            f'{name} ({fmt.format(refs[name])})',
            xy=(1.0, refs[name]),
            xycoords=('axes fraction', 'data'),
            xytext=(6, 0),
            textcoords='offset points',
            fontsize=8,
            va='center',
            annotation_clip=False,
        )
        if abs(y - refs[name]) > 1e-6:  # moved: draw it at y, point back
            ax.texts[-1].remove()
            ax.annotate(
                f'{name} ({fmt.format(refs[name])})',
                xy=(1.0, refs[name]),
                xycoords=('axes fraction', 'data'),
                xytext=(1.0, y),
                textcoords=('axes fraction', 'data'),
                fontsize=8,
                va='center',
                annotation_clip=False,
                arrowprops={'arrowstyle': '-', 'color': 'grey', 'lw': 0.6},
            )
            # shift the text right of the axis edge
            ax.texts[-1].set_position((1.02, y))


SKILLS = [
    ('judgement', 'Endgame judgement',
     'quiet solved endgames: is its most likely outcome the true one?'),
    ('the_move', 'The move',
     'same position, other side to move, different true result: is the '
     'pair ranked the right way round?'),
    ('deep_wins', 'Deep wins',
     'wins needing 25-60 turns of exact play: does it see them as won?'),
    ('shots', 'Shots',
     'a sacrifice that wins material, verified 12 turns deep: found?'),
    ('only_moves', 'Only moves',
     'every other move loses a man (12-turn search): found?'),
]  # fmt: skip


def skills_section(run: Path) -> list[str]:
    """The suites over training (see ``training.suites``), with the
    study's networks and alpha-beta for comparison."""
    import matplotlib.pyplot as plt

    from experiments import report

    rows = fileio.read_jsonl(run / 'suites.jsonl')
    if not rows:
        return []
    timeline = sorted(
        (r for r in rows if r.get('hours') is not None),
        key=lambda r: r['hours'],
    )
    refs = [r for r in rows if r.get('hours') is None]
    fig, grid = plt.subplots(2, 3, figsize=(15, 8))
    axes = grid.flat
    for ax, (key, title, _) in zip(axes, SKILLS, strict=False):
        for who, style in (('net', '--'), ('search', '-')):
            pts = [(r['hours'], r[key][who]) for r in timeline if key in r]
            if not pts:
                continue
            x = [p[0] for p in pts]
            ax.plot(
                x,
                [p[1]['rate'] for p in pts],
                style,
                marker='o',
                label='network alone' if who == 'net' else 'with search',
            )
            ax.fill_between(
                x,
                [p[1]['low'] for p in pts],
                [p[1]['high'] for p in pts],
                alpha=0.15,
            )
        marks = {}
        for r in refs:
            stats = r.get(key)
            if stats:
                best = stats.get('search', stats)  # engines: one rate
                marks[r['name']] = best['rate'] if 'rate' in best else 0
        for rate in marks.values():
            ax.axhline(rate, color='grey', lw=0.7, ls=':')
        ax.set_ylim(0, 1.02)  # before labelling: it sets their spacing
        label_lines(ax, marks, '{:.0%}')
        ax.set_title(title)
        ax.set_xlabel('hours of training')
    axes[0].legend(fontsize=8)
    grid.flat[-1].axis('off')  # five panels in a 2x3 grid
    fig.subplots_adjust(right=0.86, wspace=0.55, hspace=0.35)
    lines = [
        '## Skills',
        '',
        'Fixed puzzle sets, the same for every checkpoint '
        '(`configs/suites/`, built by `python -m training.suites build`). '
        "Shaded: 90% intervals. Dotted: the size study's networks (with "
        'search) and alpha-beta (tactics only).',
        '',
        f'![skills]({report.save(fig, run, "skills")})',
        '',
    ]
    for _, title, what in SKILLS:
        lines.append(f'- **{title}**: {what}')
    lines.append('')
    header = ['player'] + [title for _, title, _ in SKILLS]
    table = []
    for r in [*timeline[-3:], *refs]:
        cells = [r['name']]
        for key, _, _ in SKILLS:
            st = r.get(key)
            if not st:
                cells.append('')
            elif 'net' in st:
                cells.append(
                    f'{st["net"]["rate"]:.0%} / {st["search"]["rate"]:.0%}'
                )
            else:
                cells.append(f'{st["rate"]:.0%}')
        table.append(cells)
    lines += ['Latest (network alone / with search):', '']
    lines += report.table(header, table)
    return lines + ['']


def write_report(run: Path, state: dict[str, Any]) -> Path:
    """REPORT.md for a long run: gauntlet ratings over training time, the
    training curves, in-run evaluations, probes and memory."""
    import matplotlib.pyplot as plt

    from experiments import report

    r = report.Run(run.name, run)
    lines = [
        f'# Long run: {run.name}',
        '',
        f'Generated {datetime.now():%Y-%m-%d %H:%M}. Continues '
        f'`{state.get("source", "-")}`.',
        '',
    ]
    ratings_path = run / 'gauntlet' / 'ratings.json'
    if ratings_path.exists():
        ratings = json.loads(ratings_path.read_text())['ratings']
        points = [
            (g['hours'], ratings[g['name']])
            for g in state['gauntlets']
            if g['name'] in ratings
        ]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        if points:
            ax.errorbar(
                [p[0] for p in points],
                [p[1]['elo'] for p in points],
                yerr=[
                    [p[1]['elo'] - p[1]['low'] for p in points],
                    [p[1]['high'] - p[1]['elo'] for p in points],
                ],
                marker='o',
                capsize=3,
                label=run.name,
            )
        refs = {n: v['elo'] for n, v in ratings.items() if not n[0].isdigit()}
        for name, elo in refs.items():
            # alpha-beta dashed, the study's networks dotted
            style = '--' if name.startswith('ab') else ':'
            ax.axhline(elo, color='grey', lw=0.8, ls=style)
        label_lines(ax, refs)
        ax.set_xlabel('hours of training')
        ax.set_ylabel('Elo (ab4 = 0)')
        ax.set_title('Gauntlet rating over training')
        lines += [
            '## Strength',
            '',
            'Each gauntlet checkpoint against fixed opponents (alpha-beta and '
            "the size study's final networks), 200 simulations per move; one "
            'fit over all games, 90% intervals.',
            '',
            f'![gauntlet]({report.save(fig, run, "gauntlet")})',
            '',
        ]
        lines += report.table(
            ['player', 'Elo', '90% interval', 'games'],
            [
                [
                    n,
                    f'{v["elo"]:.0f}',
                    f'{v["low"]:.0f} to {v["high"]:.0f}',
                    v['games'],
                ]
                for n, v in ratings.items()
            ],
        )
        lines.append('')
    lines += skills_section(run)
    lines += report.arena_section(run, [r])
    lines += report.curves_section(run, [r])
    lines += report.probes_section(run, [r])
    health, alarms = report.health_section(run, [r], {})
    lines += health + [f'- **Check:** {a}' for a in alarms]
    path = run / 'REPORT.md'
    fileio.write_text(path, '\n'.join(lines) + '\n')
    return path


def main() -> None:
    """Command line: start, stop or report on a long run."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    parser.add_argument('command', choices=['start', 'stop', 'report'])
    parser.add_argument('run', type=Path)
    parser.add_argument(
        '--from', dest='source', type=Path, help='new run: continue this run'
    )
    parser.add_argument(
        '--for-hours', type=float, default=0, help='stop after this long'
    )
    parser.add_argument(
        '--set',
        action='append',
        default=[],
        metavar='SECTION.KEY=VALUE',
        help='change a setting from this session on (not the model)',
    )
    args = parser.parse_args()
    run = args.run
    if args.command == 'stop':
        if not run.exists():
            raise SystemExit(f'no run at {run}')
        (run / 'long_run.stop').write_text('stop')
        fileio.write_text(run / 'control.json', json.dumps({'stop': True}))
        print(
            'Stop requested: training stops after its current iteration '
            '(about a minute), a gauntlet at once.'
        )
        return
    if args.command == 'report':  # no LongRun: it would log to the file
        state = json.loads((run / 'long_run.json').read_text('utf-8'))
        print(write_report(run.resolve(), state))
        return
    if not (run / 'long_run.json').exists():
        if args.source is None:
            raise SystemExit(f'{run} is not a long run: give --from <run>')
        LongRun.init(run, args.source)
    LongRun(run).session(args.for_hours, args.set)


if __name__ == '__main__':
    main()
