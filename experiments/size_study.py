"""Network-size study: which size gets strongest in the same training time?

Three networks (10x128, 8x96, 6x64 residual blocks x channels) train one
after another for the same wall-clock budget, with the ``full`` config's
search and hyperparameters. The learning rate anneals over the budget (not
over iterations), so each run gets a complete schedule however fast it is.
A smaller network plays more games in the time; the question is whether
that beats a bigger network's better judgement.

Then a tournament (``training.tournament``) puts everything on one Elo
scale: the final networks, checkpoints at 1/4, 1/2 and 3/4 of training,
each final network without search (1 simulation: the raw policy), and
alpha-beta at depths 2-8. Finally ``experiments.report`` writes
``REPORT.md`` and plots.

While it runs, ``STATUS.md`` is refreshed every 5 minutes (stages, the
current run's progress, memory, GPU, problems) and ``study.log`` records
every stage, restart and resource reading. Training runs that crash or
stall (no progress for 30 minutes) are killed and resumed from their last
checkpoint, up to 3 times. Running the command again resumes the study.
The machine is kept awake while it runs::

    python -m experiments.size_study                  # 4.25 h per size
    python -m experiments.size_study --dry            # a 20-minute rehearsal
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psutil

from training import fileio
from training.logs import BlockFormatter, event

ROOT = Path(__file__).resolve().parent.parent

# name -> model shape and iterations between evaluations (~15 minutes each,
# from measured seconds per iteration: ~72, ~55 and ~30 s with this search)
SIZES: dict[str, dict[str, int]] = {
    '10x128': {'blocks': 10, 'channels': 128, 'value_hidden': 256, 'every': 12},
    '8x96': {'blocks': 8, 'channels': 96, 'value_hidden': 192, 'every': 16},
    '6x64': {'blocks': 6, 'channels': 64, 'value_hidden': 128, 'every': 30},
}
# the largest first: it's the one most likely to hit trouble (TensorRT)
ORDER = ['10x128', '8x96', '6x64']
CHECKPOINT_FRACTIONS = (0.25, 0.5, 0.75)  # of training, for the tournament
STATUS_EVERY = 300.0  # seconds between STATUS.md refreshes
STALL_LIMIT = 1800.0  # seconds without progress before a run is restarted
ATTEMPTS = 3  # tries per training run (later tries resume)


def keep_awake() -> None:
    """Stop Windows sleeping while this process runs (display may sleep)."""
    if sys.platform == 'win32':
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            es_continuous | es_system_required
        )


def kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill a process and every process it started (workers)."""
    try:
        parent = psutil.Process(proc.pid)
        family = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for p in family:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(family, timeout=30)


def tree_rss_mb(pid: int) -> float:
    """Resident memory of a process and its children, in MB."""
    try:
        parent = psutil.Process(pid)
        family = [parent] + parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0.0
    total = 0
    for p in family:
        try:
            total += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total / 2**20


class _MemoryStatus(ctypes.Structure):
    """Windows ``MEMORYSTATUSEX`` (see ``commit_reading``)."""

    _fields_ = [  # noqa: RUF012 (ctypes' own convention)
        ('length', ctypes.c_ulong),
        ('load', ctypes.c_ulong),
        ('total_phys', ctypes.c_ulonglong),
        ('avail_phys', ctypes.c_ulonglong),
        ('total_commit', ctypes.c_ulonglong),
        ('avail_commit', ctypes.c_ulonglong),
        ('total_virtual', ctypes.c_ulonglong),
        ('avail_virtual', ctypes.c_ulonglong),
        ('avail_extended', ctypes.c_ulonglong),
    ]


def commit_reading() -> dict[str, float]:
    """Windows commit charge and limit (RAM + page file), MB.

    Allocations fail when the charge reaches the limit, whatever the free
    RAM, so this is the number to watch on a long run. Empty elsewhere.
    """
    if sys.platform != 'win32':
        return {}
    status = _MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {}
    used = status.total_commit - status.avail_commit
    return {
        'commit_mb': round(used / 2**20),
        'commit_limit_mb': round(status.total_commit / 2**20),
    }


def gpu_reading() -> dict[str, float]:
    """GPU memory (MB), utilisation (%) and temperature (C), if readable."""
    query = 'memory.used,memory.total,utilization.gpu,temperature.gpu'
    try:
        line = subprocess.run(
            [
                'nvidia-smi',
                f'--query-gpu={query}',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.splitlines()[0]
        used, total, util, temp = (float(v) for v in line.split(','))
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return {}
    return {
        'gpu_mb': used,
        'gpu_total_mb': total,
        'gpu_util': util,
        'gpu_temp': temp,
    }


class Study:
    """The study's state, stages and monitoring (see the module docstring)."""

    def __init__(self, out: Path, hours: float, dry: bool) -> None:
        """Open (or start) the study in ``out``."""
        self.out = out
        self.dry = dry
        out.mkdir(parents=True, exist_ok=True)
        (out / 'logs').mkdir(exist_ok=True)
        self.state_path = out / 'state.json'
        state: dict[str, Any] = {}
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding='utf-8'))
        self.state = state
        state.setdefault('hours', hours)
        state.setdefault('dry', dry)
        state.setdefault('started', time.time())
        # everything should be over by then: training, tournament, report
        budget = state['hours'] * len(ORDER) + (0.6 if dry else 4.0)
        state.setdefault('deadline', state['started'] + budget * 3600)
        state.setdefault('stages', {})
        state.setdefault('restarts', [])
        self.hours = float(state['hours'])
        self._save_state()
        self.log = self._logger()
        self.current: subprocess.Popen[bytes] | None = None
        self.current_stage = ''
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # -- bookkeeping -----------------------------------------------------------

    def _logger(self) -> logging.Logger:
        """Log to ``study.log`` and the console."""
        log = logging.getLogger(f'size_study:{self.out.resolve()}')
        if not log.handlers:
            log.setLevel(logging.INFO)
            log.propagate = False
            file = logging.FileHandler(self.out / 'study.log', encoding='utf-8')
            file.setFormatter(BlockFormatter('%Y-%m-%d %H:%M:%S'))
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(BlockFormatter('%H:%M:%S'))
            log.addHandler(file)
            log.addHandler(console)
        return log

    def _save_state(self) -> None:
        """Write ``state.json`` atomically."""
        fileio.write_text(self.state_path, json.dumps(self.state, indent=2))

    def stage(self, name: str) -> dict[str, Any]:
        """The state record of stage ``name`` (created as pending)."""
        return self.state['stages'].setdefault(name, {'state': 'pending'})

    def mark(self, name: str, state: str, **info: Any) -> None:
        """Set a stage's state (and any details), and save."""
        with self._lock:
            record = self.stage(name)
            record['state'] = state
            record.update(info)
            self._save_state()

    def run_dir(self, size: str) -> Path:
        """Training run directory for ``size``."""
        return self.out / 'runs' / size

    def hours_left(self) -> float:
        """Hours until the study's deadline."""
        return (self.state['deadline'] - time.time()) / 3600

    # -- training --------------------------------------------------------------

    def config_for(self, size: str) -> dict[str, Any]:
        """The run config for ``size``: ``full`` with the size's model."""
        data = json.loads((ROOT / 'configs' / 'full.json').read_text())
        shape = SIZES[size]
        every = shape['every']
        data['model'] = {
            k: shape[k] for k in ('blocks', 'channels', 'value_hidden')
        }
        data['train'].update(
            iterations=100_000,  # the time budget ends the run
            time_budget_hours=self.hours,
            checkpoint_every=every,
        )
        data['arena'] = {
            'every': every,
            'games': 30,
            'sims': 100,
            'opponents': ['ab:4', 'ab:6'],
        }
        if self.dry:
            data['self_play'].update(
                games_per_iteration=64,
                concurrent_games=64,
                sims=50,
                fast_sims=16,
            )
            data['train'].update(
                batch_size=256,
                warmup_steps=20,
                buffer_size=50_000,
                min_buffer=2_000,
                checkpoint_every=2,
            )
            data['arena'].update(every=2, games=4, opponents=['ab:2'])
        return data

    def train(self, size: str) -> None:
        """Train ``size`` to its budget, restarting it if it fails."""
        record = self.stage(size)
        if record['state'] == 'done':
            return
        run = self.run_dir(size)
        for attempt in range(ATTEMPTS):
            if self._finished(run):
                break
            if (run / 'config.json').exists():
                args = ['train.py', '--resume', str(run)]
                what = 'resuming'
            else:
                path = self.out / 'configs' / f'{size}.json'
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(self.config_for(size), indent=2))
                args = ['train.py', '--config', str(path), '--run', str(run)]
                what = 'starting'
            event(
                self.log,
                'RUN',
                f'{size}: {what} training (attempt {attempt + 1}/{ATTEMPTS}),'
                f' budget {self.hours:g} h',
            )
            self.mark(size, 'running', attempts=attempt + 1)
            code, why = self._supervise(size, args + ['--progress', 'off'])
            if code == 0 and self._finished(run):
                break
            note = f'{size}: training stopped ({why}); will resume'
            event(self.log, 'WARN', note)
            self.state['restarts'].append({'time': time.time(), 'note': note})
            self._save_state()
        if self._finished(run):
            self.mark(size, 'done', finished=time.time())
            event(self.log, 'RUN', f'{size}: training done')
        else:
            # keep going with what it has: its checkpoints still count
            self.mark(size, 'failed', finished=time.time())
            event(self.log, 'WARN', f'{size}: gave up after {ATTEMPTS} tries')

    def _finished(self, run: Path) -> bool:
        """Whether the run in ``run`` ended normally (budget reached)."""
        path = run / 'status.json'
        if not path.exists():
            return False
        status = json.loads(path.read_text(encoding='utf-8'))
        return status.get('phase') == 'finished'

    def _supervise(self, stage: str, args: list[str]) -> tuple[int, str]:
        """Run ``python args`` until it exits; kill it if it stalls or runs
        far over time. Returns its exit code and why it ended."""
        console = self.out / 'logs' / f'{stage}.console.log'
        cap = self.hours * 3600 * 1.25 + 1800  # wall-clock cap per attempt
        start = time.monotonic()
        with console.open('ab') as sink:
            proc = subprocess.Popen(
                [sys.executable, *args],
                cwd=ROOT,
                stdout=sink,
                stderr=subprocess.STDOUT,
            )
            with self._lock:
                self.current, self.current_stage = proc, stage
            why = 'exited'
            try:
                while proc.poll() is None:
                    time.sleep(10)
                    if self._stalled(stage):
                        why = 'no progress for too long'
                        kill_tree(proc)
                    elif time.monotonic() - start > cap:
                        why = 'over its wall-clock cap'
                        kill_tree(proc)
            finally:
                with self._lock:
                    self.current, self.current_stage = None, ''
        code = proc.returncode
        return code, f'{why}, exit code {code}'

    def _stalled(self, stage: str) -> bool:
        """Whether ``stage``'s output hasn't changed for ``STALL_LIMIT``."""
        if stage in SIZES:
            paths = [self.run_dir(stage) / 'status.json']
        else:
            paths = [self.out / 'logs' / f'{stage}.console.log']
        times = [p.stat().st_mtime for p in paths if p.exists()]
        if not times:
            return False
        # other stages only log now and then (a tournament chunk at a time)
        limit = STALL_LIMIT if stage in SIZES else 2 * STALL_LIMIT
        return time.time() - max(times) > limit

    # -- tournament and report -------------------------------------------------

    def plan(self) -> Path:
        """Write the tournament plan (see ``training.tournament``)."""
        entrants: list[dict[str, Any]] = []
        hours: dict[str, float] = {}
        sizes: dict[str, str] = {}
        finals: list[str] = []
        progress: dict[float, list[str]] = {}
        config = ''
        for size in ORDER:
            run = self.run_dir(size)
            points = checkpoints(run)
            if not points:
                continue
            config = str(run / 'config.json')
            final_iter, final_hours = points[-1]
            ckpt = str(run / 'checkpoints' / f'iter_{final_iter:04d}.pt')
            finals.append(size)
            entrants.append({'name': size, 'checkpoint': ckpt, 'sims': 200})
            entrants.append(
                {'name': f'{size} raw', 'checkpoint': ckpt, 'sims': 1}
            )
            hours[size] = hours[f'{size} raw'] = final_hours
            sizes[size] = sizes[f'{size} raw'] = size
            used = {final_iter}
            for f in CHECKPOINT_FRACTIONS:
                it, h = min(points, key=lambda p: abs(p[1] - f * final_hours))
                if it in used:
                    continue
                used.add(it)
                name = f'{size} @{h:.1f}h'
                path = str(run / 'checkpoints' / f'iter_{it:04d}.pt')
                entrants.append({'name': name, 'checkpoint': path, 'sims': 200})
                hours[name], sizes[name] = h, size
                progress.setdefault(f, []).append(name)
        abs_ = ['ab2', 'ab4', 'ab6', 'ab8']
        for name in abs_:
            entrants.append({'name': name, 'spec': f'ab:{name[2:]}'})

        def round_robin(names: list[str]) -> list[list[str]]:
            """Every pair from ``names``."""
            return [[a, b] for i, a in enumerate(names) for b in names[i + 1 :]]

        raws = [f'{s} raw' for s in finals]
        blocks = [
            {
                'name': 'finals',
                'pairs': round_robin(finals)
                + [[s, ab] for s in finals for ab in ('ab4', 'ab6', 'ab8')],
            },
            {
                'name': 'raw',
                'pairs': round_robin(raws)
                + [[r, ab] for r in raws for ab in ('ab2', 'ab4')]
                + [[f'{s} raw', s] for s in finals],
            },
            {
                'name': 'progress',
                'pairs': [
                    pair
                    for f in CHECKPOINT_FRACTIONS
                    for pair in round_robin(progress.get(f, []))
                    + [
                        [n, ab]
                        for n in progress.get(f, [])
                        for ab in ('ab4', 'ab6')
                    ]
                ],
            },
            {
                'name': 'progress-vs-final',
                'pairs': [
                    [n, sizes[n]]
                    for f in CHECKPOINT_FRACTIONS
                    for n in progress.get(f, [])
                ],
            },
            {
                'name': 'anchors',
                'pairs': [['ab2', 'ab4'], ['ab4', 'ab6'], ['ab6', 'ab8']],
            },
        ]
        limit = max(0.25, min(4.0, self.hours_left() - 0.3))
        plan = {
            'config': config,
            'entrants': entrants,
            'blocks': blocks,
            'openings': 2 if self.dry else 30,
            'anchor': 'ab4',
            'chunk_games': 300,
            'time_limit_hours': 0.15 if self.dry else limit,
            'extra': {'hours': hours, 'sizes': sizes},
        }
        path = self.out / 'tournament' / 'plan.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(plan, indent=2))
        return path

    def tournament(self) -> None:
        """Play the tournament (resuming one already under way)."""
        if self.stage('tournament')['state'] == 'done':
            return
        plan = self.out / 'tournament' / 'plan.json'
        if not plan.exists():
            plan = self.plan()
        event(self.log, 'RUN', f'tournament: playing {plan}')
        self.mark('tournament', 'running')
        code, why = self._supervise(
            'tournament',
            [
                '-m',
                'training.tournament',
                str(plan),
                str(self.out / 'tournament'),
            ],
        )
        state = 'done' if code == 0 else 'failed'
        self.mark('tournament', state, finished=time.time())
        tag = 'RUN' if code == 0 else 'WARN'
        event(self.log, tag, f'tournament {state} ({why})')

    def report(self) -> None:
        """Write REPORT.md and the plots."""
        self.mark('report', 'running')
        code, why = self._supervise(
            'report', ['-m', 'experiments.report', str(self.out)]
        )
        state = 'done' if code == 0 else 'failed'
        self.mark('report', state, finished=time.time())
        tag = 'RUN' if code == 0 else 'WARN'
        event(self.log, tag, f'report {state} ({why}): {self.out}/REPORT.md')

    # -- monitoring ------------------------------------------------------------

    def monitor(self) -> None:
        """Every ``STATUS_EVERY`` s: record resources, refresh STATUS.md."""
        while not self._stop.wait(STATUS_EVERY):
            try:
                self.snapshot()
            except Exception as error:  # never let monitoring kill a study
                event(self.log, 'WARN', f'status update failed: {error!r}')

    def snapshot(self) -> None:
        """Record one resource reading and rewrite STATUS.md."""
        with self._lock:
            proc, stage = self.current, self.current_stage
        memory = psutil.virtual_memory()
        reading: dict[str, Any] = {
            'time': time.time(),
            'stage': stage,
            'system_used_mb': round(memory.used / 2**20),
            'system_total_mb': round(memory.total / 2**20),
            'stage_rss_mb': round(tree_rss_mb(proc.pid)) if proc else 0,
            'disk_free_gb': round(shutil.disk_usage(self.out).free / 2**30),
            **commit_reading(),
            **gpu_reading(),
        }
        fileio.append_jsonl(self.out / 'resources.jsonl', reading)
        event(
            self.log,
            'RUN',
            f'resources [{stage or "idle"}]: '
            f'RAM {reading["system_used_mb"] / 1024:.1f}/'
            f'{reading["system_total_mb"] / 1024:.1f} GB '
            f'(stage {reading["stage_rss_mb"] / 1024:.1f} GB), '
            f'commit {reading.get("commit_mb", 0) / 1024:.1f}/'
            f'{reading.get("commit_limit_mb", 0) / 1024:.1f} GB, '
            f'GPU {reading.get("gpu_mb", 0) / 1024:.1f} GB '
            f'{reading.get("gpu_util", 0):.0f}% '
            f'{reading.get("gpu_temp", 0):.0f}C, '
            f'disk {reading["disk_free_gb"]} GB free',
            to='file',
        )
        fileio.write_text(self.out / 'STATUS.md', self.status_text(reading))

    def status_text(self, reading: dict[str, Any]) -> str:
        """STATUS.md: where the study is and whether it's healthy."""
        now = time.time()
        started = self.state['started']
        stamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        deadline = datetime.fromtimestamp(self.state['deadline'])
        lines = [
            '# Network-size study: status',
            '',
            f'Updated {stamp}; running {(now - started) / 3600:.1f} h; '
            f'expected to finish by {deadline:%Y-%m-%d %H:%M}.',
            '',
            '| stage | state | progress |',
            '|---|---|---|',
        ]
        for size in ORDER:
            state = self.stage(size)['state']
            lines.append(f'| {size} | {state} | {self._run_note(size)} |')
        for name in ('tournament', 'report'):
            lines.append(f'| {name} | {self.stage(name)["state"]} | |')
        gpu = (
            f'{reading["gpu_mb"] / 1024:.1f}/'
            f'{reading["gpu_total_mb"] / 1024:.1f} GB, '
            f'{reading["gpu_util"]:.0f}% busy, {reading["gpu_temp"]:.0f} C'
            if 'gpu_mb' in reading
            else 'unreadable'
        )
        lines += [
            '',
            '## Resources',
            '',
            f'- RAM: {reading["system_used_mb"] / 1024:.1f} of '
            f'{reading["system_total_mb"] / 1024:.1f} GB used; current stage '
            f'{reading["stage_rss_mb"] / 1024:.1f} GB',
            f'- Commit charge (RAM + page file promised to programs; '
            f'allocations fail at the limit): '
            f'{reading.get("commit_mb", 0) / 1024:.1f} of '
            f'{reading.get("commit_limit_mb", 0) / 1024:.1f} GB',
            f'- GPU: {gpu}',
            f'- Disk: {reading["disk_free_gb"]} GB free',
            f'- Memory trend of the current stage: {self._trend()}',
            '',
            '## Problems',
            '',
        ]
        problems = [
            f'- {datetime.fromtimestamp(r["time"]):%H:%M} {r["note"]}'
            for r in self.state['restarts']
        ]
        for size in ORDER:
            log = self.run_dir(size) / 'train.log'
            if log.exists():
                warns = log.read_text(encoding='utf-8').count('WARN')
                if warns:
                    problems.append(f'- {size}: {warns} warnings in train.log')
        limit = reading.get('commit_limit_mb', 0)
        if limit and reading['commit_mb'] > 0.9 * limit:
            problems.append(
                f'- commit charge at {reading["commit_mb"] / limit:.0%} of '
                'its limit: close other programs (allocations fail at 100%)'
            )
        lines += problems or ['- none']
        return '\n'.join(lines) + '\n'

    def _run_note(self, size: str) -> str:
        """One line on a training run: time, iterations, speed, strength."""
        run = self.run_dir(size)
        rows = fileio.read_jsonl(run / 'metrics.jsonl')
        if not rows:
            return ''
        last = rows[-1]
        note = (
            f'{last.get("elapsed_hours", 0):.2f}/{self.hours:g} h, '
            f'iter {last["iteration"] + 1}, {last["seconds"]:.0f} s/iter, '
            f'{last["sims_per_second"] / 1e3:.0f}k sims/s, '
            f'RAM {last["memory"]["process_mb"] / 1024:.1f}+'
            f'{last["memory"]["workers_mb"] / 1024:.1f} GB'
        )
        arena = fileio.read_jsonl(run / 'arena.jsonl')
        if arena:
            scores = ', '.join(
                f'{r["opponent"]} {r["score"]:.2f}'
                for r in arena[-1]['results']
                if r['opponent'].startswith('ab')
            )
            note += f'; Elo chain {arena[-1]["elo"]:.0f}, vs {scores}'
        return note

    def _trend(self) -> str:
        """Memory growth of the current stage over its last 2 hours."""
        rows = fileio.read_jsonl(self.out / 'resources.jsonl')
        current = rows[-1]['stage'] if rows else ''
        rows = [
            r
            for r in rows
            if r['stage'] == current
            and r['stage_rss_mb'] > 0
            and r['time'] > time.time() - 7200
        ]
        if len(rows) < 6:
            return 'not enough readings yet'
        # compare the first and last thirds, so start-up doesn't count
        third = len(rows) // 3
        first = sum(r['stage_rss_mb'] for r in rows[:third]) / third
        last = sum(r['stage_rss_mb'] for r in rows[-third:]) / third
        span = (rows[-1]['time'] - rows[0]['time']) / 3600 * 2 / 3
        rate = (last - first) / max(span, 1e-6)
        return f'{rate:+.0f} MB/h (buffer filling explains early growth)'

    # -- the whole study -------------------------------------------------------

    def run(self) -> None:
        """Train every size, play the tournament, write the report."""
        keep_awake()
        end = datetime.fromtimestamp(self.state['deadline'])
        event(
            self.log,
            'RUN',
            f'size study in {self.out}: {", ".join(ORDER)} for '
            f'{self.hours:g} h each, then tournament and report; '
            f'everything done by {end:%Y-%m-%d %H:%M}',
        )
        watcher = threading.Thread(target=self.monitor, daemon=True)
        watcher.start()
        self.snapshot()
        try:
            for size in ORDER:
                self.train(size)
            self.tournament()
            self.report()
        finally:
            self._stop.set()
            self.snapshot()
        took = timedelta(seconds=round(time.time() - self.state['started']))
        event(self.log, 'RUN', f'size study finished after {took}')


def checkpoints(run: Path) -> list[tuple[int, float]]:
    """``(iteration, hours of training)`` of each weights checkpoint."""
    hours = {
        r['iteration']: r.get('elapsed_hours', 0.0)
        for r in fileio.read_jsonl(run / 'metrics.jsonl')
    }
    points = []
    for path in sorted((run / 'checkpoints').glob('iter_*.pt')):
        it = int(path.stem.split('_')[1])
        if it in hours:
            points.append((it, hours[it]))
    return points


def main() -> None:
    """Command line: start or resume the study."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    parser.add_argument(
        '--out', type=Path, default=ROOT / 'runs' / 'size_study'
    )
    parser.add_argument('--hours', type=float, default=4.25)
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()
    if args.dry:
        args.hours = 0.08
        if args.out == ROOT / 'runs' / 'size_study':
            args.out = ROOT / 'runs' / 'size_study_dry'
    Study(args.out, args.hours, args.dry).run()


if __name__ == '__main__':
    main()
