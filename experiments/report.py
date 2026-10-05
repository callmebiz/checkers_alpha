"""Report on a network-size study: ``REPORT.md`` plus plots.

Reads everything ``experiments.size_study`` leaves in its directory (the
training runs, the tournament, resource readings) and writes
``REPORT.md`` and ``figures/*.png`` next to them. Safe to rerun at any
time, including on a study still in progress::

    python -m experiments.report runs/size_study
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core.model import CheckersNet, ModelConfig  # noqa: E402
from training import fileio  # noqa: E402

COLORS = {'10x128': 'tab:blue', '8x96': 'tab:orange', '6x64': 'tab:green'}
MB_PER_HOUR_ALARM = 100.0  # memory growth worth flagging, late in a run


class Run:
    """One training run's records."""

    def __init__(self, size: str, root: Path) -> None:
        """Load the run in ``root``."""
        self.size = size
        self.root = root
        self.metrics = fileio.read_jsonl(root / 'metrics.jsonl')
        self.arena = fileio.read_jsonl(root / 'arena.jsonl')
        self.probes = fileio.read_jsonl(root / 'probes.jsonl')
        self.config = json.loads(
            (root / 'config.json').read_text(encoding='utf-8')
        )
        self.hours_at = {
            r['iteration']: r.get('elapsed_hours', 0.0) for r in self.metrics
        }
        self.games_at: dict[int, int] = {}
        total = 0
        for r in self.metrics:
            total += r['self_play']['games']
            self.games_at[r['iteration']] = total

    def series(self, get: Any) -> tuple[np.ndarray, np.ndarray]:
        """(hours, values) of ``get(row)`` over the metrics rows."""
        hours = [r.get('elapsed_hours', 0.0) for r in self.metrics]
        return np.array(hours), np.array([get(r) for r in self.metrics])

    def backend(self) -> str:
        """The inference backend the run ended up using."""
        log = self.root / 'train.log'
        if not log.exists():
            return '?'
        found = re.findall(
            r'inference backend: (.*)', log.read_text(encoding='utf-8')
        )
        return found[-1].strip() if found else '?'

    def warnings(self) -> list[str]:
        """WARN lines from the run's log."""
        log = self.root / 'train.log'
        if not log.exists():
            return []
        return [
            line.strip()
            for line in log.read_text(encoding='utf-8').splitlines()
            if 'WARN' in line
        ]


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over ``window`` points (shorter at the start)."""
    if window <= 1 or len(values) == 0:
        return values
    # missing points (NaN, e.g. losses before training starts) are left
    # out of the means rather than spreading through the running sum
    values = values.astype(float)
    known = ~np.isnan(values)
    sums = np.cumsum(np.insert(np.where(known, values, 0.0), 0, 0.0))
    counts = np.cumsum(np.insert(known, 0, False).astype(int))
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        lo = max(0, i + 1 - window)
        n = counts[i + 1] - counts[lo]
        if n:
            out[i] = (sums[i + 1] - sums[lo]) / n
    return out


def growth(hours: np.ndarray, mb: np.ndarray) -> float | None:
    """Memory growth in MB/h (fitted line), or None if too few points.

    Pass only readings from after the replay buffer filled: before that,
    growth is expected.
    """
    if len(hours) < 6 or np.ptp(hours) <= 0:
        return None
    return float(np.polyfit(hours, mb, 1)[0])


def parameters(model: dict[str, int]) -> int:
    """Number of weights of a network shape."""
    net = CheckersNet(ModelConfig(**model))
    return sum(p.numel() for p in net.parameters())


def save(fig: Any, out: Path, name: str) -> str:
    """Save a figure as ``figures/name.png``; return its relative path."""
    (out / 'figures').mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / 'figures' / f'{name}.png', dpi=110)
    plt.close(fig)
    return f'figures/{name}.png'


def table(header: list[str], rows: list[list[Any]]) -> list[str]:
    """A markdown table."""
    lines = ['| ' + ' | '.join(header) + ' |']
    lines.append('|' + '---|' * len(header))
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return lines


# -- sections ------------------------------------------------------------------


def setup_section(runs: list[Run], state: dict[str, Any]) -> list[str]:
    """What was trained, for how long, how fast."""
    rows = []
    for run in runs:
        m = run.metrics
        if not m:
            continue
        sp = [r['self_play'] for r in m]
        lookups = sum(s.get('cache_lookups', 0) for s in sp)
        hits = sum(s.get('cache_hits', 0) for s in sp)
        steady = m[3:] or m  # past the first iterations' start-up costs
        rate = np.mean([r['sims_per_second'] for r in steady]) / 1e3
        rows.append(
            [
                run.size,
                f'{parameters(run.config["model"]) / 1e6:.2f} M',
                f'{m[-1].get("elapsed_hours", 0):.2f}',
                len(m),
                f'{run.games_at[m[-1]["iteration"]]:,}',
                f'{sum(r["new_samples"] for r in m):,}',
                f'{m[-1]["step"]:,}',
                f'{np.mean([r["seconds"] for r in steady]):.0f}',
                f'{rate:.0f}k',
                f'{100 * hits / lookups:.0f}%' if lookups else '-',
                run.backend(),
            ]
        )
    started = datetime.fromtimestamp(state.get('started', 0))
    sp_cfg = runs[0].config['self_play'] if runs else {}
    tr_cfg = runs[0].config['train'] if runs else {}
    lines = [
        '## Setup',
        '',
        f'Started {started:%Y-%m-%d %H:%M}. Each size trained for '
        f'{state.get("hours", 0):g} h of wall-clock time (self-play, '
        'training and in-run evaluations all count), with the learning '
        'rate annealed over those hours. Search and hyperparameters are '
        f'the same for all: {sp_cfg.get("sims")}/{sp_cfg.get("fast_sims")} '
        f'simulations, {sp_cfg.get("games_per_iteration")} games per '
        f'iteration, batch {tr_cfg.get("batch_size")}, lr '
        f'{tr_cfg.get("lr")}, buffer {tr_cfg.get("buffer_size"):,}.',
        '',
    ]
    lines += table(
        [
            'size',
            'weights',
            'hours',
            'iterations',
            'games',
            'samples',
            'steps',
            's/iter',
            'sims/s',
            'cache hits',
            'backend',
        ],
        rows,
    )
    return lines + ['']


def tournament_section(
    out: Path, runs: list[Run]
) -> tuple[list[str], list[str]]:
    """Ratings on one scale; returns (lines, findings)."""
    path = out / 'tournament' / 'ratings.json'
    plan_path = out / 'tournament' / 'plan.json'
    if not path.exists() or not plan_path.exists():
        return ['## Tournament', '', 'Not played (yet).', ''], []
    ratings = json.loads(path.read_text(encoding='utf-8'))
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    hours = plan['extra']['hours']
    r = ratings['ratings']
    lines = [
        '## Tournament',
        '',
        f'{ratings["games"]:,} games, every pairing from the same random '
        'openings with both colours. Bradley-Terry Elo with '
        f'{ratings["anchor"]} = 0; brackets are 90% bootstrap intervals. '
        'Networks search 200 simulations per move; "raw" plays the '
        "network's top move with no search.",
        '',
    ]
    lines += table(
        ['player', 'Elo', '90% interval', 'games', 'trained (h)'],
        [
            [
                name,
                f'{v["elo"]:.0f}',
                f'{v["low"]:.0f} to {v["high"]:.0f}',
                v['games'],
                f'{hours[name]:.2f}' if name in hours else '',
            ]
            for name, v in r.items()
        ],
    )

    # Elo against training hours, on the tournament scale
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run in runs:
        points = sorted(
            (hours[n], r[n])
            for n in r
            if n in hours
            and plan['extra']['sizes'].get(n) == run.size
            and not n.endswith('raw')
        )
        if not points:
            continue
        x = [p[0] for p in points]
        ax.errorbar(
            x,
            [p[1]['elo'] for p in points],
            yerr=[
                [p[1]['elo'] - p[1]['low'] for p in points],
                [p[1]['high'] - p[1]['elo'] for p in points],
            ],
            marker='o',
            capsize=3,
            color=COLORS.get(run.size),
            label=run.size,
        )
    for ab in ('ab2', 'ab4', 'ab6', 'ab8'):
        if ab in r:
            ax.axhline(r[ab]['elo'], color='grey', lw=0.8, ls='--')
            # x in axes fraction, y in data: labels at the left edge
            ax.text(
                0.01,
                r[ab]['elo'],
                ab,
                va='bottom',
                fontsize=8,
                transform=ax.get_yaxis_transform(),
            )
    ax.set_xlabel('hours of training')
    ax.set_ylabel(f'Elo ({ratings["anchor"]} = 0)')
    ax.set_title('Tournament Elo against training time')
    ax.legend()
    lines += ['', f'![Elo vs hours]({save(fig, out, "tournament_elo")})', '']

    # head to head among the finals and alpha-beta
    sizes = [run.size for run in runs if run.size in r]
    players = sizes + [a for a in ('ab4', 'ab6', 'ab8') if a in r]
    pairs = {(p['a'], p['b']): p for p in ratings['pairs']}

    def cell(a: str, b: str) -> str:
        """``a``'s score against ``b`` as a percentage (games)."""
        if a == b:
            return '-'
        if (a, b) in pairs:
            p = pairs[a, b]
            return f'{100 * p["score"] / p["games"]:.0f}% ({p["games"]})'
        if (b, a) in pairs:
            p = pairs[b, a]
            s = p['games'] - p['score']
            return f'{100 * s / p["games"]:.0f}% ({p["games"]})'
        return ''

    lines += ["Head to head (row player's score, games in brackets):", '']
    lines += table(
        ['', *players], [[a, *(cell(a, b) for b in players)] for a in sizes]
    )

    findings = []
    finals = sorted(((s, r[s]) for s in sizes), key=lambda kv: -kv[1]['elo'])
    if finals:
        best, rest = finals[0], finals[1:]
        clear = [s for s, v in rest if v['high'] < best[1]['low']]
        verdict = (
            'clearly ahead of ' + ', '.join(clear)
            if clear
            else 'not clearly ahead (intervals overlap)'
        )
        findings.append(
            f'Strongest after equal training time: **{best[0]}** at '
            f'{best[1]["elo"]:.0f} Elo, {verdict}. Order: '
            + ' > '.join(f'{s} ({v["elo"]:.0f})' for s, v in finals)
            + '.'
        )
        for s, v in finals:
            raw = r.get(f'{s} raw')
            if raw:
                findings.append(
                    f'{s}: search (200 sims) is worth '
                    f'{v["elo"] - raw["elo"]:+.0f} Elo over the raw network '
                    f'({raw["elo"]:.0f}).'
                )
        for ab in ('ab6', 'ab8'):
            if ab in r:
                above = [s for s, v in finals if v['elo'] > r[ab]['elo']]
                findings.append(
                    f'Above alpha-beta depth {ab[2:]}: '
                    + (', '.join(above) if above else 'none')
                    + '.'
                )
    return lines + [''], findings


def arena_section(out: Path, runs: list[Run]) -> list[str]:
    """In-run evaluations: scores against alpha-beta, the Elo chain."""
    opponents = sorted(
        {
            res['opponent']
            for run in runs
            for row in run.arena
            for res in row['results']
            if res['opponent'].startswith('ab')
        }
    )
    if not opponents:
        return []
    fig, axes = plt.subplots(
        1, len(opponents) + 1, figsize=(5 * (len(opponents) + 1), 4)
    )
    axes = np.atleast_1d(axes)
    for run in runs:
        hours = [run.hours_at.get(row['iteration'], 0) for row in run.arena]
        for ax, opp in zip(axes, opponents, strict=False):
            scores = [
                next(
                    (
                        x['score']
                        for x in row['results']
                        if x['opponent'] == opp
                    ),
                    np.nan,
                )
                for row in run.arena
            ]
            ax.plot(
                hours,
                scores,
                marker='.',
                label=run.size,
                color=COLORS.get(run.size),
            )
            ax.set_title(f'score vs {opp}')
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel('hours')
        axes[-1].plot(
            hours,
            [row['elo'] for row in run.arena],
            marker='.',
            label=run.size,
            color=COLORS.get(run.size),
        )
    axes[-1].set_title('Elo chain (vs previous checkpoint)')
    axes[-1].set_xlabel('hours')
    axes[0].legend()
    lines = [
        '## Evaluations during training',
        '',
        'Every ~15 minutes each run played alpha-beta and its previous '
        'evaluated checkpoint (100 simulations, few games, so noisy). The '
        'Elo chain adds up checkpoint-to-checkpoint differences: it drifts, '
        'and its absolute level is not comparable between runs; the '
        'tournament above is.',
        '',
        f'![in-run arena]({save(fig, out, "arena")})',
        '',
    ]
    rows = []
    for run in runs:
        if not run.arena:
            continue
        last = run.arena[-1]
        rows.append(
            [run.size, last['iteration'] + 1]
            + [
                next(
                    (
                        f'{x["wins"]}-{x["draws"]}-{x["losses"]}'
                        for x in last['results']
                        if x['opponent'] == opp
                    ),
                    '',
                )
                for opp in opponents
            ]
        )
    lines += ['Last evaluation (wins-draws-losses):', '']
    lines += table(['size', 'iteration', *opponents], rows)
    return lines + ['']


def curves_section(out: Path, runs: list[Run]) -> list[str]:
    """Training losses and self-play statistics against hours."""
    panels = [
        ('policy loss', lambda r: r.get('policy_loss', np.nan)),
        ('value loss', lambda r: r.get('value_loss', np.nan)),
        ('value accuracy', lambda r: r.get('value_accuracy', np.nan)),
        ('policy entropy', lambda r: r.get('policy_entropy', np.nan)),
        (
            'self-play draw rate',
            lambda r: r['self_play']['draws'] / max(r['self_play']['games'], 1),
        ),
        ('average plies', lambda r: r['avg_plies']),
        ('simulations / s (k)', lambda r: r['sims_per_second'] / 1e3),
        (
            'cache hit rate',
            lambda r: (
                r['self_play'].get('cache_hits', 0)
                / max(r['self_play'].get('cache_lookups', 0), 1)
            ),
        ),
        ('learning rate', lambda r: r['lr']),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, (title, get) in zip(axes.flat, panels, strict=True):
        for run in runs:
            x, y = run.series(get)
            ax.plot(
                x,
                smooth(y, max(1, len(y) // 40)),
                label=run.size,
                color=COLORS.get(run.size),
            )
        ax.set_title(title)
        ax.set_xlabel('hours')
    axes.flat[0].legend()
    return [
        '## Training curves',
        '',
        'Smoothed over ~2.5% of each run. Losses are on the current replay '
        'buffer, which a faster run refreshes more often, so compare their '
        'shapes more than their levels.',
        '',
        f'![training curves]({save(fig, out, "curves")})',
        '',
    ]


def probes_section(out: Path, runs: list[Run]) -> list[str]:
    """What the networks make of the probe positions over time."""
    names: list[str] = []
    for run in runs:
        for row in run.probes[-1:]:
            names += [n for n in row['probes'] if n not in names]
    if not names:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for run in runs:
        hours = [run.hours_at.get(row['iteration'], 0) for row in run.probes]
        solved = [
            np.mean(
                [
                    float(bool(p['correct']))
                    for p in row['probes'].values()
                    if p.get('correct') is not None
                ]
                or [np.nan]
            )
            for row in run.probes
        ]
        start = [
            row['probes'].get('start', {}).get('value', np.nan)
            for row in run.probes
        ]
        n = max(1, len(hours) // 40)
        axes[0].plot(
            hours,
            smooth(np.array(solved), n),
            label=run.size,
            color=COLORS.get(run.size),
        )
        axes[1].plot(
            hours,
            smooth(np.array(start), n),
            label=run.size,
            color=COLORS.get(run.size),
        )
    axes[0].set_title('share of probes with a known answer solved')
    axes[1].set_title('value of the start position (side to move)')
    for ax in axes:
        ax.set_xlabel('hours')
    axes[0].legend()
    rows = []
    for name in names:
        row = [name]
        for run in runs:
            last = run.probes[-1]['probes'].get(name) if run.probes else None
            if last is None:
                row.append('')
                continue
            mark = {True: ' ok', False: ' wrong', None: ''}[last['correct']]
            row.append(f'{last["best"]} ({last["value"]:+.2f}){mark}')
        rows.append(row)
    lines = [
        '## Probe positions',
        '',
        f'![probes]({save(fig, out, "probes")})',
        '',
        'Final answers (best move after search, network value):',
        '',
    ]
    lines += table(['probe', *(run.size for run in runs)], rows)
    return lines + ['']


def health_section(
    out: Path, runs: list[Run], state: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Memory over time, restarts, warnings; returns (lines, alarms)."""
    alarms = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    rows = []
    for run in runs:
        if not run.metrics:
            continue
        for ax, key, title in zip(
            axes,
            ('process_mb', 'workers_mb', 'cuda_reserved_mb'),
            (
                'trainer process RAM (MB)',
                'workers RAM (MB)',
                'GPU reserved by PyTorch (MB)',
            ),
            strict=True,
        ):
            x, y = run.series(lambda r, k=key: r['memory'].get(k, np.nan))
            ax.plot(x, y, label=run.size, color=COLORS.get(run.size))
            ax.set_title(title)
            ax.set_xlabel('hours')
        mem = [r['memory'] for r in run.metrics]
        x, proc = run.series(lambda r: r['memory']['process_mb'])
        _, workers = run.series(lambda r: r['memory'].get('workers_mb', 0))
        _, buffer = run.series(lambda r: r['buffer_size'])
        full = buffer >= run.config['train']['buffer_size']
        rate = growth(x[full], (proc + workers)[full])
        if rate is not None and rate > MB_PER_HOUR_ALARM:
            alarms.append(
                f'{run.size}: RAM grew {rate:.0f} MB/h after the replay '
                'buffer filled - look at figures/memory.png.'
            )
        late = 'buffer never filled' if rate is None else f'{rate:+.0f}'
        rows.append(
            [
                run.size,
                f'{mem[0]["process_mb"]:,} -> {mem[-1]["process_mb"]:,}',
                f'{workers[0]:,.0f} -> {workers[-1]:,.0f}',
                f'{max(m.get("cuda_reserved_mb", 0) for m in mem):,}',
                late,
                len(run.warnings()),
            ]
        )
    axes[0].legend()
    lines = [
        '## Health',
        '',
        f'![memory]({save(fig, out, "memory")})',
        '',
    ]
    lines += table(
        [
            'size',
            'trainer RAM (MB)',
            'workers RAM (MB)',
            'peak GPU reserved (MB)',
            'growth once full (MB/h)',
            'warnings',
        ],
        rows,
    )
    resources = fileio.read_jsonl(out / 'resources.jsonl')
    if resources:
        temps = [r['gpu_temp'] for r in resources if 'gpu_temp' in r]
        used = [r['system_used_mb'] for r in resources]
        commits = [
            r['commit_mb'] / r['commit_limit_mb']
            for r in resources
            if r.get('commit_limit_mb')
        ]
        lines += [
            '',
            f'Whole study ({len(resources)} readings): system RAM peaked at '
            f'{max(used) / 1024:.1f} GB'
            + (
                f', commit charge at {max(commits):.0%} of its limit'
                if commits
                else ''
            )
            + (f', GPU temperature at {max(temps):.0f} C' if temps else '')
            + f'; {resources[-1]["disk_free_gb"]} GB disk left.',
        ]
    restarts = state.get('restarts', [])
    lines += ['', f'Restarts: {len(restarts)}.']
    for r in restarts:
        lines.append(f'- {datetime.fromtimestamp(r["time"]):%H:%M} {r["note"]}')
        alarms.append(f'restart: {r["note"]}')
    for run in runs:
        for w in run.warnings()[:10]:
            lines.append(f'- {run.size}: {w}')
    return lines + [''], alarms


def write_report(out: Path) -> Path:
    """Build REPORT.md and the figures for the study in ``out``."""
    state_path = out / 'state.json'
    state = (
        json.loads(state_path.read_text(encoding='utf-8'))
        if state_path.exists()
        else {}
    )
    runs = [
        Run(p.name, p)
        for p in sorted((out / 'runs').glob('*'))
        if (p / 'config.json').exists()
    ]
    order = list(COLORS)
    runs.sort(key=lambda r: order.index(r.size) if r.size in order else 99)

    tournament, findings = tournament_section(out, runs)
    health, alarms = health_section(out, runs, state)
    lines = [
        '# Network-size study',
        '',
        f'Generated {datetime.now():%Y-%m-%d %H:%M}.',
        '',
        '## Findings',
        '',
    ]
    lines += [f'- {f}' for f in findings] or ['- (no tournament results yet)']
    lines += [f'- **Check:** {a}' for a in alarms]
    lines.append('')
    lines += setup_section(runs, state)
    lines += tournament
    lines += arena_section(out, runs)
    lines += curves_section(out, runs)
    lines += probes_section(out, runs)
    lines += health
    path = out / 'REPORT.md'
    fileio.write_text(path, '\n'.join(lines) + '\n')
    return path


def main() -> None:
    """Command line: write the report for a study directory."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    parser.add_argument('study', type=Path)
    args = parser.parse_args()
    print(write_report(args.study))


if __name__ == '__main__':
    main()
