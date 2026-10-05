"""Run logs: one standard layout for ``train.log`` and the console.

Every entry starts with a timestamp and a fixed-width tag saying what it is;
continuation lines are indented to line up under the message::

    2026-09-30 17:33:52  ITER    2/60 in 12.0 s  (self-play 11.5, train 0.5)
                                 games     257  black 99  white 123  draws 35
                                 ...

Tags:

    RUN      run start (with a summary of the settings), finish, stop
    ITER     one iteration: games, search speed, buffer, losses, memory
    PROBES   the probe positions: best move, expected move, values
    SAVE     a checkpoint was written
    ARENA    an evaluation: results per opponent and the Elo rating
    WARN     something needs attention but training continues

Where entries go: ``train.log`` gets the full reports. The console gets
short one-line versions (or none, while progress bars show the same
information), so it stays readable during a long run. ``event(...,
to=...)`` picks the destination: ``'both'``, ``'file'`` or ``'console'``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # types only: importing the configs pulls in torch
    from training.config import RunConfig, TrainConfig

TAG_WIDTH = 7


def event(
    log: logging.Logger, tag: str, message: str, to: str = 'both'
) -> None:
    """Log ``message`` under ``tag`` to the file, the console or both."""
    log.info(message, extra={'tag': tag, 'to': to})


class BlockFormatter(logging.Formatter):
    """``<time>  <TAG>  message``, with extra lines aligned under the
    message."""

    def __init__(self, datefmt: str) -> None:
        """Use ``datefmt`` for the timestamp (full date in the file, time
        only on the console)."""
        super().__init__(datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Render one entry, indenting continuation lines."""
        tag = getattr(record, 'tag', 'WARN' if record.levelno > 20 else 'INFO')
        prefix = f'{self.formatTime(record, self.datefmt)}  {tag:<{TAG_WIDTH}} '
        lines = record.getMessage().split('\n')
        pad = ' ' * len(prefix)
        return '\n'.join([prefix + lines[0]] + [pad + ln for ln in lines[1:]])


class Destination(logging.Filter):
    """Lets through the entries meant for one destination."""

    def __init__(self, name: str) -> None:
        """``name`` is ``'file'`` or ``'console'``."""
        super().__init__()
        self.destination = name

    def filter(self, record: logging.LogRecord) -> bool:
        """Whether ``record`` is meant for this destination."""
        return getattr(record, 'to', 'both') in ('both', self.destination)


# -- reports -------------------------------------------------------------------


def _z(value: float) -> float:
    """``value`` with a rounded-away ``-0`` shown as ``+0``."""
    return value if abs(value) >= 0.005 else 0.0


def _cache_note(sp: dict[str, Any]) -> str:
    """``, cache hits 23%`` when the evaluation cache was used."""
    lookups = sp.get('cache_lookups', 0)
    if not lookups:
        return ''
    return f', cache hits {sp["cache_hits"] / lookups:.0%}'


def _mb(value: float) -> str:
    """A memory amount, e.g. ``1,641 MB``."""
    return f'{value:,.0f} MB'


def run_header(config: RunConfig, root: str, device: str, start: int) -> str:
    """The settings that matter most, at the start of a run (or resume)."""
    m, sp, t, a = config.model, config.self_play, config.train, config.arena
    first = start + 1
    what = 'start' if start == 0 else f'resume at iteration {first}'
    workers = f'{sp.workers} workers' if sp.workers else 'in-process'
    arena = (
        f'every {a.every} iterations vs {", ".join(a.opponents)} and the '
        f'previous checkpoint; {a.games} games each, {a.sims} sims'
        if a.every
        else 'off'
    )
    return '\n'.join(
        [
            f'{what}: {root}, {planned(t)} on {device}',
            f'model     {m.blocks} blocks x {m.channels} channels, '
            f'value head {m.value_hidden}',
            f'self-play {sp.games_per_iteration} games per iteration, '
            f'{workers}, {sp.concurrent_games} in progress',
            f'search    {sp.sims} sims on {sp.full_search_prob:.0%} of moves '
            f'(recorded), {sp.fast_sims} on the rest',
            f'train     batch {t.batch_size}, lr {_schedule(t)}, '
            f'buffer {t.buffer_size:,}, each sample used ~{t.sample_reuse:g}x',
            f'arena     {arena}',
        ]
    )


def _schedule(t: TrainConfig) -> str:
    """The learning-rate schedule in a few words."""
    if t.lr_schedule == 'step':
        return (
            f'{t.lr:g} x{t.lr_step_factor:g} every {t.lr_step_hours:g} h '
            f'(floor {t.lr_min:g})'
        )
    return f'{t.lr:g} -> {t.lr_min:g}'


def of_total(done: int, total: int) -> str:
    """Progress as ``done/total``.

    Just ``done`` when ``total`` is 0: a time budget ends the run, so the
    iteration count means nothing.
    """
    return f'{done}/{total}' if total > 0 else str(done)


def planned(t: TrainConfig) -> str:
    """How long a run is set to go: iterations, or a time budget."""
    if t.time_budget_hours > 0:
        return f'{t.time_budget_hours:g} h of training'
    return f'{t.iterations} iterations'


def iteration_report(row: dict[str, Any], config: RunConfig) -> str:
    """The full ``ITER`` entry for one metrics row."""
    sp, t = row['self_play'], config.train
    games = max(sp['games'], 1)
    batch = sp['evaluated'] / max(sp['batches'], 1)
    total = 0 if t.time_budget_hours > 0 else t.iterations
    budget = ''
    if t.time_budget_hours > 0 and 'elapsed_hours' in row:
        budget = f'  [{row["elapsed_hours"]:.2f} of {t.time_budget_hours:g} h]'
    lines = [
        f'{of_total(row["iteration"] + 1, total)} in {row["seconds"]:.1f} s  '
        f'(self-play {sp["seconds"]:.1f}, train {row["train_seconds"]:.1f}, '
        f'probes {row["probe_seconds"]:.1f}){budget}',
        f'games     {sp["games"]:<5}  black {sp["black_wins"]:<4} '
        f'white {sp["white_wins"]:<4} draws {sp["draws"]:<4} '
        f'({sp["draws"] / games:.0%})',
        f'length    {row["avg_plies"]:.1f} plies on average',
        f'search    {sp["sims"]:,} sims at {row["sims_per_second"]:,}/s, '
        f'{sp["batches"]:,} batches of {batch:.0f}' + _cache_note(sp),
        f'buffer    +{row["new_samples"]:,} -> {row["buffer_size"]:,} of '
        f'{t.buffer_size:,}  ({row["buffer_total"]:,} added in total)',
    ]
    if row['train_steps']:
        lines += [
            f'train     {row["train_steps"]} steps (total {row["step"]:,}), '
            f'lr {row["lr"]:.2e}, grad norm {row["grad_norm"]:.3f}',
            f'loss      policy {row["policy_loss"]:.4f}  '
            f'kl {row["policy_kl"]:.4f}  entropy {row["policy_entropy"]:.3f}',
            f'          value  {row["value_loss"]:.4f}  '
            f'accuracy {row["value_accuracy"]:.2f}',
        ]
    else:
        lines.append(
            f'train     skipped: buffer below min_buffer ({t.min_buffer:,})'
        )
    mem = row.get('memory', {})
    parts = []
    if 'process_mb' in mem:
        parts.append(f'trainer {_mb(mem["process_mb"])}')
    if 'workers_mb' in mem:
        parts.append(f'workers {_mb(mem["workers_mb"])}')
    if 'cuda_mb' in mem:
        parts.append(
            f'gpu {_mb(mem["cuda_mb"])} (peak {mem["cuda_peak_mb"]:,}, '
            f'reserved {mem["cuda_reserved_mb"]:,})'
        )
    if parts:
        lines.append('memory    ' + ', '.join(parts))
    return '\n'.join(lines)


def iteration_line(row: dict[str, Any], iterations: int) -> str:
    """The one-line ``ITER`` entry for the console (when bars are off)."""
    sp = row['self_play']
    # with a time budget (no total), the hours trained show progress
    hours = ''
    if iterations <= 0 and 'elapsed_hours' in row:
        hours = f' [{row["elapsed_hours"]:.2f} h]'
    text = (
        f'{of_total(row["iteration"] + 1, iterations)}{hours}  '
        f'B/W/D {sp["black_wins"]}/{sp["white_wins"]}/{sp["draws"]}  '
        f'{row["sims_per_second"] / 1000:.1f}k sims/s'
    )
    if row['train_steps']:
        text += (
            f'  policy {row["policy_loss"]:.3f}  value {row["value_loss"]:.3f}'
        )
    return text + f'  {row["seconds"]:.1f} s'


def probe_report(
    iteration: int,
    iterations: int,
    expected: dict[str, list[str]],
    results: dict[str, Any],
) -> str:
    """The ``PROBES`` entry: one table row per probe."""
    lines = [
        f'after {of_total(iteration + 1, iterations)}',
        f'{"probe":<20} {"best":<8} {"expected":<14} {"net":>6} {"search":>7}',
    ]
    for name, r in results.items():
        if 'terminal' in r:
            lines.append(f'{name:<20} game already over')
            continue
        want = expected.get(name)
        verdict = (
            f'{"/".join(want)} {"ok" if r["correct"] else "MISS"}'
            if want
            else '-'
        )
        lines.append(
            f'{name:<20} {r["best"]:<8} {verdict:<14} '
            f'{_z(r["value"]):>+6.2f} {_z(r["search_value"]):>+7.2f}'
        )
    return '\n'.join(lines)


def arena_report(
    row: dict[str, Any], iterations: int, previous_elo: float | None
) -> str:
    """The full ``ARENA`` entry: a results table and the Elo rating."""
    elo = row['elo']
    change = ''
    if elo is not None and previous_elo is not None:
        change = f' ({elo - previous_elo:+.0f})'
    lines = [
        f'after {of_total(row["iteration"] + 1, iterations)} in '
        f'{row["seconds"]:.0f} s: '
        f'Elo {elo if elo is not None else "n/a"}{change}',
        f'{"opponent":<14} {"won":>4} {"drawn":>6} {"lost":>5} '
        f'{"score":>6} {"Elo diff":>9}',
    ]
    for r in row['results']:
        lines.append(
            f'{r["opponent"]:<14} {r["wins"]:>4} {r["draws"]:>6} '
            f'{r["losses"]:>5} {r["score"]:>6.3f} {_z(r["elo_diff"]):>+9.0f}'
        )
    return '\n'.join(lines)


def arena_line(row: dict[str, Any], iterations: int) -> str:
    """The one-line ``ARENA`` entry for the console."""
    scores = ', '.join(
        f'{r["opponent"]} {r["wins"]}-{r["draws"]}-{r["losses"]}'
        for r in row['results']
    )
    return (
        f'after {of_total(row["iteration"] + 1, iterations)}: {scores}  '
        f'Elo {row["elo"] if row["elo"] is not None else "n/a"}'
    )


def postfix(row: dict[str, Any], elo: float | None) -> dict[str, str]:
    """Short stats for the progress bar."""
    sp = row['self_play']
    stats = {
        'draws': f'{sp["draws"] / max(sp["games"], 1):.0%}',
        'sims/s': f'{row["sims_per_second"] / 1000:.1f}k',
    }
    if row['train_steps']:
        stats['policy'] = f'{row["policy_loss"]:.3f}'
        stats['value'] = f'{row["value_loss"]:.3f}'
    if elo is not None:
        stats['elo'] = f'{elo:.0f}'
    return stats
