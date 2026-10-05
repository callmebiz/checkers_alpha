"""Tournament: rate many players on one Elo scale.

The arena during training compares each checkpoint with fixed opponents and
its predecessor. A tournament instead puts many players (checkpoints from
different runs and times, the same network with and without search,
alpha-beta at several depths) into one pool, plays the pairings it is told
to, and fits a single rating to everyone.

Ratings
-------
Ratings use the Bradley-Terry model, the statistics behind Elo: player i
scores against player j with probability ``1 / (1 + 10^((Rj - Ri) / 400))``.
A draw counts as half a win. The ratings that best explain all the games
are found by a standard iterative (minorise-maximise) fit. To keep ratings
finite for a player who never loses, each player also gets one virtual draw
against a fixed reference. Ratings are shifted so ``anchor`` is 0.

Uncertainty comes from bootstrapping: the games are resampled with
replacement many times, ratings refitted each time, and the 5th and 95th
percentiles reported (a 90% interval).

Running
-------
A plan (JSON) lists the entrants and *blocks* of pairings in priority order.
Every pairing plays ``openings`` random openings with each colour. Games run
on the self-play workers (``training.actors``), with every network served
from this process. Results are appended to ``games.jsonl`` chunk by chunk,
so an interrupted tournament resumes where it stopped, and blocks that no
longer fit in ``time_limit_hours`` are skipped (most important first)::

    python -m training.tournament plan.json out_dir
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core.inference import NetEvaluator
from core.model import load_model
from core.players import make_player
from training import fileio
from training.actors import ActorPool
from training.arena import random_openings
from training.config import RunConfig
from training.logs import BlockFormatter, event
from training.progress import TqdmHandler


@dataclass
class Entrant:
    """A tournament player: a network checkpoint, or a plain player."""

    name: str
    checkpoint: str | None = None  # path, for network players
    sims: int = 200  # search budget for network players (1 = raw network)
    spec: str | None = None  # plain players, e.g. 'ab:6'


@dataclass
class Block:
    """Pairings to play together, in a plan's priority order."""

    name: str
    pairs: list[tuple[str, str]]


@dataclass
class Plan:
    """Everything a tournament needs (see the module docstring)."""

    config: str  # a run's config.json: rules and search settings
    entrants: list[Entrant]
    blocks: list[Block]
    openings: int = 15  # per pairing; each played with both colours
    opening_plies: int = 4
    seed: int = 7
    max_plies: int = 300
    anchor: str = 'ab4'
    workers: int = 7
    chunk_games: int = 600  # games played and saved at a time
    time_limit_hours: float = 0.0  # 0: no limit
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Plan:
        """Read a plan from JSON."""
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        data['entrants'] = [Entrant(**e) for e in data['entrants']]
        data['blocks'] = [
            Block(b['name'], [tuple(p) for p in b['pairs']])
            for b in data['blocks']
        ]
        return cls(**data)


# -- ratings -------------------------------------------------------------------


def score_table(
    games: list[dict[str, Any]],
) -> dict[tuple[str, str], tuple[float, int]]:
    """``(a, b) -> (a's score, games)`` over all games between a and b."""
    table: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    for g in games:
        black, white, winner = g['black'], g['white'], g['winner']
        score = 0.5 if winner is None else float(winner == 'black')
        for a, b, s in ((black, white, score), (white, black, 1 - score)):
            table[a, b][0] += s
            table[a, b][1] += 1
    return {k: (v[0], int(v[1])) for k, v in table.items()}


def fit_ratings(
    games: list[dict[str, Any]], anchor: str | None = None
) -> dict[str, float]:
    """Bradley-Terry Elo ratings for every player in ``games``.

    One virtual draw against a fixed reference per player keeps ratings
    finite. Shifted so ``anchor`` (if present) is 0.
    """
    table = score_table(games)
    players = sorted({a for a, _ in table})
    if not players:
        return {}
    index = {p: i for i, p in enumerate(players)}
    n = len(players)
    wins = np.full(n, 0.5)  # the virtual draw's half point
    pair_games = np.zeros((n, n))
    for (a, b), (s, g) in table.items():
        wins[index[a]] += s
        pair_games[index[a], index[b]] += g
    gamma = np.ones(n)
    for _ in range(2000):
        # minorise-maximise update (Hunter 2004), plus the virtual game
        # against a reference of strength 1
        denom = (pair_games / (gamma[:, None] + gamma[None, :])).sum(1)
        denom += 1.0 / (gamma + 1.0)
        new = wins / denom
        new /= np.exp(np.mean(np.log(new)))  # fix the scale's midpoint
        if np.max(np.abs(np.log(new) - np.log(gamma))) < 1e-9:
            gamma = new
            break
        gamma = new
    elo = 400 * np.log10(gamma)
    if anchor in index:
        elo -= elo[index[anchor]]
    return {p: float(elo[index[p]]) for p in players}


def bootstrap_intervals(
    games: list[dict[str, Any]],
    anchor: str | None,
    rounds: int = 200,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """90% intervals of each rating, by refitting on resampled games."""
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(rounds):
        pick = rng.integers(0, len(games), len(games))
        for player, r in fit_ratings([games[i] for i in pick], anchor).items():
            samples[player].append(r)
    return {
        p: (float(np.percentile(v, 5)), float(np.percentile(v, 95)))
        for p, v in samples.items()
    }


# -- running -------------------------------------------------------------------


def _logger(out: Path) -> logging.Logger:
    """Log to ``tournament.log`` (full) and the console (short)."""
    log = logging.getLogger(f'tournament:{out.resolve()}')
    if not log.handlers:
        log.setLevel(logging.INFO)
        log.propagate = False
        file = logging.FileHandler(out / 'tournament.log', encoding='utf-8')
        file.setFormatter(BlockFormatter('%Y-%m-%d %H:%M:%S'))
        console = TqdmHandler()
        console.setFormatter(BlockFormatter('%H:%M:%S'))
        log.addHandler(file)
        log.addHandler(console)
    return log


def run(plan: Plan, out: Path) -> dict[str, Any]:
    """Play the plan's unfinished chunks, then fit and save ratings."""
    out.mkdir(parents=True, exist_ok=True)
    log = _logger(out)
    games_path = out / 'games.jsonl'
    done_chunks = {
        (g['block'], g['chunk']) for g in fileio.read_jsonl(games_path)
    }
    start = time.monotonic()

    config = RunConfig.load(plan.config)
    config.self_play.workers = plan.workers
    # slot capacity per worker; bigger arena batches are split into chunks
    config.self_play.concurrent_games = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.device = device

    rng = np.random.default_rng(plan.seed)
    openings = random_openings(
        plan.openings, plan.opening_plies, rng, config.rules
    )
    # blocks with games still to play, and the networks they need (a long
    # gauntlet accumulates entrants whose games are long done)
    per_block = {b.name: 2 * len(b.pairs) * len(openings) for b in plan.blocks}
    pending = [
        b
        for b in plan.blocks
        if any(
            (b.name, c) not in done_chunks
            for c in range(-(-per_block[b.name] // plan.chunk_games))
        )
    ]
    needed = {name for b in pending for pair in b.pairs for name in pair}

    # one model per checkpoint (the raw and searched players share it)
    models: dict[str, int] = {}
    evaluators: list[NetEvaluator] = []
    specs: dict[str, str] = {}
    names: dict[str, str] = {}
    # plain players record their own names (e.g. 'ab4'): map them back
    renamed: dict[str, str] = {}
    for e in plan.entrants:
        if e.checkpoint is None:
            specs[e.name] = e.spec or e.name
            renamed[make_player(specs[e.name]).name] = e.name
            continue
        if e.name not in needed:
            continue
        if e.checkpoint not in models:
            model, _ = load_model(e.checkpoint, device)
            models[e.checkpoint] = len(evaluators)
            evaluators.append(
                NetEvaluator(model, backend='graphs', max_batch=256)
            )
        spec = f'model:{models[e.checkpoint]}:{e.sims}'
        specs[e.name] = spec
        names[spec] = e.name
    if device == 'cuda' and evaluators:
        used = torch.cuda.memory_allocated() / 2**20
        event(log, 'RUN', f'loaded {len(evaluators)} networks ({used:.0f} MB)')
    settings = {
        'names': names,
        'sims': 200,
        'max_plies': plan.max_plies,
        'seed': plan.seed,
    }
    pool = None
    if pending and plan.workers > 0:
        pool = ActorPool(config, plan.workers)
    try:
        for block in pending:
            tasks = []
            for a, b in block.pairs:
                for opening in openings:
                    fen = opening.to_fen()
                    tasks.append((len(tasks), fen, specs[a], specs[b]))
                    tasks.append((len(tasks), fen, specs[b], specs[a]))
            chunks = [
                tasks[i : i + plan.chunk_games]
                for i in range(0, len(tasks), plan.chunk_games)
            ]
            for c, chunk in enumerate(chunks):
                if (block.name, c) in done_chunks:
                    continue
                hours = (time.monotonic() - start) / 3600
                if plan.time_limit_hours and hours >= plan.time_limit_hours:
                    event(
                        log,
                        'WARN',
                        f'time limit reached: skipping block {block.name} '
                        f'from chunk {c + 1}/{len(chunks)}',
                    )
                    break
                t0 = time.monotonic()
                if pool is None:
                    raise RuntimeError('tournaments need worker processes')
                games = pool.play_arena(evaluators, settings, chunk)
                for g in games:
                    g['block'], g['chunk'] = block.name, c
                    for side in ('black', 'white'):
                        g[side] = renamed.get(g[side], g[side])
                    fileio.append_jsonl(games_path, g)
                event(
                    log,
                    'ARENA',
                    f'{block.name}: chunk {c + 1}/{len(chunks)}, '
                    f'{len(games)} games in {time.monotonic() - t0:.0f} s',
                )
    finally:
        if pool is not None:
            pool.close()

    games = fileio.read_jsonl(games_path)
    ratings = fit_ratings(games, plan.anchor)
    intervals = bootstrap_intervals(games, plan.anchor)
    table = score_table(games)
    result = {
        'anchor': plan.anchor,
        'games': len(games),
        'ratings': {
            p: {
                'elo': round(r, 1),
                'low': round(intervals.get(p, (r, r))[0], 1),
                'high': round(intervals.get(p, (r, r))[1], 1),
                'games': sum(g for (a, _), (_, g) in table.items() if a == p),
            }
            for p, r in sorted(ratings.items(), key=lambda kv: -kv[1])
        },
        'pairs': [
            {'a': a, 'b': b, 'score': s, 'games': g}
            for (a, b), (s, g) in sorted(table.items())
            if a < b
        ],
    }
    fileio.write_text(out / 'ratings.json', json.dumps(result, indent=2))
    lines = [f'{len(games)} games; Elo with {plan.anchor} = 0 (90% interval)']
    for p, r in result['ratings'].items():
        lines.append(
            f'{p:<24} {r["elo"]:>7.0f}  [{r["low"]:>6.0f}, {r["high"]:>6.0f}]'
        )
    event(log, 'RUN', '\n'.join(lines))
    for handler in list(log.handlers):
        handler.close()
        log.removeHandler(handler)
    return result


def main() -> None:
    """Command line: run a plan, writing results into a directory."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    parser.add_argument('plan', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(Plan.load(args.plan), args.out)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
