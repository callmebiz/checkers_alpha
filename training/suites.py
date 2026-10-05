"""Test suites: fixed puzzle sets that measure specific skills.

The arena and gauntlet say *how strong* a network is; these say *at what*.
Each suite is built once (``build``), saved under ``configs/suites/``, and
every checkpoint is scored on the same positions (``score``), so scores are
comparable over a run's whole life.

Endgames (``endgames.json``), from exact solutions (``core.endgames``)
--------------------------------------------------------------------
Every position with up to three pieces, and every two-against-two, is
solved exactly. From those:

- **judgement**: quiet positions (no capture for the side to move), equal
  numbers won, drawn and lost, across four groups (two pieces, three,
  two kings each, two-against-two with men). Score: how often the
  network's most likely outcome (win/draw/loss) is the true one.
- **the move**: pairs of positions identical but for the side to move,
  whose true outcome differs. In checkers endgames whose turn it is often
  decides the game ("having the move"). Score: whether the network ranks
  the pair the right way round (Black's expected score higher in the
  position that's truly better for Black).
- **deep wins**: wins that take 25-60 turns of precise play (the hardest
  in these classes). Score: whether the network sees them as won.

Results count only if they hold under this program's draw rule (a draw
after 80 plies without a capture or man move): wins and losses are kept
only if they finish within ``MAX_TURNS`` turns from the position, which no
draw rule can interrupt. Draws are draws under any rule.

Tactics (``tactics.json``), from real games
-------------------------------------------
Positions from tournament and gauntlet games, searched exhaustively by
alpha-beta with a material-only evaluation, at every depth from 6 to 12
turns (capture sequences followed to the end):

- **shots**: one move, a sacrifice (it leaves the opponent a capture),
  wins at least a man and is at least a man better than every other move.
- **only moves**: one move holds the material balance; every other move
  loses at least a man.

A position is kept only if every one of those seven depths agrees on the
move and the margins. A single depth can be fooled by its horizon (an
exchange cut off at the wrong moment), and the check at each depth is
cheap (see ``verify_tactic``), so none is skipped.
Score: whether the network's preferred move (its prior, and its search)
is that move.

    python -m training.suites build            # ~30 min, once
    python -m training.suites score runs/x/checkpoints/iter_0100.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from core.endgames import (
    DRAW,
    LOSS,
    WIN,
    Solver,
    classes,
    key_of,
    position_of,
)
from core.game import A_MID, BLACK, Position

ROOT = Path(__file__).resolve().parent.parent
SUITES = ROOT / 'configs' / 'suites'
MAX_TURNS = 60  # wins/losses kept only if decided this fast (see above)
DEEP = (25, MAX_TURNS)  # turns for a "deep win"
PER_CELL = 25  # judgement positions per (group, outcome)
PAIRS = 150  # "the move" pairs
DEEP_WINS = 60
MAN, KING = 100, 160  # material values for the tactics search
MARGIN = 90  # "at least a man" (a little under 100, for rounding)
DEPTHS = tuple(range(6, 13))  # tactics: every depth must agree
MAX_EXTENSION = 10  # extra turns followed while captures continue
WIN_SCORE = 100_000


# -- endgames ------------------------------------------------------------------


def _group(material: tuple[int, int, int, int]) -> str:
    """Which judgement group a material class belongs to."""
    bm, bk, wm, wk = material
    pieces = bm + bk + wm + wk
    if pieces <= 2:
        return 'two pieces'
    if pieces == 3:
        return 'three pieces'
    if bm + wm == 0:
        return 'two kings each'
    return 'two against two, men'


def _quiet(pos: Position) -> bool:
    """Whether the side to move has no capture (judgement, not tactics)."""
    return not any(A_MID[a] >= 0 for a in pos.legal_actions())


def _decided(value: int, dtw: int) -> bool:
    """Whether a result holds under the draw rule (see the docstring)."""
    return value == DRAW or 0 < dtw <= MAX_TURNS


def build_endgames(rng: random.Random, log: bool = True) -> dict[str, Any]:
    """Solve the classes and draw the three endgame sets."""
    solver = Solver(log=log)
    materials = [
        m
        for m in classes(4)
        if sum(m) <= 3 or (m[0] + m[1] == 2 and m[2] + m[3] == 2)
    ]
    for m in materials:
        solver.table(m)
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    deep: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for m in materials:
        table = solver.tables[m]
        group = _group(m)
        for key, row in table.index.items():
            value, dtw = int(table.value[row]), int(table.dtw[row])
            pos = position_of(key)
            if dtw == 0 or not _decided(value, dtw) or not _quiet(pos):
                continue
            item = {
                'fen': pos.to_fen(),
                'value': value,
                'dtw': dtw,
                'group': group,
            }
            cells[group, value].append(item)
            if value == WIN and DEEP[0] <= dtw <= DEEP[1]:
                deep.append(item)
            if pos.turn != BLACK:
                continue
            # the same pieces with White to move
            other = position_of(key | 1 << 96)
            v2, d2 = table.lookup(key | 1 << 96)
            if d2 == 0 or not _decided(v2, d2) or not _quiet(other):
                continue
            black_b, black_w = value, -v2  # both from Black's view
            if black_b != black_w:
                pairs.append(
                    {
                        'black_to_move': pos.to_fen(),
                        'white_to_move': other.to_fen(),
                        # Black's true result in each
                        'black_result': [black_b, black_w],
                        'group': group,
                    }
                )
    judgement = []
    for _, items in sorted(cells.items()):
        judgement += rng.sample(items, min(PER_CELL, len(items)))
    # deepest first, then a spread over classes
    deep.sort(key=lambda d: -d['dtw'])
    by_group: dict[str, list] = defaultdict(list)
    for d in deep:
        by_group[d['group']].append(d)
    deep_wins: list[dict[str, Any]] = []
    while len(deep_wins) < DEEP_WINS and any(by_group.values()):
        for items in by_group.values():
            if items and len(deep_wins) < DEEP_WINS:
                deep_wins.append(items.pop(0))
    pair_groups: dict[str, list] = defaultdict(list)
    for p in pairs:
        pair_groups[p['group']].append(p)
    the_move = []
    share = PAIRS // max(len(pair_groups), 1)
    for items in pair_groups.values():
        the_move += rng.sample(items, min(share, len(items)))
    stats = {
        'positions': sum(len(t.index) for t in solver.tables.values()),
        'classes': len(solver.tables),
    }
    return {
        'built': time.strftime('%Y-%m-%d %H:%M'),
        'max_turns': MAX_TURNS,
        'solved': stats,
        'judgement': judgement,
        'the_move': the_move,
        'deep_wins': deep_wins,
    }


# -- tactics search ------------------------------------------------------------


def material(pos: Position) -> int:
    """Material balance for the side to move (men 100, kings 160)."""
    k = pos.kings
    mine, theirs = (
        (pos.black, pos.white) if pos.turn == BLACK else (pos.white, pos.black)
    )
    return (
        MAN * (mine & ~k).bit_count()
        + KING * (mine & k).bit_count()
        - MAN * (theirs & ~k).bit_count()
        - KING * (theirs & k).bit_count()
    )


def _negamax(
    pos: Position, depth: int, alpha: int, beta: int, extension: int
) -> int:
    """Alpha-beta score for the side to move: material only, following
    captures past the depth limit. Faster wins and slower losses score
    higher."""
    result = pos.result()
    if result is not None:
        return result * (WIN_SCORE + depth)
    if depth <= 0:
        capture = A_MID[pos.legal_actions()[0]] >= 0
        if not capture or extension >= MAX_EXTENSION:
            return material(pos)
        extension += 1
    for _, nxt in pos.complete_moves():
        v = -_negamax(nxt, depth - 1, -beta, -alpha, extension)
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return alpha


def move_scores(fen: str, depth: int) -> list[tuple[list[int], int]]:
    """Every full move from ``fen`` with its exact score at ``depth``."""
    pos = Position.from_fen(fen)
    out = []
    for actions, nxt in pos.complete_moves():
        score = -_negamax(nxt, depth - 1, -WIN_SCORE * 10, WIN_SCORE * 10, 0)
        out.append((list(actions), score))
    return sorted(out, key=lambda m: -m[1])


def classify_tactic(fen: str, depth: int) -> dict[str, Any] | None:
    """The puzzle in ``fen`` at ``depth``, or None if there isn't one."""
    pos = Position.from_fen(fen)
    scores = move_scores(fen, depth)
    if len(scores) < 2:
        return None
    (best, top), (_, second) = scores[0], scores[1]
    if top - second < MARGIN:
        return None  # not a single clearly best move
    now = material(pos)
    after = pos
    for a in best:
        after = after.play(a)
    sacrifice = any(A_MID[a] >= 0 for a in after.legal_actions())
    if sacrifice and top - now >= MARGIN:
        kind = 'shot'
    elif top - now >= -MARGIN // 2:
        kind = 'only move'
    else:
        return None  # everything loses material: not a clean puzzle
    return {
        'fen': fen,
        'kind': kind,
        'move': pos.move_text(best),
        'action': best[0],
        'gain': top - now,
        'margin': top - second,
    }


def verify_tactic(puzzle: dict[str, Any], depth: int) -> int | None:
    """Re-check a puzzle at a greater ``depth``; its new gain, or None if
    it no longer holds.

    The puzzle's move is scored exactly. Every other move only needs a
    yes/no answer, "within ``MARGIN`` of it?", which a zero-width
    alpha-beta window answers far faster than an exact score.
    """
    pos = Position.from_fen(puzzle['fen'])
    now = material(pos)
    wide = WIN_SCORE * 10
    best = None
    others = []
    for actions, nxt in pos.complete_moves():
        if pos.move_text(actions) == puzzle['move']:
            best = -_negamax(nxt, depth - 1, -wide, wide, 0)
        else:
            others.append(nxt)
    if best is None:
        return None
    gain = best - now
    if puzzle['kind'] == 'shot' and gain < MARGIN:
        return None
    if puzzle['kind'] == 'only move' and gain < -MARGIN // 2:
        return None
    bound = best - MARGIN  # alternatives must score at most this
    for nxt in others:
        # window (bound, bound + 1): the result is > bound exactly when
        # the move scores above it
        v = -_negamax(nxt, depth - 1, -(bound + 1), -bound, 0)
        if v > bound:
            return None
    return gain


def _check(fen: str) -> dict[str, Any] | None:
    """A puzzle found at the first depth in ``DEPTHS`` and confirmed at
    every deeper one, or None. Runs in worker processes."""
    puzzle = classify_tactic(fen, DEPTHS[0])
    if puzzle is None:
        return None
    for depth in DEPTHS[1:]:
        gain = verify_tactic(puzzle, depth)
        if gain is None:
            return None
        puzzle = {**puzzle, 'gain': gain, 'depth': depth}
    return puzzle


def candidate_positions(
    sources: Iterable[Path], rng: random.Random, limit: int
) -> list[str]:
    """Distinct middle-game positions from game files: no capture for the
    side to move, at least three moves to choose from, 6+ pieces."""
    seen: set[int] = set()
    fens: list[str] = []
    for path in sources:
        for line in path.read_text(encoding='utf-8').splitlines():
            game = json.loads(line)
            try:
                pos = Position.from_fen(game['start'])
            except (KeyError, ValueError):
                continue
            for action in game['actions']:
                if (
                    pos.jump_sq < 0
                    and not pos.is_terminal
                    and (pos.black | pos.white).bit_count() >= 6
                    and _quiet(pos)
                    and len(pos.complete_moves()) >= 3
                ):
                    key = key_of(pos)
                    if key not in seen:
                        seen.add(key)
                        fens.append(pos.to_fen())
                if action not in pos.legal_actions():
                    break
                pos = pos.play(action)
    rng.shuffle(fens)
    return fens[:limit]


def build_tactics(
    sources: list[Path],
    rng: random.Random,
    limit: int = 400_000,
    per_kind: int = 100,
    workers: int = 16,
) -> dict[str, Any]:
    """Search game positions for verified puzzles (see the docstring)."""
    fens = candidate_positions(sources, rng, limit)
    print(f'tactics: {len(fens):,} candidate positions', flush=True)
    puzzles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    t0 = time.time()
    i = -1
    with get_context('spawn').Pool(workers) as pool:
        for i, p in enumerate(pool.imap_unordered(_check, fens, chunksize=8)):
            if p is not None and len(puzzles[p['kind']]) < per_kind:
                puzzles[p['kind']].append(p)
            if (i + 1) % 2000 == 0:
                print(
                    f'  {i + 1:,} searched in {time.time() - t0:.0f} s: '
                    + ', '.join(f'{k} {len(v)}' for k, v in puzzles.items()),
                    flush=True,
                )
            if all(len(puzzles[k]) >= per_kind for k in ('shot', 'only move')):
                pool.terminate()
                break
    return {
        'built': time.strftime('%Y-%m-%d %H:%M'),
        'depths': list(DEPTHS),
        'searched': i + 1,
        'shots': puzzles['shot'],
        'only_moves': puzzles['only move'],
    }


# -- scoring -------------------------------------------------------------------


def load() -> dict[str, Any]:
    """The built suites."""
    return {
        'endgames': json.loads((SUITES / 'endgames.json').read_text()),
        'tactics': json.loads((SUITES / 'tactics.json').read_text()),
    }


def _wilson(k: int, n: int) -> tuple[float, float]:
    """90% interval of a success rate (Wilson)."""
    if n == 0:
        return 0.0, 1.0
    z = 1.645
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = (
        z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(k: int, n: int) -> dict[str, Any]:
    """A success count with its rate and interval."""
    low, high = _wilson(k, n)
    return {
        'k': k,
        'n': n,
        'rate': round(k / max(n, 1), 4),
        'low': round(low, 4),
        'high': round(high, 4),
    }


def score_network(evaluator: Any, sims: int = 200) -> dict[str, Any]:
    """Score a network on every suite, without and with search.

    For endgames the outcome guessed is the most likely of win/draw/loss:
    the network's own, and the search's (``core.mcts.search_wdl``).
    """
    from core.mcts import SearchTree, run_search, search_wdl

    suites = load()
    out: dict[str, Any] = {'sims': sims}

    def judge(fens: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Win/draw/loss (side to move) by the network and the search."""
        positions = [Position.from_fen(f) for f in fens]
        _, _, wdl = evaluator.evaluate(positions)
        trees = [SearchTree(p) for p in positions]
        run_search(trees, evaluator, sims)
        searched = np.array(
            [search_wdl(t.root) or w for t, w in zip(trees, wdl, strict=True)]
        )
        return np.asarray(wdl, dtype=float), searched

    # outcome codes 1/0/-1 -> win/draw/loss columns 0/1/2
    column = {WIN: 0, DRAW: 1, LOSS: 2}
    eg = suites['endgames']
    items = eg['judgement']
    net, srch = judge([i['fen'] for i in items])
    truth = np.array([column[i['value']] for i in items])
    groups = sorted({i['group'] for i in items})
    out['judgement'] = {
        'net': _rate(int((net.argmax(1) == truth).sum()), len(items)),
        'search': _rate(int((srch.argmax(1) == truth).sum()), len(items)),
        'by_group': {
            g: _rate(
                int(
                    sum(
                        net[j].argmax() == truth[j]
                        for j, i in enumerate(items)
                        if i['group'] == g
                    )
                ),
                sum(i['group'] == g for i in items),
            )
            for g in groups
        },
    }

    pairs = eg['the_move']
    fens = [p['black_to_move'] for p in pairs] + [
        p['white_to_move'] for p in pairs
    ]
    net, srch = judge(fens)
    n = len(pairs)

    def black_score(wdl: np.ndarray, black_moves: bool) -> np.ndarray:
        """Black's expected score from win/draw/loss for the side to move."""
        mover = wdl[:, 0] + wdl[:, 1] / 2
        return mover if black_moves else 1 - mover

    for name, v in (('net', net), ('search', srch)):
        b = black_score(v[:n], True)
        w = black_score(v[n:], False)
        right = sum(
            (bw > ww) == (p['black_result'][0] > p['black_result'][1])
            and bw != ww
            for p, bw, ww in zip(pairs, b, w, strict=True)
        )
        out.setdefault('the_move', {})[name] = _rate(int(right), n)

    deep = eg['deep_wins']
    net, srch = judge([d['fen'] for d in deep])
    out['deep_wins'] = {
        'net': _rate(int((net.argmax(1) == 0).sum()), len(deep)),
        'search': _rate(int((srch.argmax(1) == 0).sum()), len(deep)),
    }

    for kind in ('shots', 'only_moves'):
        puzzles = suites['tactics'][kind]
        positions = [Position.from_fen(p['fen']) for p in puzzles]
        priors, _, _ = evaluator.evaluate(positions)
        net_moves = [
            pos.legal_actions()[int(np.argmax(pr))]
            for pos, pr in zip(positions, priors, strict=True)
        ]
        trees = [SearchTree(p) for p in positions]
        run_search(trees, evaluator, sims)
        out[kind] = {
            'net': _rate(
                sum(
                    m == p['action']
                    for m, p in zip(net_moves, puzzles, strict=True)
                ),
                len(puzzles),
            ),
            'search': _rate(
                sum(
                    t.best_action() == p['action']
                    for t, p in zip(trees, puzzles, strict=True)
                ),
                len(puzzles),
            ),
        }
    return out


def score_engine(spec: str) -> dict[str, Any]:
    """Tactics scores of a plain engine (e.g. ``ab:6``), for comparison."""
    from core.players import make_player

    player = make_player(spec, seed=0)
    out: dict[str, Any] = {}
    for kind, puzzles in load()['tactics'].items():
        if kind not in ('shots', 'only_moves'):
            continue
        right = sum(
            player.choose(Position.from_fen(p['fen'])) == p['action']
            for p in puzzles
        )
        out[kind] = _rate(right, len(puzzles))
    return out


def score_checkpoint(path: str | Path, sims: int = 200) -> dict[str, Any]:
    """Load a checkpoint on the GPU (if any) and score it."""
    import torch

    from core.inference import NetEvaluator
    from core.model import load_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(path, device)
    evaluator = NetEvaluator(model, backend='graphs', max_batch=512)
    return score_network(evaluator, sims)


def summary(scores: dict[str, Any]) -> str:
    """Scores as one line per suite."""
    lines = []
    for name in ('judgement', 'the_move', 'deep_wins', 'shots', 'only_moves'):
        s = scores.get(name)
        if not s:
            continue
        parts = [
            f'{who} {s[who]["k"]}/{s[who]["n"]} ({s[who]["rate"]:.0%})'
            for who in ('net', 'search')
            if who in s
        ]
        lines.append(f'{name:<11} ' + ', '.join(parts))
    return '\n'.join(lines)


# -- command line --------------------------------------------------------------


def _sources() -> list[Path]:
    """Game files to mine for tactics: tournaments and gauntlets."""
    runs = ROOT / 'runs'
    return sorted(
        p
        for p in [
            *runs.rglob('tournament/games.jsonl'),
            *runs.rglob('gauntlet/games.jsonl'),
        ]
        if p.stat().st_size
    )


def main() -> None:
    """Command line: build the suites, or score a checkpoint/engine."""
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--only', choices=['endgames', 'tactics'])
    build.add_argument('--seed', type=int, default=2026)
    build.add_argument('--workers', type=int, default=16)
    score = sub.add_parser('score')
    score.add_argument('target', help='a checkpoint, or an engine: ab:6')
    score.add_argument('--sims', type=int, default=200)
    score.add_argument('--out', type=Path, help='append the scores here')
    score.add_argument('--name', help='label for --out rows')
    score.add_argument('--hours', type=float, help='training hours, for --out')
    args = parser.parse_args()

    if args.command == 'build':
        SUITES.mkdir(parents=True, exist_ok=True)
        rng = random.Random(args.seed)
        if args.only in (None, 'endgames'):
            t = time.time()
            eg = build_endgames(rng)
            (SUITES / 'endgames.json').write_text(json.dumps(eg, indent=1))
            counts = Counter((i['group'], i['value']) for i in eg['judgement'])
            print(
                f'endgames in {time.time() - t:.0f} s: '
                f'{len(eg["judgement"])} judgement {dict(counts)}, '
                f'{len(eg["the_move"])} pairs, {len(eg["deep_wins"])} deep',
                flush=True,
            )
        if args.only in (None, 'tactics'):
            t = time.time()
            tac = build_tactics(_sources(), rng, workers=args.workers)
            (SUITES / 'tactics.json').write_text(json.dumps(tac, indent=1))
            print(
                f'tactics in {time.time() - t:.0f} s: '
                f'{len(tac["shots"])} shots, {len(tac["only_moves"])} only '
                f'moves from {tac["searched"]:,} positions',
                flush=True,
            )
        return

    if args.target.startswith('ab:'):
        scores = score_engine(args.target)
    else:
        scores = score_checkpoint(args.target, args.sims)
    print(summary(scores))
    if args.out:
        from training import fileio

        row = {
            'time': time.time(),
            'name': args.name or args.target,
            'target': args.target,
            'hours': args.hours,
            **scores,
        }
        fileio.append_jsonl(args.out, row)


if __name__ == '__main__':
    main()
