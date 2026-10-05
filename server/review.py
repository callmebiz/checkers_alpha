"""Game review: how good was each move, and how accurately did each side
play?

Judging a move
--------------
A side's *expected score* is its chance of winning plus half its chance of
a draw (0 to 1). A search value ``q`` (win minus loss chance, -1 to 1)
converts exactly: expected score = (1 + q) / 2.

For each turn, the judge (a network with a deep search) values the
position before it (what the mover could expect with the best move) and
the position after it (what the mover can expect now). The difference is
the move's *cost*: the share of a point it gave away. The same idea as
chess apps' lost win probability, but draws count, and checkers is full of
them: throwing a win away into a draw costs half a point.

Categories, checked in this order (thresholds as in the chess analyzer
next door, ``E:/Projects/chess_analyzer/backend/scoring.py``)::

    forced      the only legal move (captures are compulsory, so common)
    blunder     cost >= 20%
    miss        was winning (>= 70%) and gave away >= 15%
    mistake     cost >= 10%
    inaccuracy  cost >= 5%
    great       the best move, and the only good one: every alternative
                costs >= 15%, in a position not already decided
    best        the search's choice (or within half a percent of it)
    excellent   cost < 2%
    good        cost < 5%

Accuracy
--------
Each move's cost maps to an accuracy (100 for a free move, falling
quickly) with the curve chess sites use on lost win percentage; a side's
accuracy is the mean over its unforced moves.

Searches are noisy at low budgets and would invent mistakes, so reviews
search deeply (``REVIEW_SIMS``), every position at once so the GPU sees
large batches.
"""

from __future__ import annotations

import math
from typing import Any

from core.game import COLOR_NAMES, Position
from core.inference import NetEvaluator
from core.mcts import SearchTree, run_search
from server.engine import MAX_SIMS, replay

REVIEW_SIMS = 1600
QUALITIES = (
    'forced', 'great', 'best', 'excellent', 'good',
    'inaccuracy', 'mistake', 'miss', 'blunder',
)  # fmt: skip
GREAT_GAP = 0.15  # every alternative costs at least this for 'great'
MIN_VISITS = 20  # alternatives searched less are taken to be bad


def score(q: float) -> float:
    """Expected score (0-1) from a search value (-1 to 1)."""
    return (1.0 + q) / 2.0


def move_accuracy(cost: float) -> float:
    """Accuracy (0-100) of a move that gave away ``cost`` (0-1) of a point:
    the curve chess sites apply to lost win percentage."""
    return max(0.0, min(100.0, 103.1668 * math.exp(-4.354 * cost) - 3.1669))


def classify(
    cost: float,
    before: float,
    is_best: bool,
    forced: bool,
    only_good: bool,
) -> str:
    """A move's category (see the module docstring for the rules).

    ``before`` is the mover's expected score with best play, ``is_best``
    whether the move was the search's choice, ``only_good`` whether every
    alternative costs at least ``GREAT_GAP``.
    """
    if forced:
        return 'forced'
    if cost >= 0.20:
        return 'blunder'
    if before >= 0.70 and cost >= 0.15:
        return 'miss'
    if cost >= 0.10:
        return 'mistake'
    if cost >= 0.05:
        return 'inaccuracy'
    if is_best or cost <= 0.005:
        if is_best and only_good and 0.15 <= before <= 0.85:
            return 'great'
        return 'best'
    return 'excellent' if cost < 0.02 else 'good'


def _turn_starts(positions: list[Position]) -> list[int]:
    """Indices of the positions where a turn starts (not mid-jump)."""
    starts = [0]
    for i in range(1, len(positions) - 1):
        if positions[i].jump_sq < 0:
            starts.append(i)
    return starts


def _only_good(tree: SearchTree, best_index: int) -> bool:
    """Whether every other first hop is clearly worse than the best one."""
    root = tree.root
    if len(root.actions) < 2 or not root.N[best_index]:
        return False
    best_q = root.W[best_index] / root.N[best_index]
    for i in range(len(root.actions)):
        if i == best_index or root.N[i] < MIN_VISITS:
            continue  # barely searched: the search didn't rate it
        if score(best_q) - score(root.W[i] / root.N[i]) < GREAT_GAP:
            return False
    return True


def _best_turn(tree: SearchTree) -> list[int]:
    """The search's preferred full turn: best hops while the mover stays."""
    node, hops = tree.root, []
    mover = node.pos.turn
    while node is not None and node.expanded and node.n_sum:
        i = max(range(len(node.N)), key=lambda j: (node.N[j], node.P_raw[j]))
        hops.append(node.actions[i])
        child = node.children[i]
        if child is None or child.pos.turn != mover or child.pos.jump_sq < 0:
            break
        node = child
    return hops


def review(
    start: str | None,
    actions: list[int],
    evaluator: NetEvaluator,
    sims: int = REVIEW_SIMS,
) -> dict[str, Any]:
    """Judge every complete turn of a line (see the module docstring).

    Returns ``turns`` (one per turn: colour, text, category, cost, the
    mover's expected score before and after, accuracy, and the move the
    judge preferred) and ``sides`` (accuracy and category counts for
    Black and White).
    """
    positions = replay(start, actions)
    if not actions:
        return {'sims': sims, 'turns': [], 'sides': _sides([])}
    starts = _turn_starts(positions)
    # a line ending mid multi-jump: that last turn isn't finished
    ends = starts[1:] + (
        [len(positions) - 1] if positions[-1].jump_sq < 0 else []
    )
    starts = starts[: len(ends)]
    searched = sorted({*starts, *ends})
    live = [i for i in searched if not positions[i].is_terminal]
    trees = {i: SearchTree(positions[i]) for i in live}
    if trees:
        run_search(
            list(trees.values()),
            evaluator,
            min(sims, MAX_SIMS),
            leaves_per_tree=4,
        )

    def value(i: int) -> float:
        """Search value of position i for its side to move."""
        result = positions[i].result()
        return float(result) if result is not None else trees[i].root.q()

    turns = []
    for a, b in zip(starts, ends, strict=True):
        pos, after = positions[a], positions[b]
        mover = pos.turn
        before = score(value(a))
        q_after = value(b)
        now = score(q_after if after.turn == mover else -q_after)
        cost = max(0.0, before - now)
        played = actions[a:b]
        forced = len(pos.complete_moves()) == 1
        tree = trees.get(a)
        best_hops = _best_turn(tree) if tree else []
        is_best = bool(best_hops) and best_hops == played[: len(best_hops)]
        only = False
        if tree is not None and is_best:
            only = _only_good(tree, tree.root.actions.index(played[0]))
        quality = classify(cost, before, is_best, forced, only)
        turns.append(
            {
                'color': COLOR_NAMES[mover],
                'ply': a,
                'hops': b - a,
                'text': pos.move_text(played),
                'quality': quality,
                'cost': round(cost, 4),
                'before': round(before, 4),
                'after': round(now, 4),
                'accuracy': None if forced else round(move_accuracy(cost), 1),
                'best': pos.move_text(best_hops) if best_hops else None,
            }
        )

    return {'sims': sims, 'turns': turns, 'sides': _sides(turns)}


def _sides(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy (mean over unforced moves) and category counts per side."""
    sides: dict[str, Any] = {}
    for color in COLOR_NAMES:
        mine = [t for t in turns if t['color'] == color]
        scored = [t['accuracy'] for t in mine if t['accuracy'] is not None]
        counts = {q: sum(t['quality'] == q for t in mine) for q in QUALITIES}
        sides[color] = {
            'accuracy': round(sum(scored) / len(scored), 1) if scored else None,
            'counts': counts,
        }
    return sides
