"""Exact solutions of small endgames, by working backwards from the end.

What it computes
----------------
For every position of a *material class* (how many men and kings each side
has), whether the side to move wins, draws or loses with perfect play, and
for wins and losses in how many turns (``dtw``: turns until the game ends,
with the winner hurrying and the loser delaying).

How (retrograde analysis)
-------------------------
1. List every position of the class: each way to place the pieces (men
   never on their own crowning row), with each side to move.
2. For each position, its successors after one full turn (multi-jumps
   included). A successor in another class (a capture removed a piece, or
   a man was crowned) belongs to a class solved first, recursively; classes
   only ever lead to classes with fewer pieces or more kings, so this ends.
3. A side with no move loses. Then, turn by turn: a position is won in
   ``k`` turns if some move leads to a position lost in ``k - 1``; lost in
   ``k`` if every move leads to a position won in at most ``k - 1``.
4. What's never decided is a draw: neither side can force a result, so
   play would go on (in a real game, until a draw rule ends it).

The draw rules are not part of these values: the game is drawn after 80
plies with no capture or man move (``Rules.draw_plies``), so a long enough
"win" is a draw in practice. Users of these tables that care (the probe
suites in ``training.suites``) keep only results that finish in time.

Positions are keyed by ``black | white << 32 | kings << 64 | turn << 96``
(the bitboards of ``core.game.Position``). Results are from the side to
move's view: ``WIN`` (1), ``DRAW`` (0) or ``LOSS`` (-1).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from core.game import BLACK, PROMOTION_ROW, WHITE, Position, iter_squares

WIN, DRAW, LOSS = 1, 0, -1
_UNKNOWN = 2  # while solving

# a material class: (black men, black kings, white men, white kings)
type Material = tuple[int, int, int, int]


def key_of(pos: Position) -> int:
    """A position's key (see the module docstring)."""
    return pos.black | pos.white << 32 | pos.kings << 64 | pos.turn << 96


def position_of(key: int) -> Position:
    """The position for a key (no history: draw counters start at zero)."""
    mask = (1 << 32) - 1
    return Position(
        key & mask, key >> 32 & mask, key >> 64 & mask, key >> 96 & 1
    )


def material_of(pos: Position) -> Material:
    """How many men and kings each side has."""
    k = pos.kings
    return (
        (pos.black & ~k).bit_count(),
        (pos.black & k).bit_count(),
        (pos.white & ~k).bit_count(),
        (pos.white & k).bit_count(),
    )


def placements(material: Material) -> Iterator[tuple[int, int, int]]:
    """Every ``(black, white, kings)`` placement of a material class."""
    bm, bk, wm, wk = material
    squares = range(32)
    # men can't stand on the row where they'd have been crowned
    black_men = [s for s in squares if not PROMOTION_ROW[BLACK] >> s & 1]
    white_men = [s for s in squares if not PROMOTION_ROW[WHITE] >> s & 1]

    def bits(chosen: tuple[int, ...]) -> int:
        """Bitboard of chosen squares."""
        return sum(1 << s for s in chosen)

    for a in itertools.combinations(black_men, bm):
        used = set(a)
        for b in itertools.combinations(
            [s for s in squares if s not in used], bk
        ):
            used_b = used | set(b)
            for c in itertools.combinations(
                [s for s in white_men if s not in used_b], wm
            ):
                used_c = used_b | set(c)
                for d in itertools.combinations(
                    [s for s in squares if s not in used_c], wk
                ):
                    black = bits(a) | bits(b)
                    white = bits(c) | bits(d)
                    yield black, white, bits(b) | bits(d)


@dataclass
class Table:
    """The solved positions of one material class."""

    material: Material
    index: dict[int, int]  # key -> row
    value: np.ndarray  # int8: WIN / DRAW / LOSS for the side to move
    dtw: np.ndarray  # int16: turns to the end for wins/losses, -1 draws

    def lookup(self, key: int) -> tuple[int, int]:
        """``(value, dtw)`` of a position in this class."""
        row = self.index[key]
        return int(self.value[row]), int(self.dtw[row])


@dataclass
class Solver:
    """Solves material classes on demand, remembering every table."""

    tables: dict[Material, Table] = field(default_factory=dict)
    log: bool = False

    def lookup(self, pos: Position) -> tuple[int, int]:
        """``(value, dtw)`` of any position, solving its class if needed."""
        material = material_of(pos)
        mover = material[:2] if pos.turn == BLACK else material[2:]
        if sum(mover) == 0:  # no pieces: no moves
            return LOSS, 0
        return self.table(material).lookup(key_of(pos))

    def table(self, material: Material) -> Table:
        """The solved table of a class (solving it, and what it leads to,
        the first time)."""
        if material not in self.tables:
            self.tables[material] = self._solve(material)
        return self.tables[material]

    def _solve(self, material: Material) -> Table:
        """Retrograde analysis of one class (see the module docstring)."""
        keys = [
            black | white << 32 | kings << 64 | turn << 96
            for black, white, kings in placements(material)
            for turn in (BLACK, WHITE)
        ]
        index = {k: i for i, k in enumerate(keys)}
        n = len(keys)
        # successors as flat edge lists: a row inside the class, or an
        # already known (value, dtw) from another class
        starts = np.zeros(n + 1, np.int64)
        inside: list[int] = []
        known_value: list[int] = []
        known_dtw: list[int] = []
        for i, key in enumerate(keys):
            pos = position_of(key)
            for _, nxt in pos.complete_moves():
                nkey = key_of(nxt)
                row = index.get(nkey)
                if row is not None:
                    inside.append(row)
                    known_value.append(_UNKNOWN)
                    known_dtw.append(0)
                else:
                    v, d = self.lookup(nxt)
                    inside.append(-1)
                    known_value.append(v)
                    known_dtw.append(d)
            starts[i + 1] = len(inside)
        edge_row = np.array(inside, np.int64)
        edge_value = np.array(known_value, np.int8)
        edge_dtw = np.array(known_dtw, np.int32)
        internal = edge_row >= 0
        safe_row = np.where(internal, edge_row, 0)

        value = np.full(n, _UNKNOWN, np.int8)
        dtw = np.full(n, -1, np.int32)
        counts = np.diff(starts)
        value[counts == 0] = LOSS  # no move: lost now
        dtw[counts == 0] = 0
        has = counts > 0
        seg = starts[:-1][has]  # reduceat needs non-empty segments
        horizon = int(edge_dtw[~internal].max(initial=0))

        k = 0
        while True:
            k += 1
            v = np.where(internal, value[safe_row], edge_value)
            d = np.where(internal, dtw[safe_row], edge_dtw)
            done = d <= k - 1  # results known by turn k - 1
            to_loss = np.logical_or.reduceat((v == LOSS) & done, seg)
            all_won = np.logical_and.reduceat((v == WIN) & done, seg)
            open_rows = np.flatnonzero(has)[value[has] == _UNKNOWN]
            win_now = open_rows[to_loss[value[has] == _UNKNOWN]]
            loss_now = open_rows[
                all_won[value[has] == _UNKNOWN]
                & ~to_loss[value[has] == _UNKNOWN]
            ]
            value[win_now], dtw[win_now] = WIN, k
            value[loss_now], dtw[loss_now] = LOSS, k
            if not len(win_now) and not len(loss_now) and k > horizon:
                break
        value[value == _UNKNOWN] = DRAW
        if self.log:
            won = int((value == WIN).sum())
            drawn = int((value == DRAW).sum())
            print(
                f'solved {material}: {n:,} positions, {won:,} won, '
                f'{drawn:,} drawn, longest {int(dtw.max())} turns',
                flush=True,
            )
        return Table(material, index, value, dtw.astype(np.int16))


def mirror(pos: Position) -> Position:
    """The same position with colours swapped and the board turned 180
    degrees (square s -> 31 - s): it must have the same result."""

    def turn(bits: int) -> int:
        """Rotate a bitboard 180 degrees."""
        return sum(1 << (31 - s) for s in iter_squares(bits))

    return Position(
        turn(pos.white), turn(pos.black), turn(pos.kings), 1 - pos.turn
    )


def classes(max_pieces: int) -> list[Material]:
    """Every class with 1 to ``max_pieces`` pieces in all, each side having
    at least one."""
    out = []
    for total in range(2, max_pieces + 1):
        for bm, bk, wm, wk in itertools.product(range(total + 1), repeat=4):
            if bm + bk + wm + wk == total and bm + bk and wm + wk:
                out.append((bm, bk, wm, wk))
    return out
