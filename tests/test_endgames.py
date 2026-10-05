import functools
import json
import random

import pytest

from core.endgames import (
    DRAW,
    LOSS,
    WIN,
    Solver,
    classes,
    key_of,
    material_of,
    mirror,
    placements,
    position_of,
)
from core.game import Position
from training.suites import (
    SUITES,
    _rate,
    _wilson,
    classify_tactic,
    verify_tactic,
)


@pytest.fixture(scope='module')
def solver():
    """Every class with up to three pieces, solved (~4 s)."""
    s = Solver()
    for m in classes(3):
        s.table(m)
    return s


@functools.cache
def forced(key: int, turns: int) -> int:
    """Independent brute force: 1 if the side to move can force a win
    within ``turns`` turns, -1 if it loses within them whatever it does,
    else 0. Plain minimax, no shared code with the solver beyond the
    rules."""
    pos = position_of(key)
    moves = pos.complete_moves()
    if not moves:
        return -1
    if turns == 0:
        return 0
    results = [-forced(key_of(nxt), turns - 1) for _, nxt in moves]
    if 1 in results:
        return 1
    if all(r == -1 for r in results):
        return -1
    return 0


def test_placements_respect_crowning_rows():
    """No man stands on the row where it would have been crowned, and
    every placement has the requested material."""
    for black, white, kings in placements((1, 0, 1, 0)):
        pos = Position(black, white, kings, 0)
        assert material_of(pos) == (1, 0, 1, 0)
        assert not (black & ~kings) >> 28  # black men never on 29-32
        assert not (white & ~kings) & 0xF  # white men never on 1-4


def test_results_match_brute_force(solver):
    """Wins and losses happen in exactly their stated number of turns
    (not one sooner), and draws can't be forced either way in 20 turns."""
    rng = random.Random(0)
    for material in [(0, 1, 0, 1), (0, 2, 0, 1), (1, 0, 1, 1), (2, 0, 1, 0)]:
        table = solver.tables[material]
        keys = rng.sample(list(table.index), 60)
        for key in keys:
            value, dtw = table.lookup(key)
            if value == DRAW:
                assert forced(key, 20) == 0
            else:
                assert forced(key, dtw) == value
                assert forced(key, dtw - 1) == 0 if dtw > 0 else True


def test_mirrored_positions_have_the_same_result(solver):
    """Swapping colours and turning the board can't change a result."""
    for table in solver.tables.values():
        for key in list(table.index)[::7]:
            pos = position_of(key)
            assert solver.lookup(mirror(pos)) == table.lookup(key)


def test_known_endgames(solver):
    """Two kings against one win almost everywhere; a side with no piece
    to move has lost."""
    table = solver.tables[(0, 2, 0, 1)]
    black_to_move = [table.lookup(k)[0] for k in table.index if not k >> 96]
    assert black_to_move.count(WIN) / len(black_to_move) > 0.99
    # a side with no pieces left has no move: lost on the spot
    assert solver.lookup(Position(1, 0, 1, 1)) == (LOSS, 0)


def test_built_puzzles_hold_up():
    """Puzzles in the built suite: the exact search at depth 6 finds the
    same move and kind, the fast check confirms it at depth 8, and a
    deliberately wrong answer is rejected."""
    path = SUITES / 'tactics.json'
    if not path.exists():
        pytest.skip('suites not built (python -m training.suites build)')
    suite = json.loads(path.read_text())
    for puzzle in suite['shots'][:3] + suite['only_moves'][:3]:
        found = classify_tactic(puzzle['fen'], 6)
        assert found is not None
        assert (found['move'], found['kind']) == (
            puzzle['move'],
            puzzle['kind'],
        )
        assert verify_tactic(puzzle, 8) is not None
        pos = Position.from_fen(puzzle['fen'])
        wrong = next(
            pos.move_text(a)
            for a, _ in pos.complete_moves()
            if pos.move_text(a) != puzzle['move']
        )
        assert verify_tactic({**puzzle, 'move': wrong}, 6) is None


def test_rates_have_sensible_intervals():
    """The 90% interval brackets the rate and narrows with more trials."""
    low, high = _wilson(30, 100)
    assert low < 0.3 < high
    narrow = _wilson(300, 1000)
    assert narrow[1] - narrow[0] < high - low
    assert _rate(0, 0)['rate'] == 0
