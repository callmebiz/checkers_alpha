import pytest

from core.game import (
    A_FROM,
    A_TO,
    ACTION_SIZE,
    BLACK,
    WHITE,
    Position,
    Rules,
    perft,
)


def action_for(pos, frm, to):
    """Legal action moving from PDN square ``frm`` to ``to`` (1-based)."""
    matches = [
        a
        for a in pos.legal_actions()
        if A_FROM[a] == frm - 1 and A_TO[a] == to - 1
    ]
    assert len(matches) == 1, f'{frm}->{to} not uniquely legal in\n{pos}'
    return matches[0]


def play_moves(pos, *moves):
    """Play a sequence of (from, to) moves given as PDN squares."""
    for frm, to in moves:
        pos = pos.play(action_for(pos, frm, to))
    return pos


# Published perft counts for the initial position (a multi-jump counts as one
# move)
PERFT = [7, 49, 302, 1469, 7361, 36768, 179740, 845931]


@pytest.mark.parametrize('depth', range(1, 8))
def test_perft(depth):
    """Move counts from the start match the published perft numbers."""
    assert perft(Position.initial(), depth) == PERFT[depth - 1]


@pytest.mark.slow
def test_perft_depth_8():
    """The same at depth 8 (about 2 seconds)."""
    assert perft(Position.initial(), 8) == PERFT[7]


def test_initial_position():
    """12 men each, and Black's 7 opening moves."""
    pos = Position.initial()
    assert pos.turn == BLACK
    assert bin(pos.black).count('1') == 12 and bin(pos.white).count('1') == 12
    assert pos.result() is None
    moves = sorted(pos.move_text(m) for m, _ in pos.complete_moves())
    assert moves == [
        '10-14',
        '10-15',
        '11-15',
        '11-16',
        '12-16',
        '9-13',
        '9-14',
    ]


def test_action_geometry_is_unique():
    """Exactly 170 on-board hops, each a distinct (from, to) pair."""
    pairs = [(A_FROM[a], A_TO[a]) for a in range(ACTION_SIZE) if A_TO[a] >= 0]
    assert len(pairs) == len(set(pairs)) == 170


def test_capture_is_compulsory():
    """When a capture exists, it is the only legal move."""
    pos = Position.from_fen('B:W14:B9,1')
    moves = [pos.move_text(m) for m, _ in pos.complete_moves()]
    assert moves == ['9x18']


def test_multi_jump_continues_with_same_piece():
    """After a jump the same piece continues; the full move is one turn."""
    pos = Position.from_fen('B:W6,15:B1')
    after_first = pos.play(action_for(pos, 1, 10))
    assert after_first.turn == BLACK and after_first.jump_sq == 9
    assert len(after_first.legal_actions()) == 1

    ((actions, final),) = pos.complete_moves()
    assert pos.move_text(actions) == '1x10x19'
    assert final.turn == WHITE and final.white == 0
    assert final.result() == -1


def test_crowning_ends_the_turn():
    """A man reaching the last row is crowned and its turn ends."""
    pos = Position.from_fen('B:W26,27:B22')
    after = pos.play(action_for(pos, 22, 31))
    assert after.turn == WHITE and after.jump_sq == -1
    assert after.kings >> 30 & 1


def test_kings_move_backwards_men_do_not():
    """Kings move in all four directions, men only forward."""
    king = Position.from_fen('B:W32:BK15')
    assert {A_TO[a] + 1 for a in king.legal_actions()} == {10, 11, 18, 19}
    man = Position.from_fen('B:W32:B15')
    assert {A_TO[a] + 1 for a in man.legal_actions()} == {18, 19}


def test_men_do_not_capture_backwards():
    """Men cannot capture backwards."""
    pos = Position.from_fen('B:W11:B15')
    assert all(A_TO[a] + 1 in (18, 19) for a in pos.legal_actions())


def test_blocked_side_loses():
    """A side with no legal move loses."""
    pos = Position.from_fen('W:W29:B25,22')
    assert pos.legal_actions() == []
    assert pos.result() == -1


def test_threefold_repetition_draws():
    """The third occurrence of a position is a draw."""
    pos = Position.from_fen('B:WK32:BK1')
    shuffle = [(1, 5), (32, 28), (5, 1), (28, 32)]
    pos = play_moves(pos, *shuffle)
    assert pos.reps == 2 and pos.result() is None
    pos = play_moves(pos, *shuffle)
    assert pos.reps == 3 and pos.result() == 0


def test_no_progress_draw_and_reset_by_man_move():
    """Too long without progress draws; a man move resets the count."""
    rules = Rules(draw_plies=4, repetition_limit=99)
    pos = Position.from_fen('B:WK32,29:BK1', rules)
    pos = play_moves(pos, (1, 5), (32, 28), (5, 9))
    assert pos.no_progress == 3 and pos.result() is None
    pos = play_moves(pos, (29, 25))  # man move resets the counter
    assert pos.no_progress == 0
    pos = play_moves(pos, (9, 13), (28, 24), (13, 9), (24, 28))
    assert pos.result() == 0


def test_fen_round_trip_and_ranges():
    """FEN converts both ways, with ranges and the [FEN "..."] tag."""
    assert Position.from_fen('B:W21-32:B1-12').key == Position.initial().key
    fen = 'W:WK3,18,22:B9,K27'
    pos = Position.from_fen(fen)
    assert pos.to_fen() == fen
    assert Position.from_fen(f'[FEN "{fen}"]').key == pos.key


@pytest.mark.parametrize('fen', ['X:W1:B2', 'B:W1:B1', 'B:W33:B2', 'B:W1:B30'])
def test_invalid_fen_rejected(fen):
    """Malformed or impossible FENs are rejected."""
    with pytest.raises(ValueError):
        Position.from_fen(fen)
