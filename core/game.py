"""American checkers (English draughts) rules on 32-square bitboards.

Board
-----
Only the 32 dark squares are playable. They use standard PDN numbering: 1-32 in
text, 0-31 in code. Each player's near-left corner is dark (square 29 for White,
square 4 for Black)::

         1   2   3   4       row 0   Black starts on 1-12, moves down, first
       5   6   7   8         row 1
         9  10  11  12
      13  14  15  16
        17  18  19  20
      21  22  23  24
        25  26  27  28
      29  30  31  32         row 7   White starts on 21-32 and moves up

A *bitboard* is an int whose bit ``s`` is set when square ``s`` (0-based) holds
a piece. Three bitboards describe the pieces: ``black``, ``white`` and ``kings``
(either color). Set operations are bit operations: ``black | white`` is every
occupied square, ``black & kings`` is Black's kings, ``x & -x`` isolates the
lowest set bit.

Rules
-----
- Men move one square diagonally forward; kings move diagonally both ways.
- Captures jump an adjacent enemy piece onto the empty square beyond. Capturing
  is compulsory: if any capture exists, only captures are legal.
- After a capture, the same piece must keep capturing while it can (a
  multi-jump).
- A man reaching the far row is crowned; this ends the turn even mid multi-jump.
- The side to move with no legal move (including no pieces) loses.
- Draw after ``Rules.draw_plies`` plies with no capture or man move, or when the
  same position occurs ``Rules.repetition_limit`` times.

Actions
-------
An action is one *hop*: a step or a single jump. It is encoded as ``square * 8 +
direction * 2 + is_jump`` (256 values). Only 170 of them fit on the board; the
rest are never legal. A multi-jump is several jump actions by the same side, and
``Position.jump_sq`` marks the piece that must continue.

Positions are immutable: ``pos.play(action)`` returns a new ``Position``. Every
position in a game is produced by ``Position.initial()``,
``Position.from_fen()`` (validated) or ``play()`` of a legal action, so the game
never holds an illegal state.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

BLACK, WHITE = 0, 1
COLOR_NAMES = ('black', 'white')
NUM_SQUARES = 32
ACTION_SIZE = 256
FULL = (1 << NUM_SQUARES) - 1

# Directions as (row delta, col delta). Row 0 is Black's back row, so "north" is
# toward it.
NW, NE, SW, SE = 0, 1, 2, 3
DIRECTIONS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
DIRECTION_NAMES = ('NW', 'NE', 'SW', 'SE')
# Indexed by color: the directions its men may move
FORWARD = ((SW, SE), (NW, NE))
# Index of the king entry in the move tables, after the BLACK and WHITE men
KING_TABLE = 2


@dataclass(frozen=True)
class Rules:
    """Configurable draw rules. Movement and capture rules are fixed."""

    draw_plies: int = 80  # plies with no capture or man move (40 moves each)
    repetition_limit: int = 3


DEFAULT_RULES = Rules()


def square_to_rc(s: int) -> tuple[int, int]:
    """(row, col) of square ``s`` (0-based) on the 8x8 board."""
    r = s >> 2
    return r, 2 * (s & 3) + (1 - (r & 1))


def rc_to_square(r: int, c: int) -> int:
    """Square index at (row, col); -1 for a light or off-board square."""
    if not (0 <= r < 8 and 0 <= c < 8) or (r + c) % 2 == 0:
        return -1
    return r * 4 + c // 2


def make_action(square: int, direction: int, is_jump: bool | int) -> int:
    """Encode a hop from ``square`` in ``direction`` as an action index."""
    return square * 8 + direction * 2 + int(is_jump)


def iter_squares(bits: int) -> Iterator[int]:
    """Yield the index of every set bit, lowest first."""
    while bits:
        lsb = bits & -bits
        yield lsb.bit_length() - 1
        bits ^= lsb


# ------------------------------------------------------------------------------
# Precomputed geometry: lookup tables of every hop that fits on the board,
# built once when the module is imported. Move generation and play() then use
# table lookups and bit tests instead of coordinate arithmetic.


def _action_tables() -> tuple[list[int], ...]:
    """Source, destination and captured square of every action.

    Returns six lists indexed by action: ``from``, ``to`` and ``mid``
    squares (``to`` is -1 when the hop would leave the board, ``mid`` is -1
    for steps), then the same three as single-bit masks (0 when not
    applicable), so ``play()`` can update bitboards without branching.
    """
    frm, to, mid = [-1] * ACTION_SIZE, [-1] * ACTION_SIZE, [-1] * ACTION_SIZE
    frm_bit, to_bit = [0] * ACTION_SIZE, [0] * ACTION_SIZE
    mid_bit = [0] * ACTION_SIZE
    for s in range(NUM_SQUARES):
        r, c = square_to_rc(s)
        for d, (dr, dc) in enumerate(DIRECTIONS):
            for jump in (0, 1):
                a = make_action(s, d, jump)
                dist = 2 if jump else 1
                dest = rc_to_square(r + dist * dr, c + dist * dc)
                frm[a], frm_bit[a] = s, 1 << s
                if dest < 0:
                    continue  # off the board: never legal
                to[a], to_bit[a] = dest, 1 << dest
                if jump:
                    over = rc_to_square(r + dr, c + dc)
                    mid[a], mid_bit[a] = over, 1 << over
    return frm, to, mid, frm_bit, to_bit, mid_bit


A_FROM, A_TO, A_MID, A_FROM_BIT, A_TO_BIT, A_MID_BIT = _action_tables()

# The 170 actions that stay on the board
ON_BOARD_ACTIONS = [a for a in range(ACTION_SIZE) if A_TO[a] >= 0]

StepEntry = tuple[int, int]  # (destination bit, action)
JumpEntry = tuple[int, int, int]  # (captured bit, destination bit, action)


def _step_table(dirs: tuple[int, ...]) -> tuple[tuple[StepEntry, ...], ...]:
    """For each square, the steps in ``dirs`` that stay on the board."""
    table = []
    for s in range(NUM_SQUARES):
        actions = [make_action(s, d, 0) for d in dirs]
        table.append(tuple((A_TO_BIT[a], a) for a in actions if A_TO[a] >= 0))
    return tuple(table)


def _jump_table(dirs: tuple[int, ...]) -> tuple[tuple[JumpEntry, ...], ...]:
    """For each square, the jumps in ``dirs`` that stay on the board."""
    table = []
    for s in range(NUM_SQUARES):
        actions = [make_action(s, d, 1) for d in dirs]
        table.append(
            tuple(
                (A_MID_BIT[a], A_TO_BIT[a], a) for a in actions if A_TO[a] >= 0
            )
        )
    return tuple(table)


_ALL_DIRS = (NW, NE, SW, SE)
# Indexed [BLACK man, WHITE man, KING_TABLE][square] -> candidate hops for that
# piece
STEPS = (
    _step_table(FORWARD[BLACK]),
    _step_table(FORWARD[WHITE]),
    _step_table(_ALL_DIRS),
)
JUMPS = (
    _jump_table(FORWARD[BLACK]),
    _jump_table(FORWARD[WHITE]),
    _jump_table(_ALL_DIRS),
)

# Far row where each color's men are crowned
PROMOTION_ROW = (
    sum(1 << s for s in range(28, 32)),
    sum(1 << s for s in range(4)),
)


# ------------------------------------------------------------------------------


class Position:
    """An immutable game state.

    Create the start with ``Position.initial()`` or load one with
    ``Position.from_fen()``, then advance with ``play(action)`` using an action
    from ``legal_actions()``.

    Attributes:
        black, white, kings: bitboards of the pieces.
        turn: side to move, ``BLACK`` or ``WHITE``.
        jump_sq: square of the piece that must continue a multi-jump, else -1.
        no_progress: plies since the last capture or man move (draw rule).
        reps: how many times this position has occurred, counting this one.
        prev: previous position. Only kept while ``no_progress > 0``: a
            capture or man move can't be undone, so earlier positions can
            never repeat.
        rules: draw rules this game uses.
    """

    __slots__ = (
        'black', 'white', 'kings', 'turn', 'jump_sq', 'no_progress', 'reps',
        'prev', 'rules', '_legal', '_result', '_result_known',
    )  # fmt: skip

    black: int
    white: int
    kings: int
    turn: int
    jump_sq: int
    no_progress: int
    reps: int
    prev: Position | None
    rules: Rules
    _legal: list[int] | None
    _result: int | None
    _result_known: bool

    def __init__(
        self,
        black: int,
        white: int,
        kings: int,
        turn: int,
        jump_sq: int = -1,
        no_progress: int = 0,
        prev: Position | None = None,
        rules: Rules = DEFAULT_RULES,
    ) -> None:
        """Build from bitboards; prefer ``initial``/``from_fen``/``play``."""
        self.black = black
        self.white = white
        self.kings = kings
        self.turn = turn
        self.jump_sq = jump_sq
        self.no_progress = no_progress
        self.prev = prev
        self.rules = rules
        self._legal = None
        self._result = None
        self._result_known = False
        self.reps = self._count_repetitions()

    @classmethod
    def initial(cls, rules: Rules = DEFAULT_RULES) -> Position:
        """The start: Black on 1-12, White on 21-32, Black to move."""
        return cls((1 << 12) - 1, FULL ^ ((1 << 20) - 1), 0, BLACK, rules=rules)

    # -- state -----------------------------------------------------------------

    @property
    def key(self) -> tuple[int, int, int, int, int]:
        """Hashable identity of the position. Ignores the draw counters."""
        return (self.black, self.white, self.kings, self.turn, self.jump_sq)

    def _count_repetitions(self) -> int:
        # Every position within the no-progress window was reached without
        # captures, so sides alternate each ply and only positions an even
        # number of plies back can match.
        """How many times this position has occurred, counting this one."""
        if self.no_progress < 4:
            return 1
        count = 1
        p = self.prev.prev if self.prev is not None else None
        back = 2
        while p is not None and back <= self.no_progress:
            if (
                p.black == self.black
                and p.white == self.white
                and p.kings == self.kings
            ):
                count += 1
            p = p.prev.prev if p.prev is not None else None
            back += 2
        return count

    # -- moves -----------------------------------------------------------------

    def legal_actions(self) -> list[int]:
        """Legal hops for the side to move; empty when it has lost.

        Computed once, then cached.
        """
        if self._legal is None:
            self._legal = self._generate()
        return self._legal

    def _generate(self) -> list[int]:
        """Compute the legal hops: all captures if any exist, else all steps."""
        turn = self.turn
        own, opp = (
            (self.white, self.black) if turn else (self.black, self.white)
        )
        empty = FULL & ~(own | opp)
        kings = self.kings
        # Mid multi-jump, only the jumping piece may move
        pieces = own if self.jump_sq < 0 else 1 << self.jump_sq

        # Captures first: if any exist, they are the only legal moves
        man_jumps, king_jumps = JUMPS[turn], JUMPS[KING_TABLE]
        jumps: list[int] = []
        b = pieces
        while b:
            lsb = b & -b  # one piece at a time
            b ^= lsb
            for mid, to, a in (king_jumps if kings & lsb else man_jumps)[
                lsb.bit_length() - 1
            ]:
                if opp & mid and empty & to:
                    jumps.append(a)
        if jumps or self.jump_sq >= 0:
            return jumps

        man_steps, king_steps = STEPS[turn], STEPS[KING_TABLE]
        steps: list[int] = []
        b = pieces
        while b:
            lsb = b & -b
            b ^= lsb
            for to, a in (king_steps if kings & lsb else man_steps)[
                lsb.bit_length() - 1
            ]:
                if empty & to:
                    steps.append(a)
        return steps

    def play(self, action: int) -> Position:
        """The position after ``action``.

        ``action`` must come from ``legal_actions()``. It isn't re-checked here,
        for speed.
        """
        frm, to, mid = A_FROM_BIT[action], A_TO_BIT[action], A_MID_BIT[action]
        turn = self.turn
        black, white, kings = self.black, self.white, self.kings
        move = frm | to  # XOR with this moves a piece from `frm` to `to`
        was_king = kings & frm
        if turn:
            white ^= move
            black &= ~mid  # mid is 0 for steps, so this is a no-op
        else:
            black ^= move
            white &= ~mid
        if was_king:
            kings ^= move
        kings &= ~mid
        crowned = not was_king and bool(to & PROMOTION_ROW[turn])
        if crowned:
            kings |= to

        rules = self.rules
        if mid:
            # After a capture the same piece continues if it can capture again
            # (unless it was just crowned).
            if not crowned:
                opp = black if turn else white
                empty = FULL & ~(black | white)
                table = JUMPS[KING_TABLE if was_king else turn]
                for m, t, _ in table[A_TO[action]]:
                    if opp & m and empty & t:
                        return Position(
                            black,
                            white,
                            kings,
                            turn,
                            A_TO[action],
                            0,
                            None,
                            rules,
                        )
            return Position(black, white, kings, 1 - turn, -1, 0, None, rules)

        # A man move is irreversible, so it resets the draw counter and the
        # repetition history
        no_progress = self.no_progress + 1 if was_king else 0
        prev = self if no_progress else None
        return Position(
            black, white, kings, 1 - turn, -1, no_progress, prev, rules
        )

    def complete_moves(self) -> list[tuple[tuple[int, ...], Position]]:
        """Every full turn from here as ``(actions, resulting position)``.

        Multi-jumps are expanded to their full hop sequences.
        """
        out: list[tuple[tuple[int, ...], Position]] = []

        def expand(pos: Position, path: tuple[int, ...]) -> None:
            """Follow every hop from ``pos``, recursing through multi-jumps."""
            for a in pos.legal_actions():
                nxt = pos.play(a)
                if nxt.jump_sq >= 0:
                    expand(nxt, path + (a,))
                else:
                    out.append((path + (a,), nxt))

        expand(self, ())
        return out

    # -- outcome ---------------------------------------------------------------

    def result(self) -> int | None:
        """Outcome for the side to move, or ``None`` while the game continues.

        -1 means lost (no legal moves), 0 drawn. A game never ends with the side
        to move winning.
        """
        if not self._result_known:
            if not self.legal_actions():
                self._result = -1
            elif (
                self.no_progress >= self.rules.draw_plies
                or self.reps >= self.rules.repetition_limit
            ):
                self._result = 0
            self._result_known = True
        return self._result

    @property
    def is_terminal(self) -> bool:
        """Whether the game is over (lost or drawn for the side to move)."""
        return self.result() is not None

    # -- text ------------------------------------------------------------------

    def to_fen(self) -> str:
        """Standard PDN FEN, e.g. ``B:W21,22,K30:B1,2``.

        Raises ``ValueError`` mid multi-jump, which FEN can't express.
        """
        if self.jump_sq >= 0:
            raise ValueError(
                'FEN cannot express a position in the middle of a multi-jump'
            )

        def side(bits: int) -> str:
            """One colour's squares, comma-separated, kings prefixed with K."""
            return ','.join(
                ('K' if self.kings >> s & 1 else '') + str(s + 1)
                for s in iter_squares(bits)
            )

        to_move = 'W' if self.turn else 'B'
        return f'{to_move}:W{side(self.white)}:B{side(self.black)}'

    @classmethod
    def from_fen(cls, fen: str, rules: Rules = DEFAULT_RULES) -> Position:
        """Parse PDN FEN; ranges like ``1-12`` are allowed.

        Raises ``ValueError`` for impossible positions.
        """
        text = (
            fen.strip()
            .strip('[]')
            .replace('FEN', '')
            .strip()
            .strip('"')
            .rstrip('.')
        )
        fields = [f.strip() for f in text.split(':') if f.strip()]
        if not fields or fields[0].upper() not in ('B', 'W'):
            raise ValueError(f'Bad FEN side to move: {fen!r}')
        turn = WHITE if fields[0].upper() == 'W' else BLACK
        bits = {BLACK: 0, WHITE: 0}
        kings = 0
        for field in fields[1:]:
            color = {'B': BLACK, 'W': WHITE}.get(field[0].upper())
            if color is None:
                raise ValueError(f'Bad FEN field {field!r}')
            for item in filter(None, (i.strip() for i in field[1:].split(','))):
                king = item[0].upper() == 'K'
                item = item[1:] if king else item
                lo, _, hi = item.partition('-')
                for n in range(int(lo), int(hi or lo) + 1):
                    if not 1 <= n <= 32:
                        raise ValueError(f'Square {n} out of range')
                    b = 1 << (n - 1)
                    if (bits[BLACK] | bits[WHITE]) & b:
                        raise ValueError(f'Square {n} occupied twice')
                    bits[color] |= b
                    if king:
                        kings |= b
        for color in (BLACK, WHITE):
            if bits[color] & ~kings & PROMOTION_ROW[color]:
                raise ValueError(
                    f'{COLOR_NAMES[color]} man on its crowning row'
                )
            if (bits[color]).bit_count() > 12:
                raise ValueError(
                    f'{COLOR_NAMES[color]} has more than 12 pieces'
                )
        return cls(bits[BLACK], bits[WHITE], kings, turn, rules=rules)

    def move_text(self, actions: tuple[int, ...] | list[int]) -> str:
        """PDN notation for a full turn, e.g. ``11-15`` or ``22x15x6``."""
        if A_MID[actions[0]] < 0:
            return f'{A_FROM[actions[0]] + 1}-{A_TO[actions[0]] + 1}'
        return 'x'.join(
            [str(A_FROM[actions[0]] + 1)] + [str(A_TO[a] + 1) for a in actions]
        )

    def to_grid(self) -> list[list[int]]:
        """8x8 grid of ints: White 1 (king 2), Black -1 (king -2)."""
        grid = [[0] * 8 for _ in range(8)]
        for bits, sign in ((self.white, 1), (self.black, -1)):
            for s in iter_squares(bits):
                r, c = square_to_rc(s)
                grid[r][c] = sign * (2 if self.kings >> s & 1 else 1)
        return grid

    def __str__(self) -> str:
        """ASCII board (Black at the top) and whose turn it is."""
        glyph = {0: '.', 1: 'w', 2: 'W', -1: 'b', -2: 'B'}
        rows = [
            ' '.join(
                glyph[v] if (r + c) % 2 else ' ' for c, v in enumerate(row)
            )
            for r, row in enumerate(self.to_grid())
        ]
        to_move = COLOR_NAMES[self.turn]
        if self.jump_sq >= 0:
            to_move += f' (continuing jump from {self.jump_sq + 1})'
        return '\n'.join(rows) + f'\n{to_move} to move'

    def __repr__(self) -> str:
        """Short form for debugging: the FEN when there is one."""
        try:
            return f'Position({self.to_fen()!r})'
        except ValueError:
            return f'Position(key={self.key})'


def perft(pos: Position, depth: int) -> int:
    """Count full-turn move sequences of length ``depth``.

    A multi-jump counts as one move.

    Checked against published counts to validate move generation.
    """
    if depth == 0:
        return 1
    moves = pos.complete_moves()
    if depth == 1:
        return len(moves)
    return sum(perft(nxt, depth - 1) for _, nxt in moves)
