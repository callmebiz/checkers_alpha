#!/usr/bin/env python3
"""Play checkers in the terminal, or run an engine-vs-engine match.

Player specs: human | random | ab[:depth] | policy:<checkpoint> |
              mcts:<checkpoint|uniform|rollout>[:sims]

Examples:
  # you (Black) vs alpha-beta depth 6
  python play.py
  python play.py --black ab:4 --white human
  python play.py --white mcts:runs/dev/checkpoints/latest.pt:800 --analysis
  # a 20-game match, colours alternate
  python play.py --black mcts:rollout:200 --white ab:4 --games 20
"""

from __future__ import annotations

import argparse
import time

from core.game import BLACK, COLOR_NAMES, WHITE, Position, Rules
from core.players import MCTSPlayer, Player, PolicyPlayer, make_player

# Square numbers laid out like the board, shown by the `?` command
SQUARE_GUIDE = '\n'.join(
    ('   ' if r % 2 == 0 else '')
    + '    '.join(f'{r * 4 + i + 1:>2}' for i in range(4))
    for r in range(8)
)

Seats = dict[int, Player | None]  # color -> player, None for a human


def print_board(pos: Position, last_move: str | None = None) -> None:
    """Print the board and, if given, the last move."""
    print()
    print(pos)
    if last_move:
        print(f'last move: {last_move}')


def outcome_text(pos: Position) -> str:
    """'Draw.' or which colour won, for a finished position."""
    if pos.result() == 0:
        return 'Draw.'
    return f'{COLOR_NAMES[1 - pos.turn].capitalize()} wins.'


def describe_search(player: Player) -> str | None:
    """One-line summary of an engine player's last decision."""
    if (
        isinstance(player, MCTSPlayer)
        and player.tree is not None
        and player.tree.root.expanded
    ):
        tree = player.tree
        pos = tree.root.pos
        top = sorted(tree.move_stats(), key=lambda s: -s['visits'])[:3]
        moves = ', '.join(
            f'{pos.move_text([s["action"]])} N={s["visits"]} '
            f'Q={s["q"] or 0:+.2f} P={s["prior"]:.2f}'
            for s in top
        )
        return f'  search: value {tree.root.q():+.2f} | {moves}'
    if isinstance(player, PolicyPlayer) and player.last is not None:
        _, value, wdl = player.last
        win, draw, loss = wdl
        return (
            f'  net: value {value:+.2f}  W/D/L {win:.2f}/{draw:.2f}/{loss:.2f}'
        )
    return None


def human_turn(pos: Position) -> tuple[Position, str] | str:
    """Prompt for a full move (e.g. ``11-15`` or ``22x15x6``).

    Returns ``(new position, move text)``, or the command ``'undo'`` /
    ``'quit'``.
    """
    moves = {
        pos.move_text(actions): nxt for actions, nxt in pos.complete_moves()
    }
    while True:
        text = (
            input(f'{COLOR_NAMES[pos.turn]} move (? for help): ')
            .strip()
            .lower()
        )
        text = text.replace(' ', '')
        if text in ('q', 'quit', 'exit'):
            return 'quit'
        if text == 'undo':
            return 'undo'
        if text in ('?', 'h', 'help'):
            print('Squares:\n' + SQUARE_GUIDE)
            print('Legal moves: ' + '  '.join(sorted(moves)))
            print('Commands: undo, fen, quit')
        elif text == 'fen':
            print(pos.to_fen())
        elif text in moves:
            return moves[text], text
        else:
            # Also accept the start and end of a multi-jump, e.g. "1x19" for
            # "1x10x19"
            matches = [
                m
                for m in moves
                if m.split('x')[0] + 'x' + m.split('x')[-1] == text
            ]
            if len(matches) == 1:
                return moves[matches[0]], matches[0]
            print(f'Not a legal move. Legal: {"  ".join(sorted(moves))}')


def engine_turn(player: Player, pos: Position) -> tuple[Position, str]:
    """Let an engine play a full turn (several hops for a multi-jump)."""
    start, actions = pos, []
    while True:
        actions.append(player.choose(pos))
        pos = pos.play(actions[-1])
        if pos.jump_sq < 0:
            return pos, start.move_text(actions)


def play_game(
    seats: Seats, start: Position, show: bool = True, analysis: bool = False
) -> Position | None:
    """Play one game from ``start``.

    Returns the final position, or None if a human quit.
    """
    for player in seats.values():
        if player is not None:
            player.reset()
    pos = start
    history = [start]  # position at the start of each turn, for undo
    if show:
        print_board(pos)
    while not pos.is_terminal:
        player = seats[pos.turn]
        if player is None:
            answer = human_turn(pos)
            if answer == 'quit':
                return None
            if answer == 'undo':
                pos = undo(history, seats)
                print_board(pos)
                continue
            assert isinstance(answer, tuple)
            pos, move = answer
        else:
            t0 = time.perf_counter()
            pos, move = engine_turn(player, pos)
            if show and analysis:
                took = time.perf_counter() - t0
                print(f'{player.name} played {move} in {took:.1f}s')
                info = describe_search(player)
                if info:
                    print(info)
        history.append(pos)
        if show:
            print_board(pos, move)
    if show:
        print('\n' + outcome_text(pos))
    return pos


def undo(history: list[Position], seats: Seats) -> Position:
    """Step back to this human's previous turn.

    Engine replies in between are skipped.
    """
    if len(history) < 2:
        print('Nothing to undo.')
        return history[-1]
    history.pop()
    while len(history) > 1 and seats[history[-1].turn] is not None:
        history.pop()
    return history[-1]


def run_match(a: Player, b: Player, start: Position, games: int) -> None:
    """Play a match between ``a`` and ``b`` and print the score.

    Colours alternate each game.
    """
    score = {'A wins': 0, 'draws': 0, 'B wins': 0}
    t0 = time.perf_counter()
    for g in range(games):
        a_color = BLACK if g % 2 == 0 else WHITE
        end = play_game({a_color: a, 1 - a_color: b}, start, show=False)
        assert end is not None
        if end.result() == 0:
            score['draws'] += 1
        else:
            winner = 1 - end.turn
            score['A wins' if winner == a_color else 'B wins'] += 1
        print(
            f'game {g + 1}/{games}: {outcome_text(end)}  '
            f'(A played {COLOR_NAMES[a_color]})  running: {score}'
        )
    points = score['A wins'] + 0.5 * score['draws']
    print(
        f'\nA={a.name} vs B={b.name}: {score}, A score {points / games:.1%} '
        f'in {time.perf_counter() - t0:.0f}s'
    )


def main() -> None:
    """Parse the arguments, then play one game or a match."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--black', default='human', help='player spec for Black (moves first)'
    )
    parser.add_argument('--white', default='ab:6', help='player spec for White')
    parser.add_argument('--fen', default=None, help='start from this position')
    parser.add_argument(
        '--games', type=int, default=1, help='play N games, alternating colours'
    )
    parser.add_argument(
        '--analysis', action='store_true', help='print engine thinking'
    )
    parser.add_argument(
        '--device', default='cuda', help='device for network players'
    )
    parser.add_argument('--draw-plies', type=int, default=Rules.draw_plies)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    rules = Rules(draw_plies=args.draw_plies)
    start = (
        Position.from_fen(args.fen, rules)
        if args.fen
        else Position.initial(rules)
    )

    def build(spec: str) -> Player | None:
        """The player for ``spec``, or None for a human."""
        return (
            None
            if spec == 'human'
            else make_player(spec, args.device, args.seed)
        )

    black, white = build(args.black), build(args.white)
    if args.games == 1:
        human_playing = black is None or white is None
        play_game(
            {BLACK: black, WHITE: white},
            start,
            analysis=args.analysis or human_playing,
        )
    elif black is None or white is None:
        parser.error('--games needs two engine players')
    else:
        run_match(black, white, start, args.games)


if __name__ == '__main__':
    main()
