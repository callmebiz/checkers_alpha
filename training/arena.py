"""The arena: measure playing strength by playing matches.

Training losses say how well the network predicts its own search, not how
well it plays. The arena answers that directly, with matches against:

- **Fixed opponents** such as alpha-beta at a few depths. They never change,
  so their scores are comparable across the whole run (and across runs).
- **The previous evaluated checkpoint**, which gives an Elo chain: each
  checkpoint's rating is the previous one's plus the rating difference
  implied by their head-to-head score.

Fair matches
------------
Games start from random openings (a few random plies from the start
position), for variety. Each opening is played twice with colours swapped,
so an opening that favours one side can't favour either contestant. The
openings come from a fixed seed, so every evaluation uses the same set.

Speed
-----
Like self-play, all games of a match run at once. Whenever several games
wait on the same network, their searches are batched into one network call
per round. Players without a network (alpha-beta, random) just move.

Elo
---
A score ``s`` (wins + half the draws, over games) means a rating difference
of ``-400 * log10(1 / s - 1)``: 0.5 is even, 0.64 is about +100, 0.76 about
+200. A perfect or zero score is clamped to half a game from the edge so the
difference stays finite.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.game import BLACK, COLOR_NAMES, WHITE, Position, Rules
from core.inference import Evaluator
from core.mcts import SearchConfig, SearchTree, run_search
from core.players import Player


@dataclass
class Contestant:
    """An arena participant: a network searched with MCTS, or a plain player.

    Network contestants are searched by the arena itself, one tree per game,
    so their searches can be batched. Pass a ``Player`` only for players that
    keep no per-game state (alpha-beta, random, raw policy).
    """

    name: str
    evaluator: Evaluator | None = None
    sims: int = 0
    player: Player | None = None

    @classmethod
    def network(cls, name: str, evaluator: Evaluator, sims: int) -> Contestant:
        """A contestant that searches with MCTS on ``evaluator``."""
        return cls(name, evaluator=evaluator, sims=sims)

    @classmethod
    def of(cls, player: Player) -> Contestant:
        """A contestant that is a plain player (alpha-beta, random, ...)."""
        return cls(player.name, player=player)


@dataclass
class MatchResult:
    """Outcome of a match, from contestant ``a``'s point of view."""

    a: str
    b: str
    wins: int = 0
    draws: int = 0
    losses: int = 0
    games: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Games played."""
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Wins plus half the draws, as a fraction of the games."""
        return (self.wins + 0.5 * self.draws) / max(self.total, 1)

    @property
    def elo_diff(self) -> float:
        """The rating difference the score implies (see ``elo_diff``)."""
        return elo_diff(self.score, self.total)

    def summary(self) -> dict[str, Any]:
        """The result as a JSON-friendly dict."""
        return {
            'opponent': self.b,
            'wins': self.wins,
            'draws': self.draws,
            'losses': self.losses,
            'score': round(self.score, 3),
            'elo_diff': round(self.elo_diff, 1),
        }


def elo_diff(score: float, games: int) -> float:
    """Rating difference implied by ``score`` over ``games`` games."""
    edge = 0.5 / max(games, 1)
    s = min(max(score, edge), 1 - edge)
    return -400 * math.log10(1 / s - 1)


def random_openings(
    count: int, plies: int, rng: np.random.Generator, rules: Rules
) -> list[Position]:
    """``count`` distinct positions reached by ``plies`` random plies."""
    openings: dict[tuple[int, ...], Position] = {}
    for _ in range(count * 50):
        if len(openings) == count:
            break
        pos = Position.initial(rules)
        for _ in range(plies):
            pos = pos.play(int(rng.choice(pos.legal_actions())))
            if pos.is_terminal:
                break
        while pos.jump_sq >= 0:  # finish a multi-jump in progress
            pos = pos.play(pos.legal_actions()[0])
        if not pos.is_terminal:
            openings.setdefault(pos.key, pos)
    return list(openings.values())


@dataclass
class _Game:
    pos: Position
    seats: tuple[Contestant, Contestant]  # indexed by color
    a_color: int
    opening: str
    trees: list[SearchTree | None] = field(default_factory=lambda: [None, None])
    actions: list[int] = field(default_factory=list)


def play_match(
    a: Contestant,
    b: Contestant,
    openings: list[Position],
    max_plies: int = 300,
    search: SearchConfig | None = None,
    on_game_end: Callable[[], None] | None = None,
) -> MatchResult:
    """Play every opening twice, once with ``a`` as each colour.

    ``on_game_end`` is called as each game finishes (for progress bars).
    """
    games = []
    for opening in openings:
        fen = opening.to_fen()
        games.append(_Game(opening, (a, b), BLACK, fen))
        games.append(_Game(opening, (b, a), WHITE, fen))
    _play_all(games, max_plies, search or SearchConfig(), on_game_end)

    result = MatchResult(a.name, b.name)
    for game in games:
        outcome = game.pos.result()
        winner = 1 - game.pos.turn if outcome == -1 else None
        if winner is None:
            result.draws += 1
        elif winner == game.a_color:
            result.wins += 1
        else:
            result.losses += 1
        result.games.append(
            {
                'start': game.opening,
                'black': game.seats[BLACK].name,
                'white': game.seats[WHITE].name,
                'winner': None if winner is None else COLOR_NAMES[winner],
                'plies': len(game.actions),
                'actions': game.actions,
            }
        )
    return result


def _play_all(
    games: list[_Game],
    max_plies: int,
    search: SearchConfig,
    on_game_end: Callable[[], None] | None,
) -> None:
    """Play ``games`` to the end, batching network searches across games."""

    def over(g: _Game) -> bool:
        """Whether ``g`` has ended or hit the ply cap."""
        return g.pos.is_terminal or len(g.actions) >= max_plies

    def play(g: _Game, action: int) -> None:
        """Play ``action`` in ``g`` and record it."""
        g.pos = g.pos.play(action)
        g.actions.append(action)

    active = [g for g in games if not over(g)]
    while active:
        # contestant id -> (contestant, games waiting on its search)
        waiting: dict[int, tuple[Contestant, list[tuple[_Game, SearchTree]]]]
        waiting = {}
        for g in active:
            while not over(g):
                seat = g.seats[g.pos.turn]
                legal = g.pos.legal_actions()
                if len(legal) == 1:
                    play(g, legal[0])
                elif seat.player is not None:
                    play(g, seat.player.choose(g.pos))
                else:
                    tree = g.trees[g.pos.turn]
                    if tree is None:
                        tree = SearchTree(g.pos, search)
                        g.trees[g.pos.turn] = tree
                    else:
                        tree.reroot(g.pos)
                    waiting.setdefault(id(seat), (seat, []))[1].append(
                        (g, tree)
                    )
                    break
        for seat, items in waiting.values():
            assert seat.evaluator is not None
            run_search([t for _, t in items], seat.evaluator, seat.sims)
            for g, tree in items:
                play(g, tree.best_action())
        still = [g for g in active if not over(g)]
        if on_game_end is not None:
            for _ in range(len(active) - len(still)):
                on_game_end()
        active = still
