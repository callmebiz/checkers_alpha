"""Self-play: the network plays itself to generate training data.

Many games at once
------------------
A network call on the GPU costs about the same for 1 position as for 256, so
``SelfPlay`` keeps up to ``concurrent_games`` games in progress, each with its
own search tree. Every round, each game contributes one leaf, and all leaves
go to the network in a single batch.

Games don't wait for each other: a game whose search is done plays its move
and starts the next search straight away, in the same round. Full and fast
searches (below) take different numbers of rounds, and in lockstep the fast
games would sit idle while the full ones finished, shrinking the batches.

The games are split into two groups that take turns: while the GPU works on
one group's batch (``core.inference.submit`` returns without waiting), the
CPU backs up the other group's results and walks its trees to the next
leaves. Neither the CPU nor the GPU sits idle waiting for the other.

The pool never drains: when a game ends, a new one takes its slot at once.
``play(n)`` returns after ``n`` games have finished, and games still in
progress carry over to the next call. The network has usually been trained
in between, so carried-over games start a fresh search tree.

Each move
---------
1. **Forced moves** (one legal move, common with compulsory captures) are
   played straight away. There's nothing to learn from them.
2. **Full or fast search**: with probability ``full_search_prob`` the move
   gets a full search (``sims``, root noise on) and is recorded as a training
   sample. Otherwise a fast search (``fast_sims``, no noise) just picks the
   move. This is KataGo's playout cap randomisation: more games per GPU-hour,
   so more game outcomes to learn values from.
3. **Choosing the move**: for the first ``temperature_plies`` plies the move
   is sampled from the visit counts (for opening variety), afterwards the
   most-visited move is played.
4. The chosen child becomes the new root (tree reuse), and its visits count
   toward the next search's budget.

When the game ends, every recorded position gets its value target: win, draw
or loss for the side that was to move there.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from core.encoding import canonical_actions, features, legal_mask
from core.game import ACTION_SIZE, COLOR_NAMES, Position, Rules
from core.inference import Evaluator, Pending, submit
from core.mcts import Leaf, SearchConfig, SearchTree, Selection
from training.config import SelfPlayConfig
from training.replay_buffer import DRAW, LOSS, WIN, Samples


@dataclass
class GameRecord:
    """A finished self-play game: its training samples and a move log."""

    start_fen: str
    # one entry per hop, for replay in the UI; only kept for the few games
    # chosen for saving (``logged``), to spare memory
    moves: list[dict[str, Any]]
    winner: int | None  # BLACK, WHITE, or None for a draw
    plies: int
    samples: Samples
    logged: bool = False

    def to_json(self) -> dict[str, Any]:
        """The game as JSON for the UI (without the samples)."""
        return {
            'start': self.start_fen,
            'winner': None if self.winner is None else COLOR_NAMES[self.winner],
            'plies': self.plies,
            'moves': self.moves,
        }


@dataclass
class SelfPlayStats:
    games: int = 0
    plies: int = 0
    black_wins: int = 0
    white_wins: int = 0
    draws: int = 0
    samples: int = 0
    sims: int = 0  # simulations run (visits kept by tree reuse excluded)
    evaluated: int = 0  # positions sent to the network
    batches: int = 0  # network calls
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """The statistics as a plain dict."""
        return asdict(self)


@dataclass
class _Game:
    """A game in progress."""

    tree: SearchTree
    start_fen: str
    ply: int = 0
    full: bool = False  # whether the current search is a full one
    logged: bool = False  # keep a move log (only for games to be saved)
    target: int = 0  # root visits at which the current search is done
    positions: list[Position] = field(default_factory=list)  # recorded
    targets: list[npt.NDArray[np.float32]] = field(default_factory=list)
    moves: list[dict[str, Any]] = field(default_factory=list)


class SelfPlay:
    """Plays batches of self-play games with one evaluator."""

    def __init__(
        self,
        evaluator: Evaluator,
        config: SelfPlayConfig,
        search: SearchConfig,
        rules: Rules,
        rng: np.random.Generator,
    ) -> None:
        """Games are searched with ``evaluator``; ``rng`` drives all chance."""
        self.evaluator = evaluator
        self.config = config
        self.search = search
        self.rules = rules
        self.rng = rng
        self.games: list[_Game] = []  # in progress, kept between calls
        self._to_log = 0  # new games still to be logged in this call

    def play(
        self,
        num_games: int,
        on_game_end: Callable[[GameRecord], None] | None = None,
    ) -> tuple[list[GameRecord], SelfPlayStats]:
        """Play ``num_games`` games; return their records and statistics.

        ``on_game_end`` is called as each game finishes (e.g. for a progress
        bar). A round can finish several games, so slightly more than
        ``num_games`` may be returned.
        """
        stats = SelfPlayStats()
        start_time = time.perf_counter()
        records: list[GameRecord] = []
        self._to_log = self.config.saved_games
        for game in self.games:  # the network changed since their last search
            game.tree = SearchTree(
                game.tree.root.pos, self.search, rng=self.rng
            )
            self._start_search(game)

        # Two groups take turns, so the GPU evaluates one group's batch
        # while the CPU backs up and selects leaves for the other.
        size = self.config.concurrent_games
        capacity = ((size + 1) // 2, size // 2)
        groups = [self.games[0::2], self.games[1::2]]
        in_flight: list[tuple[Pending, list[Leaf]] | None] = [None, None]
        g = 0
        while len(records) < num_games or any(in_flight):
            batch = in_flight[g]
            if batch is not None:
                self._backup_batch(*batch, stats)
                in_flight[g] = None
            if len(records) < num_games and capacity[g] > 0:
                while len(groups[g]) < capacity[g]:
                    groups[g].append(self._new_game())
                leaves: list[Leaf] = []
                playing: list[_Game] = []
                for game in groups[g]:
                    leaf = self._next_leaf(game, stats)
                    if leaf is None:
                        record = self._finish(game)
                        records.append(record)
                        if on_game_end is not None:
                            on_game_end(record)
                    else:
                        leaves.append(leaf)
                        playing.append(game)
                groups[g] = playing
                if leaves:
                    positions = [leaf.node.pos for leaf in leaves]
                    in_flight[g] = (submit(self.evaluator, positions), leaves)
            g = 1 - g
        self.games = groups[0] + groups[1]

        for record in records:
            stats.games += 1
            stats.plies += record.plies
            stats.samples += len(record.samples)
            if record.winner is None:
                stats.draws += 1
            elif record.winner == 0:
                stats.black_wins += 1
            else:
                stats.white_wins += 1
        stats.seconds = time.perf_counter() - start_time
        return records, stats

    def _backup_batch(
        self, pending: Pending, leaves: list[Leaf], stats: SelfPlayStats
    ) -> None:
        """Wait for a batch's evaluation and back it up into its trees."""
        priors, values, wdl = pending.result()
        for i, leaf in enumerate(leaves):
            leaf.tree.expand_and_backup(leaf, priors[i], values[i], wdl[i])
        stats.evaluated += len(leaves)
        stats.batches += 1
        stats.sims += len(leaves)

    # -- one game -------------------------------------------------------------

    def _new_game(self) -> _Game:
        """Start a game from the initial position and set up its search."""
        start = Position.initial(self.rules)
        tree = SearchTree(start, self.search, rng=self.rng)
        game = _Game(tree, start.to_fen(), logged=self._to_log > 0)
        self._to_log -= game.logged
        self._start_search(game)
        return game

    def _over(self, game: _Game) -> bool:
        """Whether ``game`` has ended or reached ``max_plies``."""
        return (
            game.tree.root.pos.is_terminal or game.ply >= self.config.max_plies
        )

    def _next_leaf(self, game: _Game, stats: SelfPlayStats) -> Leaf | None:
        """Run ``game`` forward until its search needs an evaluation.

        Plays any moves whose search is complete along the way. Returns the
        leaf to evaluate, or None once the game is over.
        """
        while not self._over(game):
            tree = game.tree
            if tree.root.visits >= game.target:
                self._move(game)
                self._start_search(game)
                continue
            selected = tree.select_leaf()
            if isinstance(selected, Leaf):
                return selected
            if selected is Selection.TERMINAL:
                stats.sims += 1
            # (a collision can't happen with one leaf in flight per tree)
        return None

    def _start_search(self, game: _Game) -> None:
        """Play forced moves, then set up a full or fast search."""
        cfg = self.config
        while not self._over(game):
            legal = game.tree.root.pos.legal_actions()
            if len(legal) != 1:
                break
            if game.logged:
                game.moves.append({'action': legal[0], 'kind': 'forced'})
            game.tree.advance(legal[0])
            game.ply += 1
        game.full = bool(self.rng.random() < cfg.full_search_prob)
        game.tree.set_noise(game.full)
        sims = cfg.sims if game.full else cfg.fast_sims
        game.target = sims + 1  # + the root's own evaluation

    def _move(self, game: _Game) -> None:
        """Record the finished search (if full), then pick and play a move."""
        tree, cfg = game.tree, self.config
        root = tree.root
        if game.full:
            target = np.zeros(ACTION_SIZE, np.float32)
            target[canonical_actions(root.pos)] = tree.policy(1.0)
            game.positions.append(root.pos)
            game.targets.append(target)
        if game.ply < cfg.temperature_plies:
            action = tree.sample_action(cfg.temperature)
        else:
            action = tree.best_action()
        if game.logged:
            game.moves.append(self._move_log(game, action))
        tree.advance(action)
        game.ply += 1

    def _move_log(self, game: _Game, action: int) -> dict[str, Any]:
        """Log entry for a searched move, with the root's statistics."""
        root = game.tree.root
        return {
            'action': action,
            'kind': 'full' if game.full else 'fast',
            'value': round(root.q(), 3),
            # [action, visits, prior] for every legal move
            'stats': [
                [a, int(n), round(float(p), 3)]
                for a, n, p in zip(
                    root.actions, root.N, root.P_raw, strict=True
                )
            ],
        }

    def _finish(self, game: _Game) -> GameRecord:
        """Label the recorded positions with the game's outcome."""
        final = game.tree.root.pos
        winner = 1 - final.turn if final.result() == -1 else None
        positions = game.positions
        n = len(positions)
        samples = Samples.empty(n)
        if n:
            samples.bits, samples.scalars = features(positions)
            samples.policy = np.stack(game.targets).astype(np.float16)
            samples.legal = np.packbits(legal_mask(positions), axis=1)
            samples.wdl = np.array(
                [
                    DRAW
                    if winner is None
                    else WIN
                    if p.turn == winner
                    else LOSS
                    for p in positions
                ],
                np.uint8,
            )
        return GameRecord(
            game.start_fen, game.moves, winner, game.ply, samples, game.logged
        )
