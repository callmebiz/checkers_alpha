"""Players: anything that picks a move.

Used by the CLI, the evaluation arena and the UI server.

Every player implements ``choose(pos) -> action``, returning a single hop. A
multi-jump takes one call per hop; the position then has ``jump_sq`` set and the
same side to move.

- ``RandomPlayer``: uniform over legal moves. The weakest baseline.
- ``AlphaBetaPlayer``: classical search with a material heuristic. It doesn't
  change between runs, so it's a fixed yardstick for measuring trained networks.
- ``PolicyPlayer``: plays the network's first instinct (highest prior), no
  search.
- ``MCTSPlayer``: tree search guided by an evaluator (network, uniform or
  rollouts).

``make_player(spec)`` builds any of these from a short string, for command lines
and configs.
"""

from __future__ import annotations

import random

import numpy as np

from core.game import A_MID, BLACK, WHITE, Position
from core.inference import (
    Evaluator,
    NetEvaluator,
    RolloutEvaluator,
    UniformEvaluator,
)
from core.mcts import SearchConfig, SearchTree, run_search
from core.model import load_model


class Player:
    name = 'player'

    def choose(self, pos: Position) -> int:
        """Return a legal action (one hop) for the side to move."""
        raise NotImplementedError

    def reset(self) -> None:
        """Forget per-game state before a new game."""


class RandomPlayer(Player):
    name = 'random'

    def __init__(self, seed: int | None = None) -> None:
        """A seed makes the choices reproducible."""
        self.rng = random.Random(seed)

    def choose(self, pos: Position) -> int:
        """A uniformly random legal hop."""
        return self.rng.choice(pos.legal_actions())


# ------------------------------------------------------------------------------
# Alpha-beta

WIN_SCORE = 100_000
MAN_VALUE, KING_VALUE, ADVANCE_BONUS = 100, 160, 3
_ROW_MASKS = [sum(1 << (r * 4 + i) for i in range(4)) for r in range(8)]
# (row mask, rows advanced) per color: Black men advance down the board, White
# men up
_ADVANCEMENT = {
    BLACK: [(mask, r) for r, mask in enumerate(_ROW_MASKS)],
    WHITE: [(mask, 7 - r) for r, mask in enumerate(_ROW_MASKS)],
}


def material_eval(pos: Position) -> int:
    """Heuristic score for the side to move.

    Material, plus a little for advanced men.
    """
    score = 0
    for color, sign in ((pos.turn, 1), (1 - pos.turn, -1)):
        bits = pos.white if color == WHITE else pos.black
        men, kings = bits & ~pos.kings, bits & pos.kings
        s = MAN_VALUE * men.bit_count() + KING_VALUE * kings.bit_count()
        advanced = sum(
            rows * (men & mask).bit_count()
            for mask, rows in _ADVANCEMENT[color]
        )
        s += ADVANCE_BONUS * advanced
        score += sign * s
    return score


class AlphaBetaPlayer(Player):
    """Negamax alpha-beta over full turns to a fixed depth.

    Search continues past the depth limit while a capture is pending, up to
    ``max_extension`` extra turns. Captures are compulsory, so stopping
    mid-exchange would badly misjudge the position. Among equally scored moves
    it picks randomly, so games vary.
    """

    def __init__(
        self, depth: int = 6, max_extension: int = 8, seed: int | None = None
    ) -> None:
        """Search ``depth`` full turns, plus up to ``max_extension`` for
        captures.
        """
        self.depth = depth
        self.max_extension = max_extension
        self.rng = random.Random(seed)
        self.name = f'ab{depth}'
        self.nodes = 0  # positions searched for the last move

    def choose(self, pos: Position) -> int:
        """First hop of the best full move; ties are broken at random."""
        moves = pos.complete_moves()
        if len(moves) == 1:
            return moves[0][0][0]
        self.nodes = 0
        best, best_moves = -WIN_SCORE * 10, []
        for actions, nxt in moves:
            # Window lower bound best - 1 so equally good moves are still scored
            # exactly
            v = -self._search(
                nxt, self.depth - 1, -WIN_SCORE * 10, -best + 1, 0
            )
            if v > best:
                best, best_moves = v, [actions]
            elif v == best:
                best_moves.append(actions)
        return self.rng.choice(best_moves)[0]

    def _search(
        self, pos: Position, depth: int, alpha: int, beta: int, extension: int
    ) -> int:
        """Score of ``pos`` for its side to move, ``depth`` more turns deep."""
        self.nodes += 1
        result = pos.result()
        if result is not None:
            # scaled by depth left: prefer quicker wins and slower losses
            return result * (WIN_SCORE + depth)
        if depth <= 0:
            capture_pending = A_MID[pos.legal_actions()[0]] >= 0
            if not capture_pending or extension >= self.max_extension:
                return material_eval(pos)
            extension += 1
        for _, nxt in pos.complete_moves():
            v = -self._search(nxt, depth - 1, -beta, -alpha, extension)
            if v >= beta:
                # the opponent won't allow this line; stop searching it
                return v
            alpha = max(alpha, v)
        return alpha


# ------------------------------------------------------------------------------
# Network players


class PolicyPlayer(Player):
    """Plays the evaluator's priors directly, with no search."""

    def __init__(
        self,
        evaluator: Evaluator,
        temperature: float = 0.0,
        seed: int | None = None,
        name: str = 'policy',
    ) -> None:
        """Play ``evaluator``'s priors; temperature 0 picks the top move."""
        self.evaluator = evaluator
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.name = name
        # (priors, value, wdl) of the last decision
        self.last: tuple[np.ndarray, float, np.ndarray] | None = None

    def choose(self, pos: Position) -> int:
        """The top-prior move, or one sampled from the priors."""
        priors, values, wdl = self.evaluator.evaluate([pos])
        self.last = (priors[0], float(values[0]), wdl[0])
        p = priors[0]
        if self.temperature == 0:
            return pos.legal_actions()[int(np.argmax(p))]
        p = p ** (1 / self.temperature)
        return pos.legal_actions()[int(self.rng.choice(len(p), p=p / p.sum()))]


class MCTSPlayer(Player):
    """Tree search guided by an evaluator.

    The tree is kept between moves and re-rooted on the new position, so earlier
    search isn't thrown away. ``tree`` stays readable for display.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        sims: int = 400,
        config: SearchConfig | None = None,
        temperature: float = 0.0,
        batch: int = 16,
        seed: int | None = None,
        name: str = 'mcts',
    ) -> None:
        """Search ``sims`` simulations per move, ``batch`` leaves per call."""
        self.evaluator = evaluator
        self.sims = sims
        self.config = config or SearchConfig()
        self.temperature = temperature
        self.batch = batch  # leaves evaluated per network call
        self.rng = np.random.default_rng(seed)
        self.name = name
        self.tree: SearchTree | None = None

    def reset(self) -> None:
        """Drop the search tree before a new game."""
        self.tree = None

    def choose(self, pos: Position) -> int:
        """Search from ``pos`` (reusing the tree if possible); pick a move."""
        if self.tree is None:
            self.tree = SearchTree(pos, self.config, rng=self.rng)
        else:
            self.tree.reroot(pos)
        if len(pos.legal_actions()) == 1:
            return pos.legal_actions()[0]  # forced: nothing to think about
        run_search([self.tree], self.evaluator, self.sims, self.batch)
        if self.temperature == 0:
            return self.tree.best_action()
        return self.tree.sample_action(self.temperature)


# ------------------------------------------------------------------------------


def make_player(
    spec: str, device: str = 'cpu', seed: int | None = None
) -> Player:
    """Build a player from a spec string:

    - ``random``
    - ``ab[:depth]``: alpha-beta, default depth 6
    - ``policy:<checkpoint>``
    - ``mcts:<checkpoint|uniform|rollout>[:sims]``: default 400 simulations
    """
    kind, _, rest = spec.partition(':')
    kind = kind.lower()
    if kind == 'random':
        return RandomPlayer(seed)
    if kind == 'ab':
        return AlphaBetaPlayer(int(rest or 6), seed=seed)
    if kind == 'policy':
        return PolicyPlayer(
            _net_evaluator(rest, device), seed=seed, name=f'policy:{rest}'
        )
    if kind == 'mcts':
        source, sims = rest, 400
        head, _, tail = rest.rpartition(':')
        # a checkpoint path may itself contain ':' (C:\...)
        if head and tail.isdigit():
            source, sims = head, int(tail)
        evaluator: Evaluator
        if source == 'uniform':
            evaluator = UniformEvaluator()
        elif source == 'rollout':
            evaluator = RolloutEvaluator(rng=random.Random(seed))
        else:
            evaluator = _net_evaluator(source, device)
        return MCTSPlayer(
            evaluator, sims, seed=seed, name=f'mcts:{source}:{sims}'
        )
    raise ValueError(f'Unknown player spec {spec!r}')


def _net_evaluator(path: str, device: str) -> NetEvaluator:
    """Load a checkpoint and wrap its network in an evaluator."""
    model, _ = load_model(path, device)
    return NetEvaluator(model)
