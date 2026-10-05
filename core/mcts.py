"""Monte Carlo tree search (AlphaZero PUCT) with batched evaluation.

The tree
--------
Each ``Node`` holds a position. Once expanded, it also holds per-move statistics
for its legal moves, stored in the parent rather than in the children, so
choosing a move reads one array:

    P[i]  prior: how promising the network thinks move i is
    N[i]  visit count: simulations that went through move i
    W[i]  total value of those simulations, for the side to move here

``W[i] / N[i]`` is the average outcome ``Q`` of move i for the player choosing
at this node.

One simulation
--------------
1. **Select** (``select_leaf``): from the root, repeatedly take the move
   maximising ``Q + U``::

       U = c_puct * P[i] * sqrt(sum(N) + 1) / (1 + N[i])

   ``U`` favours moves the network likes that haven't been tried much. As
   ``N[i]`` grows, ``U`` shrinks and the measured ``Q`` takes over. Stop at
   a node that isn't expanded yet (a leaf).
2. **Evaluate**: the network scores the leaf with move priors and a value. This
   happens *outside* the tree, so ``run_search`` can batch leaves from many
   trees into one network call.
3. **Expand** (``expand_and_backup``): store the priors on the leaf and make
   room for its children, which are created lazily when first selected.
4. **Backup** (same call): add the value to ``W`` along the path back to the
   root. The value is for the side to move at the leaf, so it's negated whenever
   the side to move changes. In a multi-jump the same side moves again, so the
   sign is kept.

Reaching a finished game skips steps 2-3 and backs up the actual result (-1 loss
or 0 draw for the side to move there).

After many simulations, the most-visited root move is the choice, and the visit
distribution is the training target for the policy.

Details
-------
- **First-play urgency (FPU)**: an unvisited move's ``Q`` is taken as the
  parent's value minus ``fpu_reduction``, so priors decide which untried move to
  explore first. The old code gave unvisited moves infinite score, forcing every
  move to be tried once and ignoring the priors.
- **Virtual loss**: to take several leaves from one tree before evaluating them,
  each path is temporarily counted as a visited loss. The next selection then
  prefers a different path. The real value replaces it on backup.
- **Root noise**: in self-play, Dirichlet noise over the legal moves is mixed
  into the root priors so training explores moves the network currently
  underrates.
- **Tree reuse**: after a move, the chosen child's subtree becomes the new tree,
  keeping the simulations already spent below it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from core.game import Position
from core.inference import Evaluator


@dataclass
class SearchConfig:
    c_puct: float = 1.75  # exploration strength in U
    fpu_reduction: float = 0.25  # unvisited move Q = parent value - this
    # Root noise: alpha = dirichlet_total / number of legal moves, and
    # dirichlet_epsilon is the share of the prior replaced by noise
    dirichlet_total: float = 10.0
    dirichlet_epsilon: float = 0.25
    virtual_loss: float = 1.0


class Node:
    """A position in the tree plus statistics for its moves once expanded.

    The per-move statistics are plain lists: a node has only a handful of
    moves, and for that few a Python loop is faster than numpy's per-call
    overhead (selection runs millions of times per iteration).
    """

    __slots__ = (
        'pos', 'terminal', 'expanded', 'actions', 'P', 'P_raw', 'N', 'W',
        'children', 'visits', 'n_sum', 'value', 'wdl', 'pending',
    )  # fmt: skip

    def __init__(self, pos: Position) -> None:
        """Create an unexpanded node for ``pos``."""
        self.pos = pos
        # game outcome for the side to move, if the game is over here
        self.terminal: int | None = pos.result()
        self.expanded = False
        # Filled in on expansion; index i refers to move actions[i] throughout
        self.actions: list[int] = []
        # priors used for selection (may include root noise)
        self.P: list[float] = []
        # priors from the evaluator
        self.P_raw: list[float] = []
        # visits per move, including in-flight virtual visits
        self.N: list[int] = []
        # value sum per move, for the side to move here
        self.W: list[float] = []
        # children are created when first selected
        self.children: list[Node | None] = []
        self.visits = 0  # completed simulations through this node
        self.n_sum = 0  # sum(N), kept as a running total
        self.value = 0.0  # evaluator's value of this position
        self.wdl: Any = None  # evaluator's win/draw/loss for this position
        self.pending = False  # selected and waiting for evaluation

    def expand(
        self, priors: npt.ArrayLike, value: float, wdl: Any = None
    ) -> None:
        """Store the evaluation and make room for this node's moves."""
        self.actions = self.pos.legal_actions()
        k = len(self.actions)
        self.P = self.P_raw = np.asarray(priors, dtype=np.float64).tolist()
        self.N = [0] * k
        self.W = [0.0] * k
        self.children = [None] * k
        self.value = float(value)
        self.wdl = wdl
        self.expanded = True
        self.pending = False

    def q(self) -> float:
        """Average value for the side to move here, incl. its own evaluation."""
        if self.terminal is not None:
            return float(self.terminal)
        if not self.expanded:
            return self.value
        return (self.value + sum(self.W)) / (1 + sum(self.N))


@dataclass(slots=True)
class Leaf:
    """A node awaiting evaluation, and the path that led to it.

    ``path`` lists ``(node, move index)`` pairs from the root.
    """

    tree: SearchTree
    node: Node
    path: list[tuple[Node, int]] = field(default_factory=list)


class Selection(Enum):
    """Outcomes of ``select_leaf`` that don't produce a leaf to evaluate."""

    # reached a finished game; its result is already backed up
    TERMINAL = 'terminal'
    # reached a leaf already waiting for evaluation; nothing changed
    COLLISION = 'collision'


class SearchTree:
    """A search tree rooted at one position. Drive it with ``run_search``."""

    def __init__(
        self,
        pos: Position,
        config: SearchConfig | None = None,
        add_noise: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Start a tree at ``pos``; ``add_noise`` perturbs the root priors."""
        self.config = config or SearchConfig()
        self.add_noise = add_noise
        self.rng = rng or np.random.default_rng()
        self.root = Node(pos)

    # -- one simulation --------------------------------------------------------

    def select_leaf(self) -> Leaf | Selection:
        """Step 1: walk from the root to a leaf, adding virtual loss."""
        vl = self.config.virtual_loss
        node = self.root
        path: list[tuple[Node, int]] = []
        while True:
            if node.terminal is not None:
                self._backup(path, node, float(node.terminal))
                return Selection.TERMINAL
            if not node.expanded:
                # another in-flight simulation may already have chosen it
                if node.pending:
                    self._undo_virtual_loss(path)
                    return Selection.COLLISION
                node.pending = True
                return Leaf(self, node, path)
            i = self._select_child(node)
            # a virtual visit, counted as a loss until the real value arrives
            node.N[i] += 1
            node.W[i] -= vl
            node.n_sum += 1
            path.append((node, i))
            child = node.children[i]
            if child is None:
                child = Node(node.pos.play(node.actions[i]))
                node.children[i] = child
            node = child

    def _select_child(self, node: Node) -> int:
        """Index of the move maximising ``Q + U`` (see the module docstring)."""
        cfg = self.config
        c = cfg.c_puct * math.sqrt(node.n_sum + 1)
        fpu = node.value - cfg.fpu_reduction  # Q of unvisited moves
        best_i, best = 0, -math.inf
        for i, (n, w, p) in enumerate(zip(node.N, node.W, node.P, strict=True)):
            score = (w / n if n else fpu) + c * p / (1 + n)
            if score > best:
                best_i, best = i, score
        return best_i

    def expand_and_backup(
        self, leaf: Leaf, priors: npt.ArrayLike, value: float, wdl: Any = None
    ) -> None:
        """Steps 3-4: expand ``leaf``, then back its value up the path."""
        node = leaf.node
        node.expand(priors, value, wdl)
        if node is self.root and self.add_noise:
            self._apply_noise()
        self._backup(leaf.path, node, float(value))

    def _backup(
        self, path: list[tuple[Node, int]], leaf_node: Node, value: float
    ) -> None:
        """Add ``value`` to every move on ``path``.

        ``value`` is for the side to move at ``leaf_node``.
        """
        leaf_node.visits += 1
        turn = leaf_node.pos.turn
        vl = self.config.virtual_loss
        for node, i in reversed(path):
            # when the mover changes, one side's gain is the other's loss
            if node.pos.turn != turn:
                value, turn = -value, node.pos.turn
            # + vl cancels the virtual loss added on the way down
            node.W[i] += value + vl
            node.visits += 1

    def _undo_virtual_loss(self, path: list[tuple[Node, int]]) -> None:
        """Remove the virtual loss added along ``path`` (after a collision)."""
        vl = self.config.virtual_loss
        for node, i in path:
            node.N[i] -= 1
            node.W[i] += vl
            node.n_sum -= 1

    def _apply_noise(self) -> None:
        """Mix Dirichlet noise into the root priors.

        Starts from the unmodified ``P_raw``, so noise never accumulates.
        """
        root = self.root
        k = len(root.actions)
        if k < 2:
            return
        noise = self.rng.dirichlet([self.config.dirichlet_total / k] * k)
        eps = self.config.dirichlet_epsilon
        root.P = [
            (1 - eps) * p + eps * float(x)
            for p, x in zip(root.P_raw, noise, strict=True)
        ]

    # -- results ---------------------------------------------------------------

    @property
    def sims(self) -> int:
        """Completed simulations, counting the root's own evaluation."""
        return self.root.visits

    def policy(self, temperature: float = 1.0) -> npt.NDArray[np.float64]:
        """Visit distribution over ``root.actions``.

        Sharpened by ``1 / temperature``: temperature 1 is proportional to
        visits, 0 is one-hot on the best move.
        """
        N = np.array(self.root.N, dtype=np.float64)
        if temperature == 0 or N.sum() == 0:
            pi = np.zeros_like(N)
            pi[self.best_index()] = 1.0
            return pi
        N = N ** (1.0 / temperature)
        return N / N.sum()

    def best_index(self) -> int:
        """Index of the most-visited root move (ties broken by prior)."""
        root = self.root
        return max(range(len(root.N)), key=lambda i: (root.N[i], root.P_raw[i]))

    def best_action(self) -> int:
        """The most-visited root move."""
        return self.root.actions[self.best_index()]

    def sample_action(self, temperature: float = 1.0) -> int:
        """A root move drawn from the visit counts at ``temperature``."""
        pi = self.policy(temperature)
        return self.root.actions[int(self.rng.choice(len(pi), p=pi))]

    def principal_variation(self, max_len: int = 20) -> list[int]:
        """The expected line: follow the most-visited moves down."""
        node: Node | None = self.root
        pv: list[int] = []
        while node is not None and node.expanded and len(pv) < max_len:
            if node.n_sum == 0:
                break
            i = max(range(len(node.N)), key=node.N.__getitem__)
            pv.append(node.actions[i])
            node = node.children[i]
        return pv

    def move_stats(self) -> list[dict[str, Any]]:
        """Per root move: action, prior, visits and Q.

        Q is for the side to move at the root.
        """
        root = self.root
        return [
            {
                'action': a,
                'prior': root.P_raw[i],
                'visits': root.N[i],
                'q': root.W[i] / root.N[i] if root.N[i] else None,
            }
            for i, a in enumerate(root.actions)
        ]

    # -- tree reuse ------------------------------------------------------------

    def advance(self, action: int) -> None:
        """Make the child reached by ``action`` the new root.

        Its subtree and visits are kept; the rest of the tree is freed.
        """
        root = self.root
        child = None
        if root.expanded:
            child = root.children[root.actions.index(action)]
        self._set_root(child or Node(root.pos.play(action)))

    def reroot(self, pos: Position, max_depth: int = 4) -> bool:
        """Re-root on a descendant equal to ``pos``, if one is close.

        Looks ``max_depth`` plies down, otherwise starts a fresh tree. Returns
        whether a subtree was reused.
        """
        target = (pos.key, pos.no_progress)
        frontier = [self.root]
        for _ in range(max_depth + 1):
            for node in frontier:
                if (node.pos.key, node.pos.no_progress) == target:
                    self._set_root(node)
                    return True
            frontier = [
                c for n in frontier for c in n.children if c is not None
            ]
        self._set_root(Node(pos))
        return False

    def _set_root(self, node: Node) -> None:
        """Make ``node`` the root, re-applying root noise if enabled."""
        self.root = node
        if node.expanded and self.add_noise:
            self._apply_noise()

    def set_noise(self, enabled: bool) -> None:
        """Turn root noise on or off, updating an already expanded root."""
        self.add_noise = enabled
        if self.root.expanded:
            if enabled:
                self._apply_noise()
            else:
                self.root.P = self.root.P_raw


def search_wdl(node: Node) -> tuple[float, float, float] | None:
    """The search's win/draw/loss for the side to move at ``node``.

    The search itself keeps one value per move (win minus loss), but each
    position it reached has the network's win/draw/loss; averaging those
    over the tree, weighted by visits, gives the search's view with draws
    kept apart. Its win minus loss equals the node's Q. Mid-exchange
    positions show why it matters: right after a capture with a forced
    recapture coming, the network alone sees a piece up; the search has
    already played the recapture. None for an unexamined position.
    """
    if node.terminal is not None:  # lost (-1) or drawn (0)
        return (0.0, 1.0, 0.0) if node.terminal == 0 else (0.0, 0.0, 1.0)
    if not node.expanded or node.wdl is None:
        return None
    total = [float(x) for x in node.wdl]
    weight = 1
    for i, child in enumerate(node.children):
        if child is None or not node.N[i]:
            continue
        sub = search_wdl(child)
        if sub is None:
            continue
        if child.pos.turn != node.pos.turn:  # the other side's view
            sub = (sub[2], sub[1], sub[0])
        for k in range(3):
            total[k] += node.N[i] * sub[k]
        weight += node.N[i]
    w, d, lo = (x / weight for x in total)
    return w, d, lo


def run_search(
    trees: Sequence[SearchTree],
    evaluator: Evaluator,
    num_sims: int | Sequence[int],
    leaves_per_tree: int = 1,
) -> None:
    """Search every tree to ``num_sims`` simulations.

    ``num_sims`` is one budget for all trees or one per tree. Simulations are
    counted beyond each root's own evaluation, and visits kept from tree reuse
    count toward the total.

    Each round collects up to ``leaves_per_tree`` leaves from every unfinished
    tree and evaluates them all in one batch. Use many trees with one leaf
    each for self-play, or one tree with several leaves for analysing a single
    position.
    """
    if isinstance(num_sims, int):
        num_sims = [num_sims] * len(trees)
    # (tree, root visits at which it is done); +1 for the root's evaluation
    active = [
        (tree, sims + 1)
        for tree, sims in zip(trees, num_sims, strict=True)
        if tree.root.terminal is None
    ]
    while active:
        leaves: list[Leaf] = []
        for tree, target in active:
            for _ in range(min(leaves_per_tree, target - tree.root.visits)):
                selected = tree.select_leaf()
                if selected is Selection.COLLISION:
                    break  # other paths lead to pending leaves; next round
                if isinstance(selected, Leaf):
                    leaves.append(selected)
        if leaves:
            priors, values, wdl = evaluator.evaluate(
                [leaf.node.pos for leaf in leaves]
            )
            for i, leaf in enumerate(leaves):
                leaf.tree.expand_and_backup(leaf, priors[i], values[i], wdl[i])
        active = [(t, n) for t, n in active if t.root.visits < n]
