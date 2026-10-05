"""What the server computes: game states, analyses and engine moves.

Lines, not positions
--------------------
The UI sends a *line*: a starting FEN (or none, for the standard start) plus
the actions played since. The server replays it to rebuild the exact
position, including the history the draw rules need (FEN can't hold
repetitions or the no-progress count, nor a multi-jump in progress). So the
server keeps no game state, and any line can be analysed at any time.

Everything is in hops
---------------------
An action is one hop (``core.game``), and the UI plays hop by hop: a
multi-jump is several clicks with the same piece. Analyses are per hop too,
because that's what the network and search see. For display, consecutive
hops by one side are grouped into turns with their PDN text (``22x15x6``).

Squares in the API are PDN numbers, 1-32.

Analysis
--------
An analysis is a search tree on the line's position, summarised for the UI:
the network's own view of the position (win/draw/loss and its priors) next
to what the search makes of it (visits, Q per move, and the line of best
play after each move). ``Analysis.step`` grows the tree a little at a time,
so live analysis can stream snapshots while it searches.

Models
------
``ModelPool`` finds every checkpoint under the runs directory and keeps the
most recently used networks loaded on the GPU. A model is named by its
checkpoint path relative to the runs directory, e.g.
``size_study/runs/6x64/checkpoints/iter_0324.pt``.
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from core.game import (
    A_FROM,
    A_MID,
    A_TO,
    COLOR_NAMES,
    DEFAULT_RULES,
    WHITE,
    Position,
    Rules,
    iter_squares,
)
from core.inference import NetEvaluator
from core.mcts import Node, SearchTree, run_search, search_wdl
from core.model import load_model
from core.players import make_player

MAX_SIMS = 100_000  # cap for one analysis (an "infinite" live search)
PV_LENGTH = 12  # hops shown in a principal variation
ENGINES = ['random', 'ab:1', 'ab:2', 'ab:4', 'ab:6', 'ab:8']


# -- positions -----------------------------------------------------------------


def replay(
    start: str | None, actions: list[int], rules: Rules = DEFAULT_RULES
) -> list[Position]:
    """Every position of a line: the start, then one per action.

    Raises ``ValueError`` for a bad FEN or an illegal action.
    """
    pos = Position.from_fen(start, rules) if start else Position.initial(rules)
    positions = [pos]
    for ply, action in enumerate(actions):
        if action not in pos.legal_actions():
            raise ValueError(f'illegal action {action} at ply {ply + 1}')
        pos = pos.play(action)
        positions.append(pos)
    return positions


def hop(action: int) -> dict[str, Any]:
    """An action as the UI draws it: squares 1-32, and what it captures."""
    mid = A_MID[action]
    return {
        'action': action,
        'from': A_FROM[action] + 1,
        'to': A_TO[action] + 1,
        'captures': mid + 1 if mid >= 0 else None,
    }


def turns(positions: list[Position], actions: list[int]) -> list[dict]:
    """Group a line's hops into turns, with their PDN text.

    A turn starts with a hop from a position that isn't mid multi-jump.
    ``ply`` is the index of the turn's first hop in ``actions``.
    """
    out: list[dict[str, Any]] = []
    for ply, action in enumerate(actions):
        pos = positions[ply]
        if pos.jump_sq < 0 or not out:
            out.append(
                {'color': COLOR_NAMES[pos.turn], 'ply': ply, 'actions': []}
            )
        out[-1]['actions'].append(action)
    for turn in out:
        turn['text'] = positions[turn['ply']].move_text(turn['actions'])
    return out


def outcome(pos: Position) -> dict[str, Any] | None:
    """How the game ended at ``pos``, or None if it goes on."""
    result = pos.result()
    if result is None:
        return None
    if result == -1:
        return {
            'winner': COLOR_NAMES[1 - pos.turn],
            'reason': f'{COLOR_NAMES[pos.turn]} has no moves',
        }
    if pos.reps >= pos.rules.repetition_limit:
        reason = 'repetition'
    else:
        reason = f'{pos.rules.draw_plies} plies without progress'
    return {'winner': None, 'reason': reason}


def state(start: str | None, actions: list[int]) -> dict[str, Any]:
    """Everything the UI needs to draw a line's current position."""
    positions = replay(start, actions)
    pos = positions[-1]
    pieces = [
        {'square': s + 1, 'color': color, 'king': bool(pos.kings >> s & 1)}
        for color, bits in (('black', pos.black), ('white', pos.white))
        for s in iter_squares(bits)
    ]
    over = outcome(pos)
    return {
        'fen': None if pos.jump_sq >= 0 else pos.to_fen(),
        'turn': COLOR_NAMES[pos.turn],
        'jump_square': pos.jump_sq + 1 if pos.jump_sq >= 0 else None,
        'pieces': pieces,
        'legal': [] if over else [hop(a) for a in pos.legal_actions()],
        'result': over,
        'turns': turns(positions, actions),
        'last': hop(actions[-1]) if actions else None,
        'no_progress': pos.no_progress,
        'repetitions': pos.reps,
    }


# -- analysis ------------------------------------------------------------------


def _line_after(node: Node | None) -> list[dict[str, Any]]:
    """The most-visited line of hops below ``node``."""
    line: list[dict[str, Any]] = []
    while node is not None and node.expanded and len(line) < PV_LENGTH:
        if not node.n_sum:
            break
        n, p = node.N, node.P_raw
        i = max(range(len(n)), key=lambda j: (n[j], p[j]))
        line.append(
            {**hop(node.actions[i]), 'color': COLOR_NAMES[node.pos.turn]}
        )
        node = node.children[i]
    return line


class Analysis:
    """A search on one position, grown in steps (see the module docstring)."""

    def __init__(self, pos: Position, evaluator: NetEvaluator) -> None:
        """Start a fresh tree on ``pos`` (no root noise)."""
        self.pos = pos
        self.evaluator = evaluator
        self.tree = SearchTree(pos)

    @property
    def sims(self) -> int:
        """Simulations done, not counting the root's own evaluation."""
        return max(self.tree.root.visits - 1, 0)

    def step(self, target: int, seconds: float = 0.0) -> None:
        """Search towards ``target`` simulations.

        With ``seconds``, stop early once that much time has passed, so a
        live analysis can report in between.
        """
        if self.pos.is_terminal:
            return
        start = time.perf_counter()
        while self.sims < target:
            goal = target if not seconds else min(target, self.sims + 32)
            run_search([self.tree], self.evaluator, goal, leaves_per_tree=8)
            if seconds and time.perf_counter() - start >= seconds:
                break

    def snapshot(self) -> dict[str, Any]:
        """The analysis so far, for the UI.

        Values are for the side to move: ``net`` is the network's own
        win/draw/loss estimate, ``q`` the search's average value (-1 to 1).
        Each move has its prior, visits, Q and the line expected after it.
        """
        pos, root = self.pos, self.tree.root
        if pos.is_terminal or not root.expanded:
            return {
                'turn': COLOR_NAMES[pos.turn],
                'sims': 0,
                'net': None,
                'q': None,
                'moves': [],
                'best': None,
            }
        moves = []
        for i, action in enumerate(root.actions):
            n = root.N[i]
            line = [{**hop(action), 'color': COLOR_NAMES[pos.turn]}]
            moves.append(
                {
                    **hop(action),
                    'prior': round(root.P_raw[i], 4),
                    'visits': n,
                    'q': round(root.W[i] / n, 4) if n else None,
                    'line': line + _line_after(root.children[i]),
                }
            )
        best = self.tree.best_action() if root.n_sum else None
        if best is None:  # no search yet: the network's favourite
            best = root.actions[
                max(range(len(root.P_raw)), key=root.P_raw.__getitem__)
            ]
        wdl = (
            [round(float(x), 4) for x in root.wdl]
            if root.wdl is not None
            else None
        )
        return {
            'turn': COLOR_NAMES[pos.turn],
            'sims': self.sims,
            'net': {'wdl': wdl, 'value': round(root.value, 4)},
            'q': round(root.q(), 4),
            'wdl': _rounded(search_wdl(root)),
            'moves': moves,
            'best': best,
        }


def line_values(
    start: str | None,
    actions: list[int],
    evaluator: NetEvaluator,
    sims: int = 0,
) -> list[dict[str, Any]]:
    """White's chances over a game: at the start and after every turn.

    The network's win/draw/loss for all those positions comes from one
    batch; with ``sims``, each is also searched (all trees together, so
    the batches stay large), and the shares and ``q`` are the search's
    (see ``search_wdl``). Everything is
    from White's side: ``white`` + ``draw`` + ``black`` = 1, ``q`` from -1
    (Black wins) to 1. ``turn`` is the index of the turn just completed (-1
    for the start); a line ending mid multi-jump gets a last point there.
    """
    positions = replay(start, actions)
    ends = [0] + [
        i for i in range(1, len(positions)) if positions[i].jump_sq < 0
    ]
    if positions[-1].jump_sq >= 0:
        ends.append(len(positions) - 1)
    live = [i for i in ends if not positions[i].is_terminal]
    net: dict[int, tuple[float, float, float]] = {}
    searched: dict[int, float] = {}
    searched_wdl: dict[int, tuple[float, float, float]] = {}
    if live:
        _, _, wdl = evaluator.evaluate([positions[i] for i in live])
        for k, i in enumerate(live):
            w, d, lo = (float(x) for x in wdl[k])
            net[i] = (w, d, lo)
        if sims:
            trees = [SearchTree(positions[i]) for i in live]
            run_search(trees, evaluator, min(sims, MAX_SIMS))
            for i, tree in zip(live, trees, strict=True):
                searched[i] = tree.root.q()
                searched_wdl[i] = search_wdl(tree.root) or net[i]
    points = []
    for turn, i in enumerate(ends):
        pos = positions[i]
        result = pos.result()
        if result is not None:  # the game is over: lost (-1) or drawn (0)
            mover = (0.0, 1.0, 0.0) if result == 0 else (0.0, 0.0, 1.0)
            q: float | None = float(result)
        else:
            # the search's view when there was one: the network alone
            # misjudges positions in the middle of an exchange
            mover, q = searched_wdl.get(i, net[i]), searched.get(i)
        win, draw, loss = mover
        white = pos.turn == WHITE
        points.append(
            {
                'turn': turn - 1,
                'ply': i,
                'white': round(win if white else loss, 4),
                'draw': round(draw, 4),
                'black': round(loss if white else win, 4),
                'q': None if q is None else round(q if white else -q, 4),
            }
        )
    return points


def _rounded(wdl: tuple[float, ...] | None) -> list[float] | None:
    """A win/draw/loss triple for JSON."""
    return None if wdl is None else [round(x, 4) for x in wdl]


# -- models --------------------------------------------------------------------


class ModelPool:
    """Checkpoints under a runs directory, with recently used ones loaded.

    ``lock`` serialises GPU work: hold it around any search.
    """

    def __init__(
        self, runs: Path, device: str | None = None, capacity: int = 4
    ) -> None:
        """Serve checkpoints found under ``runs`` on ``device``."""
        self.runs = runs.resolve()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.capacity = capacity
        self.loaded: OrderedDict[str, NetEvaluator] = OrderedDict()
        self.lock = threading.Lock()

    def path(self, model: str) -> Path:
        """The checkpoint file a model name refers to.

        Raises ``ValueError`` for anything that isn't a checkpoint inside
        the runs directory.
        """
        path = (self.runs / model).resolve()
        if not path.is_relative_to(self.runs) or path.suffix != '.pt':
            raise ValueError(
                f'not a checkpoint under the runs directory: {model}'
            )
        if not path.is_file():
            raise ValueError(f'no such checkpoint: {model}')
        return path

    def evaluator(self, model: str) -> NetEvaluator:
        """The loaded network for ``model``, loading it (and evicting the
        least recently used one) if needed."""
        if model in self.loaded:
            self.loaded.move_to_end(model)
            return self.loaded[model]
        net, _ = load_model(self.path(model), self.device)
        self.loaded[model] = NetEvaluator(net, backend='graphs', max_batch=256)
        while len(self.loaded) > self.capacity:
            self.loaded.popitem(last=False)
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        return self.loaded[model]

    def strongest(self) -> str | None:
        """The best-rated searching network in any tournament under the
        runs directory (see ``training.tournament``), if one exists."""
        best: tuple[float, str] | None = None
        for ratings in self.runs.rglob('tournament/ratings.json'):
            plan = ratings.parent / 'plan.json'
            if not plan.exists():
                continue
            elo = json.loads(ratings.read_text(encoding='utf-8'))['ratings']
            entrants = json.loads(plan.read_text(encoding='utf-8'))['entrants']
            for e in entrants:
                if not e.get('checkpoint') or e.get('sims', 0) <= 1:
                    continue  # alpha-beta, or a network without search
                path = Path(e['checkpoint']).resolve()
                if e['name'] not in elo or not path.is_relative_to(self.runs):
                    continue
                if not path.is_file():
                    continue
                rating = elo[e['name']]['elo']
                if best is None or rating > best[0]:
                    best = (rating, path.relative_to(self.runs).as_posix())
        return best[1] if best else None

    def catalog(self) -> list[dict[str, Any]]:
        """Every run with checkpoints, newest first, and their checkpoints.

        Each checkpoint: model name, iteration and (when the run recorded
        it) hours of training.
        """
        runs = []
        for config in self.runs.rglob('config.json'):
            root = config.parent
            files = sorted((root / 'checkpoints').glob('iter_*.pt'))
            if not files:
                continue
            hours = _training_hours(root / 'metrics.jsonl')
            shape = json.loads(config.read_text(encoding='utf-8'))['model']
            runs.append(
                {
                    'run': root.relative_to(self.runs).as_posix(),
                    'shape': f'{shape["blocks"]}x{shape["channels"]}',
                    'updated': max(f.stat().st_mtime for f in files),
                    'checkpoints': [
                        {
                            'model': f.relative_to(self.runs).as_posix(),
                            'iteration': int(f.stem.split('_')[1]),
                            'hours': hours.get(int(f.stem.split('_')[1])),
                        }
                        for f in files
                    ],
                }
            )
        runs.sort(key=lambda r: -r['updated'])
        return runs


def _training_hours(metrics: Path) -> dict[int, float]:
    """Hours of training at each iteration, from a run's metrics."""
    hours: dict[int, float] = {}
    if not metrics.exists():
        return hours
    for line in metrics.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if 'elapsed_hours' in row:
            hours[row['iteration']] = round(row['elapsed_hours'], 2)
    return hours


# -- engine moves --------------------------------------------------------------


def engine_move(
    pos: Position, player: str, pool: ModelPool
) -> tuple[int, dict[str, Any] | None]:
    """The hop ``player`` plays from ``pos``, plus its analysis if it
    searched.

    ``player`` is ``model:<name>:<sims>`` (a network searching), ``ab:<depth>``
    (alpha-beta) or ``random``.
    """
    if pos.is_terminal:
        raise ValueError('the game is over')
    legal = pos.legal_actions()
    if player.startswith('model:'):
        name, _, sims = player[len('model:') :].rpartition(':')
        if not name or not sims.isdigit():
            raise ValueError(f'expected model:<name>:<sims>, got {player}')
        analysis = Analysis(pos, pool.evaluator(name))
        analysis.step(min(int(sims), MAX_SIMS))
        return analysis.tree.best_action(), analysis.snapshot()
    if len(legal) == 1:
        return legal[0], None
    if player == 'random':
        return random.choice(legal), None
    if player.startswith('ab:') and player[3:].isdigit():
        if not 1 <= int(player[3:]) <= 10:
            raise ValueError('alpha-beta depth must be 1-10')
        return make_player(player).choose(pos), None
    raise ValueError(f'unknown player {player!r}')
