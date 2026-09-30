"""Probe positions: watch what the network thinks about chosen positions.

A probe is a named position (FEN), optionally with a note and the move you
expect. During training, every ``probes.every`` iterations, each probe is
evaluated two ways and the results are appended to ``probes.jsonl``:

- **Raw network**: the prior for every legal move and the win/draw/loss
  estimate. This is the network's instinct.
- **Search**: ``probes.sims`` simulations of MCTS. Visits and Q per move, the
  searched value, the best move and the expected line (principal variation).

Plotted over iterations, this shows what the network is learning: a winning
move's prior rising, a drawn endgame's value settling at zero, and so on.

Adding probes
-------------
Probes live in ``runs/<name>/probes.json`` (a JSON list), which starts as a
copy of ``configs/probes.json``. The trainer re-reads it every time, so a
probe added mid-run is picked up at the next iteration:

    python -m training.probes add runs/dev --name my-shot --fen "B:W...:B..."

Its history before that can be filled in by evaluating it with every saved
checkpoint (``backfill``). Back-filled rows are marked ``"backfill": true``;
readers merge rows by iteration.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from core.game import Position, Rules
from core.inference import NetEvaluator
from core.mcts import SearchTree, run_search
from core.model import load_model
from training import fileio
from training.run_store import RunStore


@dataclass
class Probe:
    name: str
    fen: str
    note: str = ''
    # The move(s) counted as correct, e.g. '21-25' or ['21-25', '22-26']
    # when several moves are equally good (say, two different wins).
    expected: str | list[str] | None = None

    def accepted(self) -> list[str]:
        """The moves counted as correct (empty if none is expected)."""
        if self.expected is None:
            return []
        if isinstance(self.expected, str):
            return [self.expected]
        return list(self.expected)


def load_probes(path: str | Path, rules: Rules) -> list[Probe]:
    """Read a probes file. Raises ``ValueError`` naming any bad entry."""
    probes = []
    for entry in json.loads(Path(path).read_text(encoding='utf-8')):
        probe = Probe(**entry)
        try:
            Position.from_fen(probe.fen, rules)
        except ValueError as e:
            raise ValueError(f'Probe {probe.name!r}: {e}') from e
        probes.append(probe)
    names = [p.name for p in probes]
    if len(set(names)) != len(names):
        raise ValueError('Probe names must be unique')
    return probes


def save_probes(path: str | Path, probes: Sequence[Probe]) -> None:
    """Write a probes file atomically, so the trainer, which re-reads it
    each iteration, never sees a half-written file."""
    text = json.dumps([asdict(p) for p in probes], indent=2) + '\n'
    fileio.write_text(Path(path), text)


def evaluate_probes(
    probes: Sequence[Probe], evaluator: NetEvaluator, sims: int, rules: Rules
) -> dict[str, dict[str, Any]]:
    """Evaluate probes with the raw network and with search.

    Returns ``{name: result}``; see the module docstring for the contents.
    """
    positions = [Position.from_fen(p.fen, rules) for p in probes]
    live = [i for i, pos in enumerate(positions) if not pos.is_terminal]
    results: dict[str, dict[str, Any]] = {
        p.name: {'terminal': pos.result()}
        for p, pos in zip(probes, positions, strict=True)
        if pos.is_terminal
    }
    if not live:
        return results

    priors, values, wdl = evaluator.evaluate([positions[i] for i in live])
    trees = [SearchTree(positions[i]) for i in live]
    run_search(trees, evaluator, sims, leaves_per_tree=8)

    for k, i in enumerate(live):
        probe, pos, tree = probes[i], positions[i], trees[k]
        root = tree.root
        moves = [
            {
                'move': pos.move_text([a]),
                'action': a,
                'prior': round(float(priors[k][j]), 4),
                'visits': int(root.N[j]),
                'q': round(float(root.W[j] / root.N[j]), 4)
                if root.N[j]
                else None,
            }
            for j, a in enumerate(root.actions)
        ]
        best = pos.move_text([tree.best_action()])
        pv, p = [], pos
        for a in tree.principal_variation(12):
            pv.append(p.move_text([a]))
            p = p.play(a)
        results[probe.name] = {
            'wdl': [round(float(x), 4) for x in wdl[k]],
            'value': round(float(values[k]), 4),
            'search_value': round(root.q(), 4),
            'best': best,
            'correct': best in probe.accepted() if probe.accepted() else None,
            'pv': pv,
            'moves': moves,
        }
    return results


def backfill(
    store: RunStore, names: Sequence[str] | None = None, device: str = 'cpu'
) -> int:
    """Evaluate probes (all, or those in ``names``) with every saved
    checkpoint; append the rows to ``probes.jsonl``. Returns rows written."""
    config = store.load_config()
    probes = load_probes(store.probes_path, config.rules)
    if names is not None:
        probes = [p for p in probes if p.name in names]
    rows = 0
    for path in store.list_checkpoints():
        model, ckpt = load_model(path, device)
        results = evaluate_probes(
            probes, NetEvaluator(model), config.probes.sims, config.rules
        )
        store.append_probe_results(
            {
                'iteration': ckpt['iteration'],
                'backfill': True,
                'probes': results,
            }
        )
        rows += 1
    return rows


def merged_results(store: RunStore) -> dict[int, dict[str, Any]]:
    """``{iteration: {probe name: result}}`` with back-filled rows merged."""
    merged: dict[int, dict[str, Any]] = {}
    for row in store.read_probe_results():
        merged.setdefault(row['iteration'], {}).update(row['probes'])
    return dict(sorted(merged.items()))


def main() -> None:
    """Command line: add, list or back-fill a run's probes."""
    parser = argparse.ArgumentParser(description="Manage a run's probes.")
    sub = parser.add_subparsers(dest='command', required=True)
    add = sub.add_parser('add', help='add a probe to a run')
    add.add_argument('run', type=Path)
    add.add_argument('--name', required=True)
    add.add_argument('--fen', required=True)
    add.add_argument('--note', default='')
    add.add_argument(
        '--expected',
        default=None,
        help='correct move(s), comma-separated: 21-25 or 21-25,22-26',
    )
    add.add_argument('--backfill', action='store_true', help='then backfill it')
    lst = sub.add_parser('list', help='show probes and their latest results')
    lst.add_argument('run', type=Path)
    fill = sub.add_parser('backfill', help='evaluate probes on old checkpoints')
    fill.add_argument('run', type=Path)
    fill.add_argument('--name', action='append', help='only these (repeatable)')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    store = RunStore.open(args.run)
    rules = store.load_config().rules
    if args.command == 'add':
        probes = load_probes(store.probes_path, rules)
        if any(p.name == args.name for p in probes):
            parser.error(f'probe {args.name!r} already exists')
        try:
            Position.from_fen(args.fen, rules)  # validate before saving
        except ValueError as e:
            parser.error(f'bad FEN: {e}')
        expected = args.expected.split(',') if args.expected else None
        probes.append(Probe(args.name, args.fen, args.note, expected))
        save_probes(store.probes_path, probes)
        print(f'Added {args.name!r}; training picks it up next iteration.')
        if args.backfill:
            rows = backfill(store, [args.name], device)
            print(f'Back-filled {rows} checkpoints.')
    elif args.command == 'backfill':
        print(f'Back-filled {backfill(store, args.name, device)} checkpoints.')
    else:
        latest = list(merged_results(store).items())[-1:]
        for probe in load_probes(store.probes_path, rules):
            line = f'{probe.name:20s} {probe.fen}'
            if latest and probe.name in latest[0][1]:
                r = latest[0][1][probe.name]
                line += (
                    f'\n    it {latest[0][0]}: best {r.get("best")} '
                    f'value {r.get("search_value")} wdl {r.get("wdl")}'
                )
            print(line)


if __name__ == '__main__':
    main()
