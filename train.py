#!/usr/bin/env python3
"""Train a checkers network by self-play.

Examples:
  # start a run from a config file
  python train.py --config configs/dev.json --run runs/dev
  # tweak settings from the command line
  python train.py --config configs/dev.json --run runs/dev2 \\
      --set train.lr=0.001 --set self_play.sims=400
  # continue a stopped run (Ctrl+C saves a checkpoint)
  python train.py --resume runs/dev
  # ... with changed settings (not the network's shape)
  python train.py --resume runs/dev --set train.lr=0.0005
  # continue past the planned end
  python train.py --resume runs/dev --iterations 200

Everything the run writes goes into the run directory; see
``training/run_store.py`` for the layout.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from training.config import RunConfig, apply_overrides, changed_config
from training.fileio import append_jsonl
from training.logs import event
from training.run_store import RunStore
from training.trainer import Trainer


def main() -> None:
    """Parse the arguments, then start a new run or resume one."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', type=Path, help='JSON config file')
    parser.add_argument('--run', type=Path, help='new run directory')
    parser.add_argument('--resume', type=Path, help='run directory to resume')
    parser.add_argument(
        '--set',
        action='append',
        default=[],
        metavar='SECTION.KEY=VALUE',
        help='override a config value (repeatable)',
    )
    parser.add_argument(
        '--iterations', type=int, help='when resuming: new total iterations'
    )
    parser.add_argument(
        '--progress',
        choices=['auto', 'on', 'off'],
        default='auto',
        help='progress bars (auto: only in an interactive terminal)',
    )
    args = parser.parse_args()

    if args.resume:
        if args.config or args.run:
            parser.error('--resume takes --set, --iterations and --progress')
        changes = []
        if args.set:
            store = RunStore.open(args.resume)
            try:
                config, changes = changed_config(store.load_config(), args.set)
            except ValueError as error:
                parser.error(str(error))
            config.save(store.config_path)
            if changes:
                append_jsonl(
                    store.root / 'config_changes.jsonl',
                    {'time': time.time(), 'changes': changes},
                )
        trainer = Trainer.resume(args.resume)
        if changes:
            event(trainer.log, 'RUN', 'config changed: ' + '; '.join(changes))
        if args.iterations:
            trainer.config.train.iterations = args.iterations
            trainer.config.save(trainer.store.config_path)
    else:
        if not args.run:
            parser.error('give --run for a new run, or --resume')
        data = json.loads(args.config.read_text()) if args.config else {}
        apply_overrides(data, args.set)
        trainer = Trainer.new(RunConfig.from_dict(data), args.run)
    trainer.run({'auto': None, 'on': True, 'off': False}[args.progress])


if __name__ == '__main__':
    main()
