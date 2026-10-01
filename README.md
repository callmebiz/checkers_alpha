# Checkers Alpha

An AlphaZero-style engine for American checkers (English draughts). A neural
network learns by playing against itself, guided by Monte Carlo tree search,
with no human games or hand-written strategy.

> **Status:** mid-refactor on the `refactor` branch. Rules, search, network,
> players, training and evaluation are done and tested; the web UI comes
> next. See [docs/ROADMAP.md](docs/ROADMAP.md).

## Setup

```sh
conda env create -f environment.yml    # creates `alpha_checkers2`
conda activate alpha_checkers2
```

Needs an NVIDIA GPU with CUDA 13 drivers for training; play and tests also
run on CPU.

## Play

```sh
python play.py                                   # you (Black) vs alpha-beta
python play.py --black ab:4 --white human        # you play White
python play.py --white mcts:rollout:400 --analysis
python play.py --black ab:4 --white random --games 20   # quick match
```

Enter moves in PDN notation: `11-15` for a step, `22x15` or `22x15x6` for
captures. Type `?` for the square numbers and legal moves, `undo`, `fen` or
`quit`.

Player specs:

| Spec                            | Player                               |
| ------------------------------- | ------------------------------------ |
| `human`                         | you, at the keyboard                 |
| `random`                        | uniform random legal moves           |
| `ab[:depth]`                    | alpha-beta search (default depth 6)  |
| `policy:<checkpoint>`           | the network's top move, no search    |
| `mcts:<checkpoint>[:sims]`      | tree search guided by the network    |
| `mcts:uniform[:sims]`           | tree search, no network              |
| `mcts:rollout[:sims]`           | tree search with random playouts     |

## Train

```sh
python train.py --config configs/debug.json --run runs/debug   # ~1 min check
python train.py --config configs/dev.json --run runs/dev       # ~30 s/iter
python train.py --config configs/dev.json --run runs/dev2 --set train.lr=0.001
python train.py --resume runs/dev              # Ctrl+C saves; this continues
python train.py --resume runs/dev --iterations 100   # extend a finished run
```

Each iteration plays a batch of self-play games (256 at once, spread over
`self_play.workers` processes sharing the GPU), adds the searched positions
to a replay buffer, and trains on it.

How self-play uses the machine: worker processes (`self_play.workers`)
run the tree searches on CPU cores, and the trainer process serves all their
network evaluations on the GPU in large combined batches (see
[training/actors.py](training/actors.py)). On an i9-12900K + RTX 3070 the
dev config reaches ~150k simulations/s with 7 workers, against ~40k in a
single process. Past ~7 workers the single server process is the limit, so
more don't help. Each worker needs ~0.5 GB of RAM. `self_play.pin_cpus`
(on by default) gives the server a core of its own and puts workers on the
fast cores first.

In a terminal you get two progress bars: the run (with the latest losses,
draw rate, speed and Elo) and the current phase (self-play games, training
steps, arena games). Only notable events are printed, such as arena
results. Everything else goes to `train.log` in detail, in one standard
layout: each entry has a timestamp and a tag (`RUN`, `ITER`, `PROBES`,
`SAVE`, `ARENA`, `WARN`), then labelled rows (games, search, buffer, train,
loss, memory) or an aligned table (probes, arena results). Use
`--progress off` for plain one-line summaries (the default when output is
redirected to a file).

Everything lands in the run directory:

```
runs/<name>/
  config.json        resolved config (edit via --set, not by hand)
  metrics.jsonl      one row per iteration: losses, outcomes, speed
  arena.jsonl        strength: scores vs alpha-beta, Elo over checkpoints
  probes.json        positions to watch (edit any time)
  probes.jsonl       what the network thinks of them, per iteration
  train.log          detailed log: every iteration, arena, checkpoint
  games/             sampled self-play games and all arena games
  checkpoints/       iter_NNNN.pt, latest.pt, buffer.npz
```

### Measuring progress

Every `arena.every` iterations the network plays matches (colours swapped
over random openings) against alpha-beta and the previous evaluated
checkpoint. `arena.jsonl` records the scores and an Elo chain; the log shows
lines like `arena it 10: ab2 12-5-3 (+160), iter 4 9-8-3 (+106) | Elo 245`.

### Probe positions

Probes are positions the network is evaluated on as it trains: its prior for
each move, its win/draw/loss estimate, and what a search concludes. The run
starts with [configs/probes.json](configs/probes.json); add your own any
time, and back-fill their history over earlier checkpoints:

```sh
python -m training.probes add runs/dev --name kings-corner \
    --fen "W:WK32:BK1,K5" --backfill
python -m training.probes list runs/dev
```

Configs: `debug` (tiny smoke test), `dev` (6x64 network, ~30 s per
512-game iteration on an RTX 3070), `full` (10x128, ~2.5 min per 512-game
iteration, ~12 h for 300). Any
checkpoint can play:

```sh
python play.py --white mcts:runs/dev/checkpoints/latest.pt:400 --analysis
```

## Develop

```sh
pytest -m "not slow"          # fast tests (~5 s); drop -m for everything
ruff format . && ruff check .
pyright --pythonpath "$(python -c 'import sys; print(sys.executable)')"
```

Code style: 80-column hard limit, single quotes, type hints, and a docstring
on every module and function. Each module opens with an explanation of its
concepts; read them in this order:

1. [core/game.py](core/game.py): rules on bitboards, positions, moves, FEN
2. [core/encoding.py](core/encoding.py): what the network sees, and why the
   board is rotated for Black
3. [core/model.py](core/model.py): the residual network and its two heads
4. [core/inference.py](core/inference.py): batched position evaluation
5. [core/mcts.py](core/mcts.py): the tree search, one simulation at a time
6. [core/players.py](core/players.py): everything that can pick a move
7. [training/self_play.py](training/self_play.py): generating games
8. [training/replay_buffer.py](training/replay_buffer.py): what's stored
9. [training/trainer.py](training/trainer.py): the loop and the loss
   ([training/actors.py](training/actors.py): self-play in worker processes)
10. [training/arena.py](training/arena.py): matches and Elo
11. [training/probes.py](training/probes.py): watching chosen positions
12. [training/config.py](training/config.py),
    [training/run_store.py](training/run_store.py): settings and run files

## Layout

```
core/       rules, encoding, network, search, players
training/   self-play, replay buffer, trainer, arena, probes, run directory
configs/    debug / dev / full training configs, default probes
tests/      rules (perft), encoding, search tactics, players
docs/       roadmap
play.py     terminal play and engine matches
train.py    start or resume a training run
```
