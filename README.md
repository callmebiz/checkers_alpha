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

## Play in the browser

Double-click **`start.bat`**: it builds the UI if needed, starts the
server and opens the browser (close its window to stop). A server left
running from before a code change is replaced automatically. By hand:

```sh
cd ui && npm install && npm run build && cd ..   # once, and after UI changes
python serve.py                                  # http://127.0.0.1:8020
```

Play against any checkpoint under `runs/` (pick the run, iteration and
search budget per side), against alpha-beta, or let two computers play.
Click a piece, then a marked square; a multi-jump continues with the same
piece. Live analysis shows the network's win/draw/loss estimate (the bar
beside the board), an arrow per move sized by its share of the search
(or the network's prior), and a table of prior, visits, Q and the line
expected after each move; hover a row to see that line on the board.
Under the board, chess-style controls: to the start, back, play/pause,
forward, to the end (Home ← space → End). Play/pause decides whether
computers move on their own: while running, a person moves only their
own side; paused, you can move any piece for either side (force a bad
move on a computer, then play on to see how it copes). Games with a
person start running, games between computers start paused, and forward
steps one computer move. Players and models can be changed at any point,
a new game can start with either side, and **Edit position** sets up any
position (it loads paused, so you can study the analysis first).
Below them, the evaluation graph (Black's, the draw's and White's
chances over the game; click to jump to a turn). Click any turn in the
record to go there, flip (F), take back, copy or load a FEN.

**Game review** has the strongest checkpoint (the best-rated in any
tournament under `runs/`) search every turn and judge it, as chess apps
do: each move's cost is the share of a point it gave away (draws count
half), and that sorts it into great / best / excellent / good /
inaccuracy / mistake / miss / blunder, or forced when it was the only
legal move. Each side gets an accuracy, symbols appear in the record and
on the board, and errors are marked on the graph. A review takes ~5-15 s
(see [server/review.py](server/review.py) for the rules).

For UI development, run `npm run dev` in `ui/` (it forwards `/api` to
`serve.py`) and `npm run check` before finishing (types, Prettier, 80
columns, tests).

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
[training/actors.py](training/actors.py)). Workers also play the arena's
games. Past ~7 workers the single server process is the limit, so more
don't help. Each worker needs ~0.5 GB of RAM. `self_play.pin_cpus` (on by
default) gives the server a core of its own and puts workers on the fast
cores first.

How the network is run (`backend`): `auto` (default) runs iteration 2 on
TensorRT and iteration 3 on CUDA graphs, then keeps whichever played
faster (the log says which, with both speeds). TensorRT computes the
network ~2x faster but costs more CPU per call, so it wins for the large
`full` network (GPU-bound) and loses for the small `dev` one (whose server
is CPU-bound). Without the optional TensorRT packages, CUDA graphs are
used.
Repeated positions are answered from a per-worker cache
(`self_play.eval_cache`): 15-30% of lookups hit and skip the GPU.

Speed on an i9-12900K + RTX 3070, per 512-game iteration:

| config | before workers | now |
|---|---|---|
| dev (6x64, 200 sims) | ~60 s | ~20 s |
| full (10x128, 400 sims) | ~290 s | ~70 s |

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

Configs: `debug` (tiny smoke test), `dev` (6x64 network, ~20 s per
512-game iteration on an RTX 3070), `full` (10x128, ~75 s per 512-game
iteration, ~6 h for 300). `train.time_budget_hours` stops a run after that
much training instead (the learning rate then anneals over the hours).
Any checkpoint can play:

```sh
python play.py --white mcts:runs/dev/checkpoints/latest.pt:400 --analysis
```

### Long runs: train for days, on and off

One network trained over days or weeks, stopped and resumed at will
([experiments/long_run.py](experiments/long_run.py)). Double-click
**`train.bat`** to start or resume, **`train_stop.bat`** to stop cleanly
(training finishes its iteration and saves; the GPU is then free for the
UI). Resuming restores everything: weights, optimiser, counters, random
state and the replay buffer, however long the break.

- The learning rate is open-ended: 0.002, halved every 24 h of training,
  down to 0.0001. Change it (or any setting but the network's shape)
  when resuming: `train.bat --set train.lr=0.0005`.
- Training runs in segments of a few hours (fresh processes); between
  them a gauntlet rates the newest checkpoint against fixed opponents
  (alpha-beta 4/6/8 and the size study's networks), on one scale for the
  whole run. Crashed or stalled segments resume on their own.
- `runs/long_10x128/STATUS.md` (every 5 minutes) and `REPORT.md` (hourly
  and after each gauntlet) show where it is. Its checkpoints appear in the UI.

`python train.py --resume <run> --set train.lr=0.001` changes settings
for any run; `control.json` with `{"stop": true}` in a run directory
stops it cleanly after the current iteration.

### Skills: endgames and tactics

Fixed puzzle sets ([training/suites.py](training/suites.py), saved in
`configs/suites/`) say what a network is good at, not just how often it
wins:

- **Endgames**, solved exactly by [core/endgames.py](core/endgames.py)
  (every position with up to three pieces, and every two-against-two,
  ~5.6 million): *judgement* (is the predicted outcome the true one?),
  *the move* (same position, other side to move, different result), and
  *deep wins* (25-60 turns of exact play).
- **Tactics** from real games, confirmed by alpha-beta at every depth from
  6 to 12 turns: *shots* (a sacrifice that wins material) and *only
  moves* (everything else loses a man).

The long run scores each gauntlet checkpoint on them (`REPORT.md`,
"Skills"). By hand:

```sh
python -m training.suites score runs/long_10x128/checkpoints/iter_0864.pt
python -m training.suites build    # rebuild the sets (~10 min)
```

### Experiments and tournaments

[training/tournament.py](training/tournament.py) rates many players
(checkpoints with or without search, alpha-beta) on one Elo scale from a
JSON plan; it resumes if interrupted. The network-size study trains
10x128, 8x96 and 6x64 networks for equal time, plays a tournament, and
writes `REPORT.md` with plots (`STATUS.md` while it runs):

```sh
python -m experiments.size_study          # ~17 h; rerun to resume
python -m experiments.size_study --dry    # ~30 min rehearsal
python -m experiments.report runs/size_study   # rebuild the report
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
server/     FastAPI: game states, analysis, engine moves, live WebSocket
ui/         React + TypeScript (Vite) board, play and analysis
experiments/ studies built on training/ (network size), their reports
tests/      rules (perft), encoding, search tactics, players
docs/       roadmap
play.py     terminal play and engine matches
train.py    start or resume a training run
serve.py    the web UI and its API
```
