# Checkers Alpha: refactor roadmap

Goal: an AlphaZero-style checkers engine that trains fast on one machine
(RTX 3070, 24 threads), and a web UI to play it, watch it, inspect what it
thinks, and follow what it learns while training runs.

Keep the existing shape: `core/` (rules, encoding, model, search), `training/`
(self-play, trainer), `train.py` / `play.py` entry points. Add `server/` and
`ui/` for the web app, and `tests/`.

## Why a rewrite rather than patches

The old code had correctness bugs in the part everything depends on (search),
and its speed problems were structural, not tuning. Each problem and its fix:

- **Selection maximised the child's value**, i.e. picked the opponent's best
  move (`get_ucb`). Now Q is taken from the choosing side's perspective.
- **Terminal leaves backed up a stale network value**. Now the real game
  result is backed up.
- **Unvisited children scored infinity**, so priors were ignored until every
  move had been tried. Now first-play urgency (FPU).
- **Dirichlet noise over all actions**, not just legal ones. Now legal only.
- **One network call per node** (batch size 1). Now leaves are batched across
  many concurrent games.
- **State was a dict of numpy boards**; every move copied the repetition dict
  and scanned all actions. Now an immutable bitboard position.
- **The net couldn't see a multi-jump in progress**, and the board wasn't
  oriented to the mover. Now canonical side-to-move planes plus a jump plane.
- **Trained 20 epochs on only the latest games**. Now a replay buffer with a
  fixed sample-reuse ratio (Phase 3).
- **No measure of strength**. Now an arena vs random, alpha-beta and older
  checkpoints, with Elo (Phase 4).
- **Metrics recovered by parsing log lines**. Now structured `metrics.jsonl`.
- **`play.py` duplicated; `--mcts-searches` ignored**. Now rewritten around a
  `Player` interface.

Old checkpoints were trained against the inverted search with an incompatible
encoding, so they were deleted.

## Conventions

- **80 columns, hard limit, in every code file.** Single quotes, type hints on
  all functions. Enforced with `ruff format`, `ruff check` and `pyright`
  (config in `pyproject.toml`). Run `pyright --pythonpath <env python>` so it
  resolves the conda env's packages.
- Every module opens with a docstring explaining its concepts; every docstring
  starts with a one-line summary. Suggested reading order:
  `core/game.py` -> `core/encoding.py` -> `core/model.py` ->
  `core/inference.py` -> `core/mcts.py` -> `core/players.py` -> `training/`.
- Tests live in `tests/` and run with `pytest` (`-m "not slow"` skips perft
  depth 8).

## Target layout

```
core/
  game.py          bitboard rules: immutable Position, moves, draws, FEN
  encoding.py      Position -> canonical planes; board <-> network actions
  model.py         ResNet: per-square policy head + win/draw/loss head
  inference.py     evaluators: network (batched, bf16), uniform, rollouts
  mcts.py          PUCT search with batched leaves, FPU, noise, tree reuse
  players.py       random, alpha-beta, raw policy, MCTS; make_player(spec)
training/
  config.py        dataclass configs, JSON files, `--set a.b=c` overrides
  self_play.py     N concurrent games, one GPU batch per search step
  actors.py        self-play in worker processes sharing the GPU
  progress.py      console progress bars, memory stats
  fileio.py        atomic, Windows-lock-tolerant file writes
  replay_buffer.py windowed buffer of compact features
  trainer.py       self-play -> train -> probes -> checkpoint -> arena
  arena.py         matches from randomised openings, Elo chain
  probes.py        named positions evaluated during training
  run_store.py     run directory, jsonl writers, checkpoints, resume
server/            FastAPI: analysis, games, runs, probes, live feed
ui/                React + TypeScript frontend (Vite)
tests/             rules (perft), encoding, tactics, training smoke test
train.py  play.py  serve.py
```

## Run directory: the contract between trainer and UI

The trainer only writes files; the UI server only reads them (plus a control
file). Either can restart without the other.

```
runs/<name>/
  config.json            full resolved config
  status.json            heartbeat: phase, progress, ETA
  metrics.jsonl          per iteration: losses, lr, grad norm, entropy,
                         sims/s, outcomes, game length, buffer size, timings
  arena.jsonl            arena results + Elo
  probes.json            probe positions (editable while training)
  probes.jsonl           per iteration: prior, visits, Q, WDL per move
  games/iter_NNNN.jsonl  sampled self-play games with per-move search stats
  checkpoints/           iter_NNNN.pt, latest.pt: model, optimiser, config,
                         RNG state (+ replay buffer on latest)
  control.json           written by the UI: pause / stop / checkpoint now
  train.log              human-readable log
```

## Key design choices

- **Rules**: American checkers / English draughts (compulsory capture, men
  move and capture forward, kings both ways, crowning ends the turn).
  Multi-jumps are hop-by-hop actions, keeping the action space at 256. Draw
  rules are configurable.
- **Canonical view**: the network always sees the board from the side to
  move, rotated 180 degrees for Black (never mirrored: rotation keeps dark
  squares dark).
- **Win/draw/loss value head** instead of a single tanh: draws are common in
  checkers, and the UI can show all three.
- **Search**: forced moves skip search. Temperature 1 for the opening plies,
  then argmax. Tree reuse between moves.
- **Self-play parallelism**: many games (128-512) in one process with leaves
  batched into one forward pass. If Python tree work becomes the bottleneck,
  add actor processes feeding a shared GPU inference process.
- **Training**: replay window of recent games, training steps set by a
  sample-reuse ratio, loss masked to legal moves, AdamW, bf16 autocast.
- **Evaluation**: alpha-beta is the fixed yardstick; checkpoints also play
  each other from randomised openings for an Elo curve. No gating.
- **Probes**: a probe is a named FEN with an optional note and expected move.
  The trainer re-reads `probes.json` each iteration, so a position added from
  the UI mid-run is picked up. A back-fill action evaluates a new probe
  against every saved checkpoint.

## UI features

- **Play / analyse**: SVG board, click-to-move including multi-jumps,
  legal-move highlights, arrows sized by probability (prior vs search visits),
  WDL eval bar, move table (P, N, Q, WDL), principal variation on hover,
  undo/redo, flip, FEN copy/paste, model and sims per side, live search.
- **Watch**: any player vs any player; speed, pause, step; eval graph.
- **Position editor**: place pieces, set side to move, then analyse, play
  from here, or save as a probe.
- **Training dashboard**: live curves, run comparison, pause/stop.
- **Probe explorer**: checkpoint slider animating the arrows, plus charts of
  each move's probability and the WDL over training.
- **Game browser**: replay sampled self-play games with their search stats.
- **Model compare**: two checkpoints on one position; head-to-head matches.

## Phases

Each phase ends usable and tested.

0. ✅ **Housekeeping**: branch, `environment.yml`, pytest, dead files removed.
1. ✅ **Engine**: bitboard `game.py`, FEN, draw rules. Perft matches the
   published counts to depth 8; ~15x faster than the old engine.
2. ✅ **Search + model**: encoding, model, fixed MCTS, players including
   alpha-beta, tactical tests, `play.py` CLI. ~16-21k sims/s batched on the
   GPU vs ~35-200 before.
3. ✅ **Training**: self-play with a never-draining pool of 256 games (full
   GPU batches), playout cap randomisation, replay buffer, masked loss,
   cosine LR, run directory, Ctrl+C checkpoint and resume. Then tuned:
   self-play in two alternating groups so GPU and CPU work overlap,
   list-based tree statistics, vectorised softmax, CUDA graphs for the
   forward pass. ~35-40k sims/s; a 256-game dev iteration takes ~40 s.
   Progress bars on the console, a detailed `train.log` with memory use,
   crash-safe (atomic, Windows-lock-tolerant) writes.
4. ✅ **Measurement**: arena (batched matches from colour-swapped random
   openings vs alpha-beta and the previous checkpoint, Elo chain), probe
   positions evaluated every iteration (raw prior + search), editable
   mid-run, back-fill over old checkpoints, `python -m training.probes`
   CLI. An evaluation costs ~100 s on the GPU.
5. **Server + core UI**: API, board, play, analysis overlays.
6. **Full UI**: watch, editor, dashboard, probe explorer, game browser,
   compare.
7. ✅ **Performance round 2**: an inference server. CPU-only worker
   processes run the searches; the trainer process holds the only copy of
   the network on the GPU and serves every worker's evaluations in combined,
   dynamically sized batches, with up to two batches queued on the GPU.
   Positions and outputs pass through shared memory with a state mailbox
   (Windows pipes are too slow for the hot path). Channels-last tensors
   (+20-45% GPU throughput) and CPU pinning (server on its own performance
   core) on top. Dev config: ~40k -> ~150k sims/s at 7 workers. (A first
   version gave each worker its own GPU copy; on Windows the GPU
   time-slices between CUDA contexts, so it stopped scaling at 2.)
   Remaining ideas, if ever needed: send compact features instead of
   planes and expand them on the GPU, a Cython/numba engine and search.
   Soak-tested: memory is flat over long runs once the replay buffer has
   filled (a GPU leak of ~8 MB per CUDA graph capture, from a new stream
   each time, was found this way and fixed).
