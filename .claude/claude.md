# Checkers Alpha: notes for Claude

AlphaZero-style engine for American checkers, mid-refactor on branch
`refactor`. The plan and phase status live in `docs/ROADMAP.md`; read it
before starting work. The end goal is a web UI (FastAPI + React/TypeScript)
for playing, watching, analysing and following training.

## Environment

- Conda env `alpha_checkers2` (Python 3.13, torch 2.9.1+cu130, RTX 3070).
  Interpreter: `C:/Users/biz/miniconda3/envs/alpha_checkers2/python.exe`.
  The old `alpha_checkers` env is broken (OpenMP clash); don't use it.
- Before finishing any change, run all four and get them clean:
  - `python -m pytest -q -m "not slow"`
  - `python -m ruff format .`
  - `python -m ruff check .`
  - `python -m pyright --pythonpath <env python>` (scope in pyproject.toml)
- UI (`ui/`, React + TypeScript): run `npm run check` there (tsc,
  Prettier, 80 columns via `scripts/check-width.mjs`, Vitest) and
  `npm run build` before finishing. Same rules as Python: 80 columns, a
  comment on every function (`/** ... */`), a header comment per file.
  Browser checks: Playwright with `channel='msedge'` (installed Edge).
- `start.bat` (via `scripts/start.ps1`) reuses a running server only if it
  started after the last change to `server/`, `core/` or `serve.py`;
  otherwise it stops it (and its window) and starts a fresh one. When
  testing by hand, restart the server after changing server code.
- Ports: `serve.py` 8020, the UI dev server 5203. Other projects in
  `E:/Projects` run servers too: check and update the registry in
  `E:/Projects/.claude` before changing a port.
- Smoke-test training end to end with
  `python train.py --config configs/debug.json --run runs/debug`
  (delete `runs/debug` afterwards).

## Code style (hard rules)

- **80 columns maximum in every code file**, including comments, docstrings
  and strings. `ruff` enforces it via `pyproject.toml`; wrap by hand what the
  formatter can't.
- Single quotes, type hints on all function signatures.
- Every module starts with a docstring that explains its concepts, so the
  user can read the code to learn how it works. **Every function and method,
  including tests, nested helpers and `__init__`, has at least a one-line
  docstring.** Every docstring opens with a one-line summary, then a blank
  line and details. Comment non-obvious logic
  (bit tricks, sign flips); don't narrate obvious code.
- Match the existing modules' structure: section dividers
  `# -- name ---...` padded to exactly 80 columns.

## Invariants to keep

- Squares are 0-31 internally, PDN 1-32 in text. Black starts on 1-12,
  moves down and moves first.
- Actions are single hops: `square * 8 + direction * 2 + is_jump` (256; 170
  on the board). Directions are NW, NE, SW, SE, so `3 - d` is the opposite.
- `Position` is immutable; the game only holds positions from `initial()`,
  `from_fen()` or `play()` of a legal action.
- The network sees the side to move as White: Black-to-move positions are
  rotated 180 degrees (`s -> 31 - s`), never mirrored. Actions convert with
  `to_canonical` / `from_canonical`.
- Values are always for the side to move at that node. In search, `W[i]` is
  for the chooser at the node; backup negates only when the mover changes
  (multi-jumps keep the sign).
- Policy is softmaxed over legal moves only, in search and in training.
- Training samples store compact features (`core.encoding.features`), the
  visit distribution in the network frame, a packed legal mask and a
  win/draw/loss class for the side to move.
- `runs/<name>/` is the trainer/UI contract (`training/run_store.py`): the
  trainer writes, the UI reads (`probes.json` may be edited by either).
  Keep its layout stable or update both. Result rows carry a 0-based
  `iteration`; `probes.jsonl` rows may be `backfill` rows, merged by
  iteration (`training.probes.merged_results`).
- Test configs must set `arena.every` (0 or tiny matches): the default
  arena plays 60 full games and makes tests take over a minute.
- Files other processes may read (status, checkpoints, buffer, probes) are
  written through `training/fileio.py`: temp file + rename, retried while
  Windows holds the target open. Never write them with a plain `open()`.
- Logging goes through `training.logs.event(log, TAG, message, to=...)`
  with a tag from RUN / ITER / PROBES / SAVE / ARENA / WARN. `to='file'`
  for full multi-line reports (built by the `*_report` helpers in
  `training/logs.py`), `to='console'` for one-liners, `'both'` for rare
  events. Keep the layout: labelled rows, aligned tables.
- Performance-sensitive paths: `SearchTree.select_leaf/_select_child`
  (plain lists, no numpy), `Position._generate/play`, `NetEvaluator.submit`
  (CUDA graphs, pinned async copy). Profile before and after changing them.
- Self-play workers (`training/actors.py`) are spawned processes: anything
  they need must be picklable and importable at module level, and scripts
  that start them must be files (not stdin). Workers are CPU-only; the
  trainer process is the GPU inference server. Hot-path data goes through
  shared memory and the mailbox states, never pipes. Worker tests are
  marked `slow`.
- Benchmarks: compare seconds per iteration and sims/s, not games/s
  (games carry over between iterations; cache hits count as sims without
  using the GPU). Close other GPU programs first, never benchmark while a
  user's training run is going, and never edit code a running benchmark
  imports (each run spawns fresh workers that load the files on disk).
  When stopping a background benchmark, kill the whole process tree
  (shell, train.py and workers): stopping the task alone leaves them.
- TensorRT (optional `torch-tensorrt`): engines must be re-wrapped after a
  refit (in-place refit silently breaks the recorded CUDA graphs). Players,
  probe back-fill and the arena's previous checkpoint use `backend='graphs'`
  to avoid an engine build. Each refit leaks ~0.65 MB of host RAM inside
  torch-tensorrt (GPU memory stays flat; `torch._dynamo.reset()` doesn't
  help): ~130 MB over a 4 h run, accepted.
- Memory limit that matters on this machine is the Windows commit charge
  (RAM + page file), not free RAM: ~34 of 47 GB is committed at idle with
  the editor and a browser open, and a full training run adds ~10 GB.
- The long run (`runs/long_10x128`, `experiments/long_run.py`) may be
  training: check `runs/long_10x128/long_run.pid` / STATUS.md before any
  GPU work, and never edit `core/`, `training/` or `train.py` while a
  segment runs (each segment and gauntlet starts fresh processes that
  import the files on disk). Stop it with `train_stop.bat`.
- Skill suites (`configs/suites/*.json`) are fixed so scores compare over
  time. Rebuilding them (`python -m training.suites build`) invalidates
  every stored score: delete `runs/*/suites.jsonl` and re-score.
  Endgame truth comes from `core/endgames.py` (no draw rule in the
  values; the suites keep only results decided within 60 turns).
- Long experiments run detached (`Start-Process`), so they survive the
  editor session; `experiments/size_study.py` supervises its training runs
  (restarts stalled or crashed ones from checkpoints) and writes
  `STATUS.md`. Never start GPU work while it runs.
- The machine is an i9-12900K (8 hyperthreaded P-cores = CPUs 0-15, 8
  E-cores = 16-23) with an RTX 3070 (8 GB), 32 GB RAM.

## Working agreements

- Don't commit unless the user asks. Work on the `refactor` branch.
- Delete old data/artifacts as they're replaced (the user asked for this).
- Keep `README.md`, `docs/ROADMAP.md` and this file current when
  commands, layout or phase status change.
