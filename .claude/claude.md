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
  they need must be picklable and importable at module level. Worker tests
  are marked `slow`. Benchmarks on this machine are noisy (other programs
  use the GPU); compare configurations over several iterations.

## Working agreements

- Don't commit unless the user asks. Work on the `refactor` branch.
- Delete old data/artifacts as they're replaced (the user asked for this).
- Keep `README.md`, `docs/ROADMAP.md` and this file current when
  commands, layout or phase status change.
