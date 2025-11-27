# Checkers Alpha Training Metrics

This file documents the runtime metrics and artifacts produced by training runs.

Per-run layout
- `models/<run>/models/`
  - `config_snapshot.json`: compact snapshot of `args` and basic game config used for the run.
- `models/<run>/logs/`
  - `training_<timestamp>.log`: human-readable, timestamped log with one-line-per-iteration summaries.
- `models/<run>/checkpoints/` and `models/<run>/models/`
  - Saved PyTorch model and optimizer state files and checkpoints.
- `mlruns/` and `mlflow.db` (optional)
  - MLflow artifact folder and SQLite DB if MLflow is enabled.

Per-iteration log format

Each iteration emits a single timestamped INFO line in the run log with only the fields required by the user. Example format (exact):

`2025-11-27 16:46:24,627 - INFO - 1: Games 3 | P1:1 P-1:2 D:0 [Balanced] | Avg moves 77.3 | Avg MCTS/move 0.02 | Samples 232 | Policy Loss 3.824 Value Loss 0.372 Total Loss 4.196 | LR 0.0100 | Time 4.5s | Selfplay 3.93s Train 0.48s | ETA 0h 0m | Captures 57 Promotions 8 MultiJumps 6 | Max Win Streak 2 | GradNorm 2.071 MaxGrad 0.362`

Fields included in the line (in order):
- `iteration`: training iteration index (prefixed before the colon).
- `Games`: number of completed self-play games that iteration.
- `P1 / P-1 / D`: counts of wins for player 1, player -1, and draws.
- `[Balanced]` / bias tag: a short balance indicator inserted after the outcome counts.
- `Avg moves`: average moves per game (float).
- `Avg MCTS/move`: wall-clock seconds spent in MCTS per move (float).
- `Samples`: number of training samples collected for the iteration.
- `Policy Loss` / `Value Loss` / `Total Loss`: training losses for the iteration.
- `LR`: learning rate used.
- `Time`: total wall-clock time for the iteration (seconds).
- `Selfplay` and `Train`: breakdown of time spent in self-play and training (seconds).
- `ETA`: estimated remaining time as `Xh Ym`.
- `Captures` / `Promotions` / `MultiJumps`: aggregated event counts for the iteration.
- `Max Win Streak`: longest winning streak observed during the iteration.
- `GradNorm` / `MaxGrad`: gradient L2 norm and maximum absolute gradient value for the iteration.

Notes
- The per-iteration log line is intentionally compact and stable for easy grepping and parsing.
- The repo contains a `training/utils.py` helper to reconstruct a JSON history from logs if needed. The canonical `training_history` JSON is produced once at the end of training.

Quick Run Commands (Windows `cmd.exe`)

- Play locally (human vs human):

```cmd
python play.py --mode pvp
```

- Play human vs agent (load a model checkpoint, human controls bottom player by default):

```cmd
python play.py --mode pva --checkpoint models/my_run/models/model_10.pt --device cpu --mcts-searches 100
```

- Run an agent vs agent debug match (no checkpoint = uniform random agents):

```cmd
python play.py --mode ava --mcts-searches 50
```

- Train (basic):

```cmd
python train.py --config configs/dev.json --model-dir models/my_run
```

- Train with a fixed seed (applies to Python/NumPy/PyTorch/CUDA):

```cmd
python train.py --config configs/dev.json --model-dir models/my_run --seed 42
```

- If you need reproducible GPU runs, set the following environment variables in your cmd session before starting Python (they must be set before importing CUDA/PyTorch):

```cmd
set CUBLAS_WORKSPACE_CONFIG=:4096:8
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
```

Then run the `python train.py ... --seed 42` command in the same session.

- Start the MLflow UI (from repo root). This points the UI at the local SQLite store and `./mlruns` artifacts directory:

```cmd
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

Open `http://localhost:5000` to view runs and artifacts.

**Using MLflow**

- **Install**: MLflow is optional. Install with `pip` if you want to enable logging:

```cmd
pip install mlflow
```

- **Enable during training**: MLflow can be enabled either in the JSON config (`training_args.use_mlflow = true`) or via the training CLI flags. The CLI flags override config values:

```cmd
python train.py --config configs/dev.json --model-dir models/my_run --use-mlflow
```

- **Set experiment / tracking URI**: By default the project maps file-style URIs to an on-disk SQLite DB at `mlflow.db` and stores artifacts under `./mlruns`. You can override the experiment name and tracking URI via config or CLI:

```cmd
python train.py --config configs/dev.json --model-dir models/my_run --use-mlflow --mlflow-experiment my-experiment --mlflow-tracking-uri sqlite:///C:\path\to\mlflow.db
```

- **Defaults & behavior**: If no tracking URI is provided, the trainer will use `sqlite:///mlflow.db` and `./mlruns` as the artifact root. The `training/mlflow_logger.py` wrapper will silently no-op if `mlflow` isn't installed, so enabling `--use-mlflow` without installing the package will not crash the run (it will log that MLflow is disabled).

- **What gets logged**: The trainer logs selected metrics each iteration, optional artifacts (checkpoints and final `training_history` JSON), and run-level params such as training hyperparameters. Use `mlflow_log_interval` in your config to control metric logging frequency.

Notes on determinism

- Setting `CUBLAS_WORKSPACE_CONFIG` and limiting BLAS/OpenMP threads reduces GPU nondeterminism, but some ops may still be non-deterministic or unsupported in deterministic mode. If `torch.use_deterministic_algorithms(True)` raises an error, it indicates an unsupported op; consider running on CPU for full reproducibility or replacing the offending op.

