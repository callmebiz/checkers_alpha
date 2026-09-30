#!/usr/bin/env python3
"""Simple training entry point for the Checkers AlphaZero project.

Usage examples:
  python train.py --config configs/dev.json --model-dir models --resume-from models/checkpoint_latest.pt

The script loads the JSON config via `training.config.load_config`, builds the game,
encoder, model, MCTS and Trainer, then runs `Trainer.train()`.
"""
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.optim as optim
from datetime import datetime

from core.game import Checkers
from core.state_encoder import StateEncoder
from core.model import ResNet
from core.mcts import MCTS
from training.trainer import Trainer
from training.config import load_config, validate_config, print_config_summary

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def main():
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument('--config', type=str, default='configs/dev.json', help='Path to JSON config')
    parser.add_argument('--model-dir', type=str, default=None, help='Directory to save models/checkpoints (default: runs/<timestamp>)')
    parser.add_argument('--run-name', type=str, default=None, help='Optional run name used when defaulting model directory')
    parser.add_argument('--resume-from', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA even if available')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: None -> random)')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel self-play (uses multiprocessing)')
    parser.add_argument('--parallel-processes', type=int, default=None, help='Number of parallel worker processes to use for self-play')
    # MLflow overrides
    parser.add_argument('--use-mlflow', action='store_true', help='Enable MLflow logging (overrides config)')
    parser.add_argument('--mlflow-experiment', type=str, default=None, help='MLflow experiment name (overrides config)')
    parser.add_argument('--mlflow-tracking-uri', type=str, default=None, help='MLflow tracking URI (overrides config)')
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)
    print_config_summary(config)

    

    # Global seeding: apply early so all components use the same RNGs
    if args.seed is not None:
      import random as _rand
      import numpy as _np
      _rand.seed(args.seed)
      _np.random.seed(args.seed)
      os.environ['PYTHONHASHSEED'] = str(args.seed)
      try:
        import torch as _torch
        _torch.manual_seed(args.seed)
        if _torch.cuda.is_available() and not args.no_cuda:
          _torch.cuda.manual_seed_all(args.seed)
      except Exception:
        pass

    game_cfg = config['game_config']
    model_cfg = config['model_config']
    training_args = config['training_args']
    optim_cfg = config['optimizer_config']

    # Device selection
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Build game and encoder
    game = Checkers(row_count=game_cfg.get('row_count', 8), col_count=game_cfg.get('col_count', 8),
                    buffer_count=game_cfg.get('buffer_count', 2),
                    draw_move_limit=game_cfg.get('draw_move_limit', 50),
                    repetition_limit=game_cfg.get('repetition_limit', 3),
                    history_timesteps=game_cfg.get('history_timesteps', 4))

    encoder = StateEncoder(game)

    # Build model
    model = ResNet(input_channels=encoder.get_num_channels(),
                   action_size=game.action_size,
                   board_size=(game.row_count, game.col_count),
                   num_resBlocks=model_cfg.get('num_resBlocks', 3),
                   num_hidden=model_cfg.get('num_hidden', 64),
                   device=device,
                   use_checkpoint=model_cfg.get('use_checkpoint', False))

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=optim_cfg.get('lr', 1e-3), weight_decay=optim_cfg.get('weight_decay', 0.0))

    # MCTS: training_args passed through for num_searches, C, etc.
    mcts = MCTS(game, training_args, model, encoder)

    # If resuming from a checkpoint and no explicit model_dir was provided,
    # try to detect the original run directory and resume in-place when
    # the checkpoint's configs/args match the current config.
    def _find_run_dir(path):
        p = os.path.abspath(os.path.dirname(path))
        for _ in range(6):
            # heuristic: a run dir contains a 'logs' subdirectory (and usually 'models'/'checkpoints')
            if os.path.isdir(os.path.join(p, 'logs')) or os.path.isdir(os.path.join(p, 'checkpoints')) or os.path.isdir(os.path.join(p, 'models')):
                return p
            parent = os.path.dirname(p)
            if parent == p:
                break
            p = parent
        return None

    def _checkpoint_compatible(ckpt, game_cfg, model_cfg, training_args):
        # Basic compatibility checks: compare model_config and game_config and a few training args
        try:
            c_model = ckpt.get('model_config', {})
            c_game = ckpt.get('game_config', {})
            c_args = ckpt.get('args', {})
            # Compare model and game configs
            for k in ('num_resBlocks', 'num_hidden', 'use_checkpoint'):
                if k in model_cfg and c_model.get(k) != model_cfg.get(k):
                    return False
            for k in ('row_count', 'col_count', 'buffer_count', 'history_timesteps'):
                if k in game_cfg and c_game.get(k) != game_cfg.get(k):
                    return False
            # Compare a few key training args that affect runtime layout
            for k in ('num_iterations', 'num_self_play_iterations', 'num_searches', 'batch_size', 'num_epochs'):
                if k in training_args and c_args.get(k) != training_args.get(k):
                    return False
        except Exception:
            return False
        return True

    resume_inplace = False
    if args.resume_from and not args.model_dir:
        try:
            ck = torch.load(args.resume_from, map_location='cpu')
            run_dir = _find_run_dir(args.resume_from)
            if run_dir and _checkpoint_compatible(ck, game_cfg, model_cfg, training_args):
                model_dir = run_dir
                resume_inplace = True
                print(f"Resuming in-place into existing run directory: {model_dir}")
        except Exception:
            # If anything fails, fall back to creating a new run directory as usual
            pass

    # Trainer
    # Apply CLI MLflow overrides if provided
    if args.use_mlflow:
      training_args['use_mlflow'] = True
    if args.mlflow_experiment:
      training_args['mlflow_experiment_name'] = args.mlflow_experiment
    if args.mlflow_tracking_uri:
      training_args['mlflow_tracking_uri'] = args.mlflow_tracking_uri

    # Default model_dir to a run folder. If a base dir is given (e.g. --model-dir models)
    # create a per-run subdirectory under it named by --run-name or default `model_<timestamp>`.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.model_dir:
      base_model_root = args.model_dir
      run_name = args.run_name or f"model_{timestamp}"
      model_dir = os.path.join(base_model_root, run_name)
    else:
      # If we already decided to resume in-place above, model_dir is set.
      if not resume_inplace:
        run_name = args.run_name or f"model_{timestamp}"
        model_dir = os.path.join('runs', run_name)

    # Define per-run log directory and pass it to Trainer
    log_dir = os.path.join(model_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    trainer = Trainer(model, optimizer, game, mcts, encoder, training_args, initial_state=None,
                      log_dir=log_dir, use_mixed_precision=config.get('use_mixed_precision', True))

    os.makedirs(model_dir, exist_ok=True)

    # Pass parallelism flags into Trainer.train()
    trainer.train(
      model_dir=model_dir,
      resume_from=args.resume_from,
      use_parallel=args.parallel,
      num_processes=args.parallel_processes,
      config_path=args.config
    )

    # No in-process predictor to clean up (predictor disabled/rolled back)


if __name__ == '__main__':
    main()
