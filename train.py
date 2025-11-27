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
      run_name = args.run_name or f"model_{timestamp}"
      model_dir = os.path.join('runs', run_name)

    # Define per-run log directory and pass it to Trainer
    log_dir = os.path.join(model_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    trainer = Trainer(model, optimizer, game, mcts, encoder, training_args, initial_state=None,
                      log_dir=log_dir, use_mixed_precision=config.get('use_mixed_precision', True))

    os.makedirs(model_dir, exist_ok=True)

    trainer.train(model_dir=model_dir, resume_from=args.resume_from, config_path=args.config)


if __name__ == '__main__':
    main()
