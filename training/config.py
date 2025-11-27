# training/config.py

import json
import os
import sys


def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"   Loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found")
        print("Available configurations:")
        for file in os.listdir('.'):
            if file.endswith('.json'):
                print(f"  - {file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{config_path}': {e}")
        sys.exit(1)


def validate_config(config):
    """Validate configuration has required sections"""
    required_sections = ['game_config', 'model_config', 'training_args', 'optimizer_config']
    missing_sections = [section for section in required_sections if section not in config]
    
    if missing_sections:
        print(f"Error: Missing required sections in config: {missing_sections}")
        sys.exit(1)
    
    print("Configuration validation passed")


def print_config_summary(config):
    """Print a concise configuration summary — compact, one-screen friendly.

    Shows the most important settings for quick inspection before running.
    """
    game_config = config.get('game_config', {})
    model_config = config.get('model_config', {})
    training_args = config.get('training_args', {})
    optimizer_config = config.get('optimizer_config', {})

    print("\nCONFIG SUMMARY")

    print("\nGame: {}x{}, buffer={}, history={}".format(
        game_config.get('row_count', 8), game_config.get('col_count', 8),
        game_config.get('buffer_count', 2), game_config.get('history_timesteps', 4)
    ))

    print("Model: ResNet blocks={}, hidden={}, checkpointing={}".format(
        model_config.get('num_resBlocks', '?'), model_config.get('num_hidden', '?'), model_config.get('use_checkpoint', False)
    ))

    print("Training: iterations={}, self-play/games={}, epochs={}, batch={}, MCTS={}".format(
        training_args.get('num_iterations', '?'), training_args.get('num_self_play_iterations', '?'),
        training_args.get('num_epochs', '?'), training_args.get('batch_size', '?'), training_args.get('num_searches', '?')
    ))

    print("Optimizer: lr={}, weight_decay={}, mixed_precision={}".format(
        optimizer_config.get('lr', '?'), optimizer_config.get('weight_decay', 0.0), config.get('use_mixed_precision', True)
    ))