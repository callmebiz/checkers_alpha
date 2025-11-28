# training/self_play.py

import numpy as np
import time
import os
import tempfile
import torch
import multiprocessing as mp
from tqdm import tqdm


class SelfPlayManager:
    def __init__(self, game, mcts, state_encoder, logger=None):
        self.game = game
        self.mcts = mcts
        self.state_encoder = state_encoder
        self.logger = logger
    
    def get_temperature_schedule(self, iteration, move_count, args):
        """Get temperature for current training iteration and move"""
        progress = iteration / args['num_iterations']
        
        base_temp = args.get('initial_temperature', 1.2)
        final_temp = args.get('final_temperature', 0.8)
        training_temp = base_temp - (base_temp - final_temp) * progress
        
        if move_count < 10:
            move_temp_multiplier = 1.2
        elif move_count < 30:
            move_temp_multiplier = 1.0
        else:
            move_temp_multiplier = 0.8
        
        final_temperature = training_temp * move_temp_multiplier
        return max(0.1, min(2.0, final_temperature))
    
    def play_game(self, starting_player, game_idx=None, iteration=0, args=None, initial_state=None):
        """Play a single self-play game and return training data"""
        memory = []
        if initial_state:
            state = initial_state.copy()
        else:
            state = self.game.get_initial_state()

        current_player = starting_player
        move_count = 0
        mcts_time = 0
        inference_time = 0.0
        # Counters for game events
        captures = 0
        promotions = 0
        multi_jumps = 0
        
        game_start_time = time.time()
        
        while True:
            # Check for terminal state
            _, is_terminal = self.game.get_value_and_terminated(state, current_player)
            if is_terminal:
                break
                
            # Get temperature for this move
            temperature = self.get_temperature_schedule(iteration, move_count, args)
            
            # MCTS search with temperature control
            mcts_start = time.time()
            action_probs = self.mcts.search(
                state, 
                current_player, 
                temperature=temperature,
                add_noise=(move_count < 30)
            )
            mcts_time += time.time() - mcts_start
            inference_time += self.mcts.get_last_search_inference_time()
            
            # Store the state from the perspective of the current player
            memory.append([state.copy(), action_probs.copy(), current_player])
            
            # Select action from MCTS probabilities
            if np.sum(action_probs) == 0:
                if self.logger:
                    self.logger.warning(f"No valid moves for player {current_player} in game {game_idx}")
                break
            
            # Sample action from MCTS probabilities
            action = np.random.choice(self.game.action_size, p=action_probs)
            
            # Apply the move
            prev_board = state['board'].copy()
            prev_jump_again = bool(state.get('jump_again'))
            state = self.game.get_next_state(state, action, current_player)
            # Detect captures (opponent piece removed)
            # count pieces before/after for opponent
            opp = self.game.get_opponent(current_player)
            prev_opp_count = (prev_board == opp).sum() + (prev_board == 2*opp).sum()
            new_board = state['board']
            new_opp_count = (new_board == opp).sum() + (new_board == 2*opp).sum()
            if new_opp_count < prev_opp_count:
                captures += (prev_opp_count - new_opp_count)
            # Detect promotion: piece at destination changed from man to king
            # Rough heuristic: if destination square now has abs==2 and previously was not king
            # We can't know destination easily here, but detect increases in king counts for current player
            prev_king_count = (prev_board == 2*current_player).sum() + (prev_board == -2*current_player).sum()
            new_king_count = (new_board == 2*current_player).sum() + (new_board == -2*current_player).sum()
            if new_king_count > prev_king_count:
                promotions += (new_king_count - prev_king_count)
            # Multi-jump: if jump again flag is set after a capture
            if state.get('jump_again'):
                # if a jump occurred and there is a continuation
                # treat as a multi-jump event
                if new_opp_count < prev_opp_count:
                    multi_jumps += 1
            
            # Switch players if not multi-jumping
            if not state.get('jump_again'):
                current_player = self.game.get_opponent(current_player)
            
            move_count += 1
            
            # # Safety check for extremely long games
            # if move_count > 1000:
            #     if self.logger:
            #         self.logger.warning(f"Game {game_idx} exceeded 1000 moves, treating as draw")
            #     break
            
        # Game has ended
        final_state = state
        game_time = time.time() - game_start_time
        
        # Create training examples with proper value assignment
        return_memory = []
        for hist_state, hist_action_probs, hist_player in memory:
            # Get the final game outcome from this historical player's perspective
            hist_outcome, _ = self.game.get_value_and_terminated(final_state, hist_player)
            
            # Encode state from the perspective of the player who made the move
            encoded_state = self.state_encoder.encode_state(hist_state, hist_player)
            
            return_memory.append((
                encoded_state,
                hist_action_probs,
                hist_outcome
            ))
            
        # For statistics, get the outcome from player 1's perspective
        final_outcome_p1, _ = self.game.get_value_and_terminated(final_state, 1)
        
        # Return game statistics
        game_stats = {
            'moves': move_count,
            'outcome_p1': final_outcome_p1,
            'game_time': game_time,
            'mcts_time': mcts_time,
            'avg_mcts_per_move': mcts_time / max(move_count, 1),
            'inference_time': inference_time,
            'avg_inference_per_move': inference_time / max(move_count, 1),
            'training_examples': len(return_memory),
            'captures': captures,
            'promotions': promotions,
            'multi_jumps': multi_jumps
        }
        
        return return_memory, final_outcome_p1, move_count, game_stats

    def play_games_parallel(self, num_games, starting_player, iteration, args, initial_state=None, num_processes=None):
        """Wrapper method to run multiple self-play games in parallel.

        This instance method delegates to the module-level `play_games_parallel` helper
        so existing callers on `SelfPlayManager` continue to work.
        """
        # Import local reference to avoid circular import at module import time
        return play_games_parallel(self, num_games, starting_player, iteration, args, initial_state=initial_state, num_processes=num_processes)


# Worker globals populated by pool initializer
_WORKER_STATE = {}


def _worker_init(model_state_path, model_config, game_config, training_args):
    """Initializer for worker processes: construct game, encoder, model and MCTS once."""
    # Import here so worker processes don't import heavy modules at parent import time
    from core.game import Checkers
    from core.state_encoder import StateEncoder
    from core.model import ResNet
    from core.mcts import MCTS

    # Recreate game and encoder
    game = Checkers(**game_config)
    encoder = StateEncoder(game)

    # Build model on CPU for worker inference
    device = torch.device('cpu')
    model = ResNet(
        input_channels=model_config['input_channels'],
        action_size=model_config['action_size'],
        board_size=tuple(model_config['board_size']),
        num_resBlocks=model_config['num_resBlocks'],
        num_hidden=model_config['num_hidden'],
        device=device,
        use_checkpoint=model_config.get('use_checkpoint', False)
    )

    # Load state dict saved by parent (map to CPU)
    try:
        sd = torch.load(model_state_path, map_location='cpu')
        model.load_state_dict(sd)
    except Exception:
        # If loading fails, continue with randomly initialized model
        pass
    model.eval()

    # MCTS uses the shared training args passed in
    mcts = MCTS(game, training_args, model, encoder)

    # Store into global worker state
    _WORKER_STATE['game'] = game
    _WORKER_STATE['encoder'] = encoder
    _WORKER_STATE['model'] = model
    _WORKER_STATE['mcts'] = mcts


def _worker_play_one(task):
    """Play a single game in a worker using the pre-initialized worker state.

    Task is a tuple: (task_idx, starting_player, iteration, args, initial_state)
    Returns the tuple (game_memory, final_outcome_p1, move_count, game_stats)
    """
    from training.self_play import SelfPlayManager  # local import to avoid circular issues

    _, starting_player, iteration, args, initial_state = task

    game = _WORKER_STATE['game']
    mcts = _WORKER_STATE['mcts']
    encoder = _WORKER_STATE['encoder']

    spm = SelfPlayManager(game, mcts, encoder, logger=None)
    return spm.play_game(starting_player, iteration=iteration, args=args, initial_state=initial_state)


def play_games_parallel(self, num_games, starting_player, iteration, args, initial_state=None, num_processes=None):
    """Play multiple self-play games in parallel using a process pool.

    Returns a list of per-game results in the same format as `play_game`.
    """
    # Determine worker count
    workers = num_processes or max(1, (os.cpu_count() or 1) - 1)

    # Prepare model snapshot for workers (save to temp file)
    model_state_path = None
    try:
        # Prefer using the model contained in the MCTS instance
        model = self.mcts.model
        fd, model_state_path = tempfile.mkstemp(suffix='.pt')
        os.close(fd)
        torch.save(model.state_dict(), model_state_path)
    except Exception:
        model_state_path = None

    # Build serializable model_config and game_config
    model_config = {
        'input_channels': self.state_encoder.get_num_channels(),
        'action_size': self.game.action_size,
        'board_size': (self.game.row_count, self.game.col_count),
        'num_resBlocks': getattr(self.mcts.model, 'num_resBlocks', 3),
        'num_hidden': getattr(self.mcts.model, 'num_hidden', 64),
        'use_checkpoint': getattr(self.mcts.model, 'use_checkpoint', False)
    }

    game_config = {
        'row_count': self.game.row_count,
        'col_count': self.game.col_count,
        'buffer_count': self.game.buffer_count,
        'draw_move_limit': self.game.draw_move_limit,
        'repetition_limit': self.game.repetition_limit,
        'history_timesteps': self.game.history_timesteps
    }

    # Create tasks: each task represents a single game to keep progress reporting fine-grained
    tasks = []
    for gid in range(num_games):
        tasks.append((gid, starting_player, iteration, args, initial_state))

    results = []

    # Use multiprocessing Pool with initializer to load model once per worker
    try:
        with mp.Pool(processes=workers, initializer=_worker_init, initargs=(model_state_path, model_config, game_config, args)) as pool:
            for res in tqdm(pool.imap_unordered(_worker_play_one, tasks), total=len(tasks), desc='Parallel self-play'):
                results.append(res)
    finally:
        # Clean up temporary model file
        try:
            if model_state_path and os.path.exists(model_state_path):
                os.remove(model_state_path)
        except Exception:
            pass

    return results