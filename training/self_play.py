# training/self_play.py

import numpy as np
import time


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
            'training_examples': len(return_memory),
            'captures': captures,
            'promotions': promotions,
            'multi_jumps': multi_jumps
        }
        
        return return_memory, final_outcome_p1, move_count, game_stats