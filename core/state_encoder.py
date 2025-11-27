# core/state_encoder.py

import numpy as np


class StateEncoder:
    def __init__(self, game):
        self.game = game
    
    def encode_state(self, state, player):
        """State encoding with historical timesteps and additional features."""
        board = state['board']
        board_timesteps = state.get('board_timesteps', [board])
        
        # Ensure we have enough timesteps (pad with zeros if needed)
        timesteps_needed = self.game.history_timesteps
        while len(board_timesteps) < timesteps_needed:
            zero_board = np.zeros_like(board)
            board_timesteps = [zero_board] + board_timesteps
        
        # Take only the most recent timesteps
        recent_timesteps = board_timesteps[-timesteps_needed:]
        
        # Create timestep planes for each historical board
        timestep_planes = []
        for timestep_board in recent_timesteps:
            if player == 1:
                # Player 1's perspective
                planes = [
                    timestep_board == -2,  # Opponent kings
                    timestep_board == -1,  # Opponent normal pieces
                    timestep_board == 1,   # Player normal pieces  
                    timestep_board == 2    # Player kings
                ]
            else:  # player == -1
                # Player -1's perspective  
                planes = [
                    timestep_board == 2,   # Opponent kings
                    timestep_board == 1,   # Opponent normal pieces
                    timestep_board == -1,  # Player normal pieces
                    timestep_board == -2   # Player kings
                ]
            timestep_planes.extend(planes)
        
        # Additional feature planes
        board_hash = board.tobytes()
        current_repetitions = state['state_repetitions'].get(board_hash, 0)
        repetition_plane = np.full((self.game.row_count, self.game.col_count), 
                                 min(current_repetitions / self.game.repetition_limit, 1.0), 
                                 dtype=np.float32)
        
        no_progress_plane = np.full((self.game.row_count, self.game.col_count), 
                                  min(state['no_progress_moves'] / self.game.draw_move_limit, 1.0), 
                                  dtype=np.float32)
        
        player_plane = np.full((self.game.row_count, self.game.col_count), 
                             1.0 if player == 1 else 0.0, 
                             dtype=np.float32)
        
        # Combine all planes
        all_planes = timestep_planes + [repetition_plane, no_progress_plane, player_plane]
        encoded_state = np.stack(all_planes).astype(np.float32)
        
        return encoded_state
    
    def get_num_channels(self):
        """Return the number of channels in the encoded state."""
        return 4 * self.game.history_timesteps + 3  # 4 for each timestep, plus 3 additional feature planes