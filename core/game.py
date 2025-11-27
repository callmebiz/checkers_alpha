# core/game.py

import numpy as np
from collections import defaultdict


class Checkers:
    def __init__(self, row_count=8, col_count=8, buffer_count=2,
                 draw_move_limit=50, repetition_limit=3, history_timesteps=4):
        self.row_count = row_count
        self.col_count = col_count
        self.buffer_count = buffer_count
        self.draw_move_limit = draw_move_limit
        self.repetition_limit = repetition_limit
        self.history_timesteps = history_timesteps

        self.playable_positions = []
        self.move_map = {}

        self.move_names = {
            (-1, -1): "Up-Left", (-1, 1): "Up-Right",
            (1, -1): "Down-Left", (1, 1): "Down-Right",
            (-2, -2): "Jump Up-Left", (-2, 2): "Jump Up-Right",
            (2, -2): "Jump Down-Left", (2, 2): "Jump Down-Right"
        }

        self.action_size = self.initialize_action_encoding()

    def initialize_action_encoding(self):
        action_size = 0
        for r in range(self.row_count):
            for c in range(self.col_count):
                # Only consider playable positions (black squares)
                if (r + c) % 2 == 0:
                    continue
                
                # Check all possible moves from this position
                valid_moves = []
                for (dr, dc) in self.move_names.keys():
                    new_r, new_c = r + dr, c + dc
                    
                    # Check if the move is within bounds
                    if 0 <= new_r < self.row_count and 0 <= new_c < self.col_count:
                        valid_moves.append((new_r, new_c, action_size, dr, dc))
                        action_size += 1

                # Store the playable position and its valid moves
                self.playable_positions.append((r, c))
                self.move_map[(r, c)] = valid_moves

        return action_size

    def get_initial_state(self):
        board = np.zeros((self.row_count, self.col_count), dtype=int)
        player_rows = (self.row_count - self.buffer_count) // 2

        # Place pieces for Player -1 (Top)
        for r in range(player_rows):
            for c in range(self.col_count):
                if (r + c) % 2 == 1:
                    board[r, c] = -1  

        # Place pieces for Player 1 (Bottom)
        for r in range(self.row_count - player_rows, self.row_count):
            for c in range(self.col_count):
                if (r + c) % 2 == 1:
                    board[r, c] = 1  

        # Initialize state with empty repetitions and no progress moves
        return {
            'board': board,
            'state_repetitions': defaultdict(int),
            'no_progress_moves': 0,
            'jump_again': None,
            'board_timesteps': [board.copy()]
        }

    def get_valid_moves(self, state, player, enforce_piece=None):
        board = state['board']
        valid_moves = np.zeros(self.action_size, dtype=int)
        jump_moves_exist = False
        potential_jumps = []

        # Check all playable positions for the current player
        for (r, c) in self.playable_positions:
            piece = board[r, c]
            # Check if the piece belongs to the current player
            if piece == 0 or (piece != player and piece != 2 * player):
                continue

            # If a jump is required, check if this piece can jump
            if state['jump_again'] and (r, c) != state['jump_again']:
                continue
            
            # If enforcing a specific piece, skip others
            if enforce_piece and (r, c) != enforce_piece:
                continue

            is_king = abs(piece) == 2
            # Check all possible moves from this position
            for new_r, new_c, action_idx, dr, dc in self.move_map[(r, c)]:
                # Jump move
                if abs(new_r - r) == 2:
                    mid_r, mid_c = (r + new_r) // 2, (c + new_c) // 2
                    if board[mid_r, mid_c] in {-player, -2 * player} and board[new_r, new_c] == 0:
                        if is_king or (player == 1 and dr == -2) or (player == -1 and dr == 2):
                            valid_moves[action_idx] = 1
                            jump_moves_exist = True
                            potential_jumps.append(action_idx)

                # Normal move
                elif board[new_r, new_c] == 0:
                    if not enforce_piece and (is_king or (player == 1 and dr == -1) or (player == -1 and dr == 1)):
                        valid_moves[action_idx] = 1

        # If jump moves exist, remove all normal moves
        if jump_moves_exist:
            valid_moves = np.zeros(self.action_size, dtype=int)
            for action_idx in potential_jumps:
                valid_moves[action_idx] = 1

        return valid_moves

    def get_next_state(self, state, action, player):
        # Create a new state based on the current state
        new_state = {
            'board': state['board'].copy(),
            'state_repetitions': state['state_repetitions'].copy(),
            'no_progress_moves': state['no_progress_moves'],
            'jump_again': state['jump_again'],
            'board_timesteps': state['board_timesteps'].copy()
        }
        
        # Check if the action is valid
        for (r, c), moves in self.move_map.items():
            for new_r, new_c, action_idx, _, _ in moves:
                if action == action_idx:
                    new_state['board'][new_r, new_c] = new_state['board'][r, c]
                    new_state['board'][r, c] = 0
                    
                    # Handle capture
                    capture_made = False
                    if abs(new_r - r) == 2:
                        mid_r, mid_c = (r + new_r) // 2, (c + new_c) // 2
                        new_state['board'][mid_r, mid_c] = 0
                        capture_made = True
                        
                        # Check if the piece can jump again
                        if self.get_valid_moves(new_state, player, enforce_piece=(new_r, new_c)).any():
                            new_state['jump_again'] = (new_r, new_c)
                        else:
                            new_state['jump_again'] = None
                    else:
                        new_state['jump_again'] = None
                    
                    # Handle promotion
                    promotion_made = False
                    if ((player == 1 and new_r == 0) or (player == -1 and new_r == self.row_count - 1)) and abs(new_state['board'][new_r, new_c]) == 1:
                        new_state['board'][new_r, new_c] *= 2
                        promotion_made = True
                    
                    # Reset move counter if capture or promotion
                    if capture_made or promotion_made:
                        new_state['no_progress_moves'] = 0
                    else:
                        new_state['no_progress_moves'] += 1
                    
                    # Update state repetitions
                    board_hash = new_state['board'].tobytes()
                    new_state['state_repetitions'][board_hash] += 1
                    
                    # Update board timesteps
                    if new_state['jump_again'] is None:
                        new_state['board_timesteps'].append(new_state['board'].copy())
                        if len(new_state['board_timesteps']) > self.history_timesteps:
                            new_state['board_timesteps'] = new_state['board_timesteps'][-self.history_timesteps:]
                    
                    return new_state
                    
        return new_state

    def check_win(self, state, player):
        opponent = self.get_opponent(player)

        # Check if opponent has any pieces left
        opponent_pieces = np.where((state['board'] == opponent) | (state['board'] == 2 * opponent))
        if len(opponent_pieces[0]) == 0:
            return True

        # Check if opponent has any valid moves left
        state_copy = state.copy()
        state_copy['jump_again'] = None
        valid_moves_opponent = self.get_valid_moves(state_copy, opponent)
        if not valid_moves_opponent.any():
            return True
        return False
    
    def check_draw(self, state):
        # Check for repetition of states
        if any(count >= self.repetition_limit for count in state['state_repetitions'].values()):
            return True

        # Check for no progress moves
        if state['no_progress_moves'] >= self.draw_move_limit:
            return True
        return False
    
    def get_value_and_terminated(self, state, player):
        # Check for terminal conditions
        if self.check_win(state, player):
            return 1, True
        if self.check_win(state, self.get_opponent(player)):
            return -1, True
        if self.check_draw(state):
            return 0, True
        return 0, False
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value

    def show_board(self, state, player=None):
        # Display the board in a human-readable format
        board = state['board']        
        max_col_width = len(str(self.col_count - 1))
        max_piece_width = max(len(str(np.max(board))), len(str(np.min(board))))
        cell_width = max(max_col_width, max_piece_width) + 1

        # Create a formatted string for the board
        col_numbers = " " + " " * (cell_width + 2) + " ".join(f"{i:>{cell_width}}" for i in range(self.col_count))
        border = " " * (cell_width + 1) + "-" * ((cell_width + 1) * self.col_count + 1)
        print(col_numbers)
        print(border)
        
        # Print each row
        for i, row in enumerate(board):
            row_str = f"{i:>{cell_width}} | " + " ".join(f"{val:>{cell_width}}" for val in row)
            print(row_str)
        
        print()
        
        # If a specific player is provided, show valid moves for that player
        if player:
            print(f"Player {player}")
            valid_moves = self.get_valid_moves(state, player)
            self.show_valid_moves(valid_moves)
    
    def show_valid_moves(self, valid_moves):
        print("Valid Moves")
        print("------------")

        # Display valid moves
        move_list = []
        for (r, c), moves in self.move_map.items():
            for new_r, new_c, action_idx, dr, dc in moves:
                if valid_moves[action_idx] == 1:
                    move_name = self.move_names[(dr, dc)]
                    move_list.append(f"{len(move_list)}: ({r}, {c}) -> ({new_r}, {new_c}) ({move_name}) [idx={action_idx}]")

        for move in move_list:
            print(move)