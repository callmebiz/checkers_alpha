#!/usr/bin/env python3
"""play.py - interactive runner for Checkers project

Supports three modes:
- pvp : human vs human (local, turn-based)
- pva : human vs agent
- ava : agent vs agent (useful for debugging)

Optional: load a model checkpoint for agent play. If no model is provided the agent will sample uniformly
from valid moves.
"""
import argparse
import time
import numpy as np

from core.game import Checkers
from core.state_encoder import StateEncoder
from core.mcts import MCTS
from core.model import ResNet
from training.checkpoints import load_model_from_checkpoint


def build_model_from_checkpoint(checkpoint, encoder, device):
    cfg = checkpoint.get('model_config', {})
    num_resBlocks = cfg.get('num_resBlocks', 3)
    num_hidden = cfg.get('num_hidden', 64)
    use_checkpoint = cfg.get('use_checkpoint', False)

    model = ResNet(
        input_channels=encoder.get_num_channels(),
        action_size=encoder.game.action_size,
        board_size=(encoder.game.row_count, encoder.game.col_count),
        num_resBlocks=num_resBlocks,
        num_hidden=num_hidden,
        device=device,
        use_checkpoint=use_checkpoint,
    )

    # Load weights if available
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    return model


def human_move(game, state, player):
    # Assumes caller already displayed the board and valid moves
    valid = game.get_valid_moves(state, player)
    if not valid.any():
        print("No valid moves available.")
        return None

    # Build sequence -> action_idx mapping in the same order as `show_valid_moves`
    seq_to_action = []
    for (r, c), moves in game.move_map.items():
        for new_r, new_c, action_idx, dr, dc in moves:
            if valid[action_idx] == 1:
                seq_to_action.append(action_idx)

    if len(seq_to_action) == 0:
        print("No valid moves available.")
        return None

    while True:
        try:
            raw = input(f"Player {player} - enter move number (0..{len(seq_to_action)-1}) or 'q' to quit: ")
            if raw.lower() in ('q', 'quit', 'exit'):
                return 'quit'
            seq = int(raw.strip())
            if 0 <= seq < len(seq_to_action):
                return seq_to_action[seq]
            else:
                print(f"Invalid move number. Enter a number between 0 and {len(seq_to_action)-1}.")
        except ValueError:
            print("Please enter a numeric move number.")


def agent_move(game, mcts, state, player, temperature=0.0):
    probs = mcts.search(state, player, temperature=temperature, add_noise=False)
    if np.sum(probs) == 0:
        return None
    # deterministic selection
    action = int(np.argmax(probs))
    return action


def uniform_agent_move(game, state, player):
    valid = game.get_valid_moves(state, player)
    idxs = np.where(valid == 1)[0]
    if len(idxs) == 0:
        return None
    return int(np.random.choice(idxs))


def build_seq_to_action(game, valid):
    seq_to_action = []
    for (r, c), moves in game.move_map.items():
        for new_r, new_c, action_idx, dr, dc in moves:
            if valid[action_idx] == 1:
                seq_to_action.append(action_idx)
    return seq_to_action


def display_state_info(game, state, player):
    board_hash = state['board'].tobytes()
    repeats = state.get('state_repetitions', {}).get(board_hash, 0)
    print((f"Repetitions: {repeats}/{game.repetition_limit} | "
           f"No-progress: {state.get('no_progress_moves',0)}/{game.draw_move_limit}"))
    game.show_board(state, player)


def run_game(mode, game, encoder, mcts=None, human_starts=1):
    state = game.get_initial_state()
    current = human_starts
    move_count = 0

    while True:
        # Check terminal from current player's perspective
        val, terminated = game.get_value_and_terminated(state, current)
        if terminated:
            print("Game ended")
            display_state_info(game, state, 1)
            outcome_p1, _ = game.get_value_and_terminated(state, 1)
            if outcome_p1 == 1:
                print("Player 1 (bottom) wins")
            elif outcome_p1 == -1:
                print("Player -1 (top) wins")
            else:
                print("Draw")
            return

        # Show state info and board before each move
        display_state_info(game, state, current)

        if mode == 'pvp':
            action = human_move(game, state, current)
            if action == 'quit':
                print('Quitting...')
                return

        elif mode == 'pva':
            if current == human_starts:
                action = human_move(game, state, current)
                if action == 'quit':
                    print('Quitting...')
                    return
            else:
                valid = game.get_valid_moves(state, current)
                seq_to_action = build_seq_to_action(game, valid)
                if mcts is not None:
                    action = agent_move(game, mcts, state, current)
                else:
                    action = uniform_agent_move(game, state, current)

                if action is None:
                    print(f"No valid moves for player {current}, finishing.")
                    return

                # Map action to the sequence number (always present for valid moves)
                seq_num = seq_to_action.index(action)
                print(f"Agent (P{current}) chose move {seq_num}")

        elif mode == 'ava':
            raw = input("Press Enter to continue (or 'q' to quit) (AI vs AI)...")
            if raw.lower() in ('q', 'quit', 'exit'):
                print('Quitting...')
                return
            valid = game.get_valid_moves(state, current)
            seq_to_action = build_seq_to_action(game, valid)
            if mcts is not None:
                action = agent_move(game, mcts, state, current)
            else:
                action = uniform_agent_move(game, state, current)

            if action is None:
                print(f"No valid moves for player {current}, finishing.")
                return

            seq_num = seq_to_action.index(action)
            print(f"Agent (P{current}) chose move {seq_num}")

        else:
            raise ValueError("Unknown mode")

        if action is None:
            print(f"No valid moves for player {current}, finishing.")
            return

        state = game.get_next_state(state, action, current)

        if not state.get('jump_again'):
            current = game.get_opponent(current)

        move_count += 1


def main():
    parser = argparse.ArgumentParser(description='Play Checkers (PVP/PVA/AVA)')
    parser.add_argument('--mode', choices=['pvp', 'pva', 'ava'], default='pvp')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint .pt to load model')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--mcts-searches', type=int, default=50)
    parser.add_argument('--human-starts', type=int, choices=[1, -1], default=1, help='Which player the human controls (for pva mode).')
    args = parser.parse_args()

    game = Checkers()
    encoder = StateEncoder(game)

    device = args.device

    model = None
    mcts = None

    if args.checkpoint:
        checkpoint = load_model_from_checkpoint(args.checkpoint, device)
        if checkpoint is None:
            print(f"Failed to load checkpoint: {args.checkpoint}. Agents will use uniform random policy.")
        else:
            model = build_model_from_checkpoint(checkpoint, encoder, device)
            cp_args = checkpoint.get('args', {}) if isinstance(checkpoint, dict) else {}
            mcts_args = {
                'num_searches': args.mcts_searches,
                'C': cp_args.get('C', 1.0),
                'dirichlet_epsilon': cp_args.get('dirichlet_epsilon', 0.25),
                'dirichlet_alpha': cp_args.get('dirichlet_alpha', 0.03)
            }
            mcts = MCTS(game, mcts_args, model, encoder)
            print(f"Loaded model from {args.checkpoint}; MCTS configured with {args.mcts_searches} searches.")

    elif args.mode in ('pva', 'ava'):
        # even without checkpoint, prepare a lightweight MCTS with uninitialized model replaced by uniform policy
        # Here we won't create a model; MCTS requires a model in this implementation, so leave mcts None
        print("No checkpoint provided: agents will act uniformly at random.")

    print(f"Starting mode: {args.mode}")
    run_game(args.mode, game, encoder, mcts=mcts, human_starts=args.human_starts)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""play.py - interactive runner for Checkers project

Supports three modes:
- pvp : human vs human (local, turn-based)
- pva : human vs agent
- ava : agent vs agent (useful for debugging)

Optional: load a model checkpoint for agent play. If no model is provided the agent will sample uniformly
from valid moves.
"""
import argparse
import time
import numpy as np

from core.game import Checkers
from core.state_encoder import StateEncoder
from core.mcts import MCTS
from core.model import ResNet
from training.checkpoints import load_model_from_checkpoint


def build_model_from_checkpoint(checkpoint, encoder, device):
    cfg = checkpoint.get('model_config', {})
    num_resBlocks = cfg.get('num_resBlocks', 3)
    num_hidden = cfg.get('num_hidden', 64)
    use_checkpoint = cfg.get('use_checkpoint', False)

    model = ResNet(
        input_channels=encoder.get_num_channels(),
        action_size=encoder.game.action_size,
        board_size=(encoder.game.row_count, encoder.game.col_count),
        num_resBlocks=num_resBlocks,
        num_hidden=num_hidden,
        device=device,
        use_checkpoint=use_checkpoint,
    )

    # Load weights if available
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    return model


def human_move(game, state, player):
    # Assumes caller already displayed the board and valid moves
    valid = game.get_valid_moves(state, player)
    if not valid.any():
        print("No valid moves available.")
        return None

    # Build sequence -> action_idx mapping in the same order as `show_valid_moves`
    seq_to_action = []
    for (r, c), moves in game.move_map.items():
        for new_r, new_c, action_idx, dr, dc in moves:
            if valid[action_idx] == 1:
                seq_to_action.append(action_idx)

    if len(seq_to_action) == 0:
        print("No valid moves available.")
        return None

    while True:
        try:
            raw = input(f"Player {player} - enter move number (0..{len(seq_to_action)-1}) or 'q' to quit: ")
            if raw.lower() in ('q', 'quit', 'exit'):
                return 'quit'
            seq = int(raw.strip())
            if 0 <= seq < len(seq_to_action):
                return seq_to_action[seq]
            else:
                print(f"Invalid move number. Enter a number between 0 and {len(seq_to_action)-1}.")
        except ValueError:
            print("Please enter a numeric move number.")


def agent_move(game, mcts, model, encoder, state, player, temperature=0.0, device='cpu'):
    """Return (action_idx, probs) using MCTS if available, otherwise the model's policy.

    Args:
        game: Checkers game instance
        mcts: optional MCTS instance
        model: optional ResNet model
        encoder: StateEncoder instance
        state: current state dict
        player: current player (1 or -1)
    """
    probs = None
    # Prefer MCTS when available
    if mcts is not None:
        probs = mcts.search(state, player, temperature=temperature, add_noise=False)
    elif model is not None:
        # Use the model directly to get policy logits -> probs
        encoded = encoder.encode_state(state, player)
        import torch
        x = torch.from_numpy(encoded).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            logits, _ = model(x)
        logits = logits.cpu().numpy().flatten()
        # Mask invalid moves
        valid = game.get_valid_moves(state, player)
        logits = np.where(valid == 1, logits, -1e9)
        # Softmax (stable)
        exps = np.exp(logits - np.max(logits))
        denom = np.sum(exps)
        if denom == 0:
            probs = np.zeros_like(exps)
        else:
            probs = exps / denom

    if probs is None or np.sum(probs) == 0:
        return None, None
    action = int(np.argmax(probs))
    return action, probs


def uniform_agent_move(game, state, player):
    valid = game.get_valid_moves(state, player)
    idxs = np.where(valid == 1)[0]
    if len(idxs) == 0:
        return None
    return int(np.random.choice(idxs))


def build_seq_to_action(game, valid):
    seq_to_action = []
    for (r, c), moves in game.move_map.items():
        for new_r, new_c, action_idx, dr, dc in moves:
            if valid[action_idx] == 1:
                seq_to_action.append(action_idx)
    return seq_to_action


def format_action_desc(game, action_idx):
    """Return a human-friendly description for an action index."""
    for (r, c), moves in game.move_map.items():
        for new_r, new_c, a_idx, dr, dc in moves:
            if a_idx == action_idx:
                # move_names maps (dr,dc) -> human name if available
                direction = getattr(game, 'move_names', {}).get((dr, dc), '')
                return f"({r},{c}) -> ({new_r},{new_c}) ({direction}) [idx={a_idx}]"
    return f"[idx={action_idx}]"


def display_state_info(game, state, player):
    board_hash = state['board'].tobytes()
    repeats = state.get('state_repetitions', {}).get(board_hash, 0)
    print((f"Repetitions: {repeats}/{game.repetition_limit} | "
           f"No-progress: {state.get('no_progress_moves',0)}/{game.draw_move_limit}"))
    game.show_board(state, player)


def run_game(mode, game, encoder, model=None, mcts=None, human_starts=1, device='cpu'):
    state = game.get_initial_state()
    current = human_starts
    move_count = 0

    while True:
        # Check terminal from current player's perspective
        val, terminated = game.get_value_and_terminated(state, current)
        if terminated:
            print("Game ended")
            display_state_info(game, state, 1)
            outcome_p1, _ = game.get_value_and_terminated(state, 1)
            if outcome_p1 == 1:
                print("Player 1 (bottom) wins")
            elif outcome_p1 == -1:
                print("Player -1 (top) wins")
            else:
                print("Draw")
            return

        # Show state info and board before each move
        display_state_info(game, state, current)

        if mode == 'pvp':
            action = human_move(game, state, current)
            if action == 'quit':
                print('Quitting...')
                return

        elif mode == 'pva':
            if current == human_starts:
                action = human_move(game, state, current)
                if action == 'quit':
                    print('Quitting...')
                    return
            else:
                valid = game.get_valid_moves(state, current)
                seq_to_action = build_seq_to_action(game, valid)
                # Get policy probs (from MCTS or model) if available
                action = None
                probs = None
                if mcts is not None or model is not None:
                    action, probs = agent_move(game, mcts, model, encoder, state, current, device=device)
                else:
                    # no policy available
                    pass

                # Print full list of valid moves with probabilities when available
                for i, a_idx in enumerate(seq_to_action):
                    desc = format_action_desc(game, a_idx)
                    p = None
                    if probs is not None:
                        try:
                            p = float(probs[a_idx])
                        except Exception:
                            p = None
                    prob_str = f" [prob: {p:.3f}]" if p is not None else ""
                    print(f"{i}: {desc}{prob_str}")

                if action is None:
                    # Fall back to uniform if no policy provided
                    action = uniform_agent_move(game, state, current)

                if action is None:
                    print(f"No valid moves for player {current}, finishing.")
                    return

                # Map action to the sequence number (always present for valid moves)
                seq_num = seq_to_action.index(action)
                chosen_desc = format_action_desc(game, action)
                chosen_prob = float(probs[action]) if (probs is not None and action is not None) else None
                prob_str = f" [prob: {chosen_prob:.3f}]" if chosen_prob is not None else ""
                print(f"Agent (P{current}) chose move {seq_num}: {chosen_desc}{prob_str}")

        elif mode == 'ava':
            raw = input("Press Enter to continue (or 'q' to quit) (AI vs AI)...")
            if raw.lower() in ('q', 'quit', 'exit'):
                print('Quitting...')
                return
            valid = game.get_valid_moves(state, current)
            seq_to_action = build_seq_to_action(game, valid)
            probs = None
            action = None
            if mcts is not None or model is not None:
                action, probs = agent_move(game, mcts, model, encoder, state, current, device=device)

            # Print full list of valid moves with probabilities when available
            for i, a_idx in enumerate(seq_to_action):
                desc = format_action_desc(game, a_idx)
                p = None
                if probs is not None:
                    try:
                        p = float(probs[a_idx])
                    except Exception:
                        p = None
                prob_str = f" [prob: {p:.3f}]" if p is not None else ""
                print(f"{i}: {desc}{prob_str}")

            if action is None:
                action = uniform_agent_move(game, state, current)

            if action is None:
                print(f"No valid moves for player {current}, finishing.")
                return

            seq_num = seq_to_action.index(action)
            desc = format_action_desc(game, action)
            chosen_prob = float(probs[action]) if (probs is not None and action is not None) else None
            prob_str = f" [prob: {chosen_prob:.3f}]" if chosen_prob is not None else ""
            print(f"Agent (P{current}) chose move {seq_num}: {desc}{prob_str}")

        else:
            raise ValueError("Unknown mode")

        if action is None:
            print(f"No valid moves for player {current}, finishing.")
            return

        state = game.get_next_state(state, action, current)

        if not state.get('jump_again'):
            current = game.get_opponent(current)

        move_count += 1


def main():
    parser = argparse.ArgumentParser(description='Play Checkers (PVP/PVA/AVA)')
    parser.add_argument('--mode', choices=['pvp', 'pva', 'ava'], default='pvp')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint .pt to load model')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--mcts-searches', type=int, default=50)
    parser.add_argument('--human-starts', type=int, choices=[1, -1], default=1, help='Which player the human controls (for pva mode).')
    args = parser.parse_args()

    game = Checkers()
    encoder = StateEncoder(game)

    device = args.device

    model = None
    mcts = None

    if args.checkpoint:
        checkpoint = load_model_from_checkpoint(args.checkpoint, device)
        if checkpoint is None:
            print(f"Failed to load checkpoint: {args.checkpoint}. Agents will use uniform random policy.")
        else:
            model = build_model_from_checkpoint(checkpoint, encoder, device)
            # Use the model directly for policy decisions (no MCTS)
            print(f"Loaded model from {args.checkpoint}; agent will use the model policy (no MCTS).")

    elif args.mode in ('pva', 'ava'):
        # even without checkpoint, prepare a lightweight MCTS with uninitialized model replaced by uniform policy
        # Here we won't create a model; MCTS requires a model in this implementation, so leave mcts None
        print("No checkpoint provided: agents will act uniformly at random.")

    print(f"Starting mode: {args.mode}")
    run_game(args.mode, game, encoder, model=model, mcts=None, human_starts=args.human_starts, device=device)


if __name__ == '__main__':
    main()
