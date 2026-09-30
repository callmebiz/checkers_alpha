import time
import numpy as np
import torch
from core.game import Checkers
from core.state_encoder import StateEncoder
from core.model import ResNet
from core.mcts import MCTS


def main():
    # Config
    runs = 200
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Build small game and encoder
    game = Checkers(row_count=8, col_count=8, buffer_count=2, draw_move_limit=50, repetition_limit=3, history_timesteps=4)
    encoder = StateEncoder(game)

    # Build model
    model = ResNet(input_channels=encoder.get_num_channels(),
                   action_size=game.action_size,
                   board_size=(game.row_count, game.col_count),
                   num_resBlocks=3,
                   num_hidden=64,
                   device=device,
                   use_checkpoint=False)
    model.eval()

    # MCTS for reference
    mcts = MCTS(game, {'C':1.0, 'num_searches': 10, 'dirichlet_epsilon':0.25, 'dirichlet_alpha':0.3}, model, encoder)

    # Prepare a sample state
    state = game.get_initial_state()
    player = 1

    # Warmup
    for _ in range(5):
        enc = encoder.encode_state(state, player)
        t = torch.tensor(enc, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _ = model(t)

    timings = {'encode':0.0, 'to_tensor':0.0, 'forward':0.0, 'post':0.0}

    for i in range(runs):
        # encode
        t0 = time.time()
        enc = encoder.encode_state(state, player)
        timings['encode'] += time.time() - t0

        # tensor conversion (numpy->torch on device)
        t0 = time.time()
        t = torch.tensor(enc, dtype=torch.float32, device=device).unsqueeze(0)
        timings['to_tensor'] += time.time() - t0

        # forward
        t0 = time.time()
        with torch.no_grad():
            out_policy, out_value = model(t)
        timings['forward'] += time.time() - t0

        # postprocessing: softmax, mask, numpy copy
        t0 = time.time()
        probs = torch.softmax(out_policy, dim=1).squeeze(0).cpu().numpy()
        valid_moves = game.get_valid_moves(state, player)
        probs *= valid_moves
        if probs.sum() > 0:
            probs /= probs.sum()
        timings['post'] += time.time() - t0

    print(f"Device: {device}")
    print(f"Runs: {runs}")
    for k, v in timings.items():
        print(f"{k}: total {v:.4f}s avg {v/runs*1000:.3f}ms")

    # Run a few full MCTS.search calls (with num_searches small) to measure end-to-end
    mcts_iters = 100
    total_mcts = 0.0
    total_inf = 0.0
    for _ in range(mcts_iters):
        t0 = time.time()
        _ = mcts.search(state, player)
        total_mcts += time.time() - t0
        total_inf += mcts.get_last_search_inference_time()

    print(f"MCTS runs: {mcts_iters} total_time {total_mcts:.4f}s avg {total_mcts/mcts_iters*1000:.3f}ms")
    print(f"MCTS reported inference total {total_inf:.4f}s avg {total_inf/mcts_iters*1000:.3f}ms")

    # Print cache statistics if available
    try:
        stats = mcts.get_cache_stats()
        print(f"MCTS cache: hits={stats['hits']} lookups={stats['lookups']} hit_rate={stats['hit_rate']:.3f}")
    except Exception:
        print("MCTS cache stats not available")


if __name__ == '__main__':
    main()
