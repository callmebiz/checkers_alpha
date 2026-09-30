import random

import numpy as np
import pytest
import torch

from core.encoding import (
    FLIP_ACTION,
    NUM_PLANES,
    canonical_actions,
    encode,
    features,
    legal_mask,
    reverse_bits,
)
from core.game import (
    A_FROM,
    A_TO,
    ACTION_SIZE,
    BLACK,
    ON_BOARD_ACTIONS,
    WHITE,
    Position,
)
from core.inference import NetEvaluator, UniformEvaluator
from core.mcts import SearchTree, run_search
from core.model import CheckersNet, ModelConfig, load_model, save_model
from core.players import (
    AlphaBetaPlayer,
    MCTSPlayer,
    Player,
    RandomPlayer,
    make_player,
)

TINY = ModelConfig(blocks=1, channels=8, value_hidden=16)


def move_of(action: int) -> tuple[int, int]:
    """The (from, to) PDN squares of an action, for readable asserts."""
    return A_FROM[action] + 1, A_TO[action] + 1


def rotated(pos: Position) -> Position:
    """The same position for the other side.

    Rotate 180 degrees and swap colours.

    Rebuilt through FEN so it passes the same validation as any loaded position.
    """
    swapped = Position(
        reverse_bits(pos.white),
        reverse_bits(pos.black),
        reverse_bits(pos.kings),
        1 - pos.turn,
    )
    return Position.from_fen(swapped.to_fen())


def random_positions(n_games: int, seed: int = 0) -> list[Position]:
    """Every position of ``n_games`` random games."""
    rng = random.Random(seed)
    out = []
    for _ in range(n_games):
        pos = Position.initial()
        while not pos.is_terminal:
            out.append(pos)
            pos = pos.play(rng.choice(pos.legal_actions()))
    return out


# Black to move. 21-25 blocks White's only man: an immediate win.
WIN_IN_ONE = 'B:W29:B21,22'
# Black to move. 10-15 lets 19 capture Black's only man; 10-14 is safe.
AVOID_LOSS = 'B:W19:B10'


# -- encoding ------------------------------------------------------------------


def test_rotation_maps_dark_squares_to_dark_squares():
    """Rotation keeps hops on the board and reverses them exactly."""
    assert (FLIP_ACTION[FLIP_ACTION] == np.arange(ACTION_SIZE)).all()
    for a in ON_BOARD_ACTIONS:
        b = int(FLIP_ACTION[a])
        assert A_TO[b] >= 0, (
            'an on-board hop must stay on the board after rotation'
        )
        assert (A_FROM[b], A_TO[b]) == (31 - A_FROM[a], 31 - A_TO[a])


def test_rotated_positions_are_legal_and_equivalent():
    """Rotated positions are valid and equivalent to the originals.

    Every position from random games, seen from the other side, is a valid
    position with the same moves and the same network input.
    """
    positions = [p for p in random_positions(20) if p.jump_sq < 0]
    for pos in positions:
        other = rotated(pos)  # from_fen raises on anything invalid
        assert len(other.legal_actions()) == len(pos.legal_actions())
        assert sorted(canonical_actions(other)) == sorted(
            canonical_actions(pos)
        )
    # FEN carries no draw counters, so compare the piece planes only
    others = [rotated(p) for p in positions]
    assert np.array_equal(encode(positions)[:, :5], encode(others)[:, :5])


def test_planes_show_jump_piece_and_counters():
    """The jump plane marks the piece mid multi-jump, in the rotated frame."""
    pos = Position.from_fen('B:W6,15:B1')
    mid_jump = pos.play(pos.legal_actions()[0])
    bits, _ = features([mid_jump])
    assert bits[0, 4] == 1 << (31 - mid_jump.jump_sq)  # Black's view is rotated
    x = encode([mid_jump])
    assert x.shape == (1, NUM_PLANES, 8, 8)
    assert x[0, 4].sum() == 1 and (x[0, -1] == 1).all()


def test_legal_mask_matches_legal_actions():
    """The legal mask matches legal_actions and skips off-board hops."""
    positions = random_positions(3)
    mask = legal_mask(positions)
    assert (mask.sum(1) == [len(p.legal_actions()) for p in positions]).all()
    assert not mask[:, [a for a in range(ACTION_SIZE) if A_TO[a] < 0]].any()


# -- model ---------------------------------------------------------------------


def test_model_shapes_and_checkpoint_round_trip(tmp_path):
    """Output shapes are right and a saved model reloads identically."""
    model = CheckersNet(TINY).eval()
    x = torch.from_numpy(
        encode([Position.initial(), Position.from_fen(WIN_IN_ONE)])
    )
    policy, wdl = model(x)
    assert policy.shape == (2, ACTION_SIZE) and wdl.shape == (2, 3)
    save_model(tmp_path / 'm.pt', model, iteration=3)
    loaded, ckpt = load_model(tmp_path / 'm.pt')
    assert ckpt['iteration'] == 3
    assert torch.allclose(loaded(x)[0], policy)


def test_net_evaluator_priors_cover_legal_moves():
    """Priors cover exactly the legal moves and sum to one."""
    positions = [Position.initial(), Position.from_fen('W:WK3,18,22:B9,K27')]
    priors, values, wdl = NetEvaluator(CheckersNet(TINY)).evaluate(positions)
    for p, pos in zip(priors, positions, strict=True):
        assert len(p) == len(pos.legal_actions()) and abs(p.sum() - 1) < 1e-5
    assert np.allclose(wdl.sum(1), 1, atol=1e-5) and (np.abs(values) <= 1).all()


# -- search --------------------------------------------------------------------


@pytest.mark.parametrize('leaves_per_tree', [1, 8])
def test_mcts_takes_immediate_win(leaves_per_tree):
    """Search finds a one-move win, with or without batched leaves."""
    tree = SearchTree(
        Position.from_fen(WIN_IN_ONE), rng=np.random.default_rng(0)
    )
    run_search([tree], UniformEvaluator(), 200, leaves_per_tree)
    assert move_of(tree.best_action()) == (21, 25)
    assert tree.sims == 201
    stats = {move_of(s['action']): s for s in tree.move_stats()}
    assert stats[(21, 25)]['q'] == 1.0


@pytest.mark.parametrize('leaves_per_tree', [1, 8])
def test_mcts_avoids_immediate_loss(leaves_per_tree):
    """Search avoids a move that loses at once."""
    tree = SearchTree(
        Position.from_fen(AVOID_LOSS), rng=np.random.default_rng(0)
    )
    run_search([tree], UniformEvaluator(), 200, leaves_per_tree)
    assert move_of(tree.best_action()) == (10, 14)
    stats = {move_of(s['action']): s for s in tree.move_stats()}
    assert stats[(10, 15)]['q'] < -0.5


def test_mcts_value_sign_through_multi_jump():
    # Black's double jump 1x10x19 wins. The continuation hop keeps Black to
    # move, so the backed-up value must not flip between the two hops.
    """Values keep their sign between one player's multi-jump hops."""
    tree = SearchTree(Position.from_fen('B:W6,15:B1'))
    run_search([tree], UniformEvaluator(), 100)
    assert tree.root.q() > 0.9


def test_many_trees_in_one_batch():
    """Trees searched in one batch each get their own answer."""
    trees = [
        SearchTree(Position.from_fen(f)) for f in (WIN_IN_ONE, AVOID_LOSS)
    ] * 3
    run_search(trees, UniformEvaluator(), 100)
    assert [move_of(t.best_action()) for t in trees] == [(21, 25), (10, 14)] * 3


def test_tree_reuse_keeps_visits():
    """Advancing keeps the child subtree; re-rooting finds known nodes."""
    tree = SearchTree(Position.initial())
    run_search([tree], UniformEvaluator(), 300)
    a = tree.best_action()
    child = tree.root.children[tree.root.actions.index(a)]
    assert child is not None
    tree.advance(a)
    assert tree.root.visits == child.visits > 0
    assert tree.reroot(tree.root.pos) is True


def test_root_noise_only_on_legal_moves():
    """Root noise covers only the legal moves and changes the priors."""
    tree = SearchTree(
        Position.initial(), add_noise=True, rng=np.random.default_rng(1)
    )
    run_search([tree], UniformEvaluator(), 1)
    assert len(tree.root.P) == 7 and abs(sum(tree.root.P) - 1) < 1e-9
    assert not np.allclose(tree.root.P, tree.root.P_raw)


# -- players -------------------------------------------------------------------


@pytest.mark.parametrize(
    'fen, move', [(WIN_IN_ONE, (21, 25)), (AVOID_LOSS, (10, 14))]
)
def test_players_solve_tactics(fen, move):
    """Alpha-beta and MCTS both find the tactical moves."""
    pos = Position.from_fen(fen)
    assert move_of(AlphaBetaPlayer(depth=2, seed=0).choose(pos)) == move
    assert (
        move_of(MCTSPlayer(UniformEvaluator(), sims=200, seed=0).choose(pos))
        == move
    )


def play(black: Player, white: Player, max_plies: int = 400) -> str:
    """Play a game; return the winner's colour or 'draw'."""
    pos = Position.initial()
    for _ in range(max_plies):
        if pos.is_terminal:
            break
        pos = pos.play((white if pos.turn == WHITE else black).choose(pos))
    if pos.result() == -1:
        return 'white' if pos.turn == BLACK else 'black'
    return 'draw'


def test_alpha_beta_beats_random():
    """Alpha-beta at depth 4 beats random with either colour."""
    results = [
        play(AlphaBetaPlayer(4, seed=i), RandomPlayer(i)) for i in range(2)
    ]
    results += [
        play(RandomPlayer(i), AlphaBetaPlayer(4, seed=i)) for i in range(2)
    ]
    assert results == ['black', 'black', 'white', 'white']


def test_make_player_specs(tmp_path):
    """Player spec strings build the right kind of player."""
    path = tmp_path / 'net.pt'
    save_model(path, CheckersNet(TINY))
    ab = make_player('ab:3')
    uniform = make_player('mcts:uniform:50')
    net = make_player(f'mcts:{path}:64')
    assert isinstance(ab, AlphaBetaPlayer) and ab.depth == 3
    assert isinstance(uniform, MCTSPlayer) and uniform.sims == 50
    assert isinstance(net, MCTSPlayer) and net.sims == 64
    start = Position.initial()
    assert make_player(f'policy:{path}').choose(start) in start.legal_actions()
    with pytest.raises(ValueError):
        make_player('nope')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_cuda_graphs_match_eager_and_follow_weight_updates():
    """Graph replays match plain execution and see weight updates."""
    torch.manual_seed(0)
    model = CheckersNet(TINY).cuda()
    positions = random_positions(2)[:37]  # pads up to a 64-row graph
    graphs = NetEvaluator(model, use_amp=False)
    eager = NetEvaluator(model, use_amp=False, use_cuda_graphs=False)
    for batch in (positions[:1], positions[:5], positions):
        got, want = graphs.evaluate(batch), eager.evaluate(batch)
        assert np.allclose(got.values, want.values, atol=1e-5)
        for p, q in zip(got.priors, want.priors, strict=True):
            assert np.allclose(p, q, atol=1e-5)

    # a training step changes the weights in place; replays must see it
    before = graphs.evaluate(positions).values.copy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    model.train()
    policy, wdl = model(torch.from_numpy(encode(positions)).cuda())
    (policy.square().mean() + wdl.square().mean()).backward()
    optimizer.step()
    model.eval()
    after = graphs.evaluate(positions).values
    assert not np.allclose(before, after)
    assert np.allclose(after, eager.evaluate(positions).values, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_evaluators_with_cuda_graphs_do_not_leak_gpu_memory():
    """Creating and discarding evaluators (as each arena evaluation does for
    the previous checkpoint) must not grow GPU memory."""
    import gc

    positions = [Position.initial()] * 5

    def cycle() -> int:
        """Create, use and discard an evaluator; return GPU memory in use."""
        model = CheckersNet(TINY).cuda()
        NetEvaluator(model).evaluate(positions)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated()

    cycle()  # first use allocates one-time workspaces
    baseline = cycle()
    for _ in range(3):
        assert cycle() <= baseline
