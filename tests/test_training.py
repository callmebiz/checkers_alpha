import json
from typing import cast

import numpy as np
import pytest
import torch
from torch import nn

from core.encoding import FLIP_ACTION, legal_mask
from core.game import ACTION_SIZE, Position
from core.inference import NetEvaluator
from core.mcts import SearchConfig
from core.model import CheckersNet, ModelConfig
from core.players import MCTSPlayer, make_player
from training.config import (
    RunConfig,
    SelfPlayConfig,
    apply_overrides,
    changed_config,
)
from training.replay_buffer import DRAW, LOSS, WIN, ReplayBuffer, Samples
from training.run_store import RunStore
from training.self_play import SelfPlay
from training.trainer import Trainer, compute_loss

TINY_MODEL = {'blocks': 1, 'channels': 8, 'value_hidden': 16}


def tiny_config() -> RunConfig:
    """A tiny CPU config for fast end-to-end runs (arena off)."""
    return RunConfig.from_dict(
        {
            'model': TINY_MODEL,
            'self_play': {
                'games_per_iteration': 4,
                'concurrent_games': 4,
                'sims': 8,
                'fast_sims': 4,
                'full_search_prob': 0.5,
                'max_plies': 60,
                'saved_games': 2,
            },
            'train': {
                'iterations': 2,
                'batch_size': 16,
                'min_buffer': 1,
                'warmup_steps': 2,
                'buffer_size': 1000,
                'checkpoint_every': 1,
            },
            'arena': {'every': 0},  # covered by test_evaluation.py
            'probes': {'sims': 8},
            'seed': 0,
            'device': 'cpu',
        }
    )


def random_samples(n: int, rng: np.random.Generator) -> Samples:
    """``n`` samples with random bitboards and outcomes."""
    samples = Samples.empty(n)
    samples.bits = rng.integers(0, 2**32, (n, 5), dtype=np.uint32)
    samples.wdl = rng.integers(0, 3, n, dtype=np.uint8)
    return samples


# -- config --------------------------------------------------------------------


def test_config_round_trip_and_overrides(tmp_path):
    """Overrides apply, defaults fill in, and save/load round-trips."""
    data = {'model': {'blocks': 3}}
    apply_overrides(data, ['train.lr=0.01', 'self_play.sims=64', 'seed=7'])
    config = RunConfig.from_dict(data)
    assert config.model.blocks == 3 and config.model.channels == 64
    assert config.train.lr == 0.01 and config.self_play.sims == 64
    assert config.seed == 7
    config.save(tmp_path / 'c.json')
    assert RunConfig.load(tmp_path / 'c.json') == config


@pytest.mark.parametrize(
    'data', [{'modle': {}}, {'train': {'learning_rate': 1}}, {'x': 1}]
)
def test_config_rejects_unknown_keys(data):
    """Misspelled sections or keys raise instead of being ignored."""
    with pytest.raises(ValueError):
        RunConfig.from_dict(data)


# -- replay buffer -------------------------------------------------------------


def test_replay_buffer_keeps_newest_and_round_trips(tmp_path):
    """The buffer keeps the newest rows and reloads at a new capacity."""
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(10)
    parts = [random_samples(n, rng) for n in (4, 4, 5)]
    for part in parts:
        buffer.add(part)
    assert len(buffer) == 10 and buffer.total_added == 13
    newest = Samples.concat(parts).take(slice(3, 13))
    assert np.array_equal(buffer.ordered().bits, newest.bits)

    buffer.save(tmp_path / 'b.npz')
    loaded = ReplayBuffer.load(tmp_path / 'b.npz', capacity=6)
    assert len(loaded) == 6 and loaded.total_added == 13
    assert np.array_equal(loaded.ordered().bits, newest.bits[4:])
    assert len(buffer.sample(32, rng)) == 32


def test_buffer_add_larger_than_capacity():
    """Adding more than the capacity keeps only the newest rows."""
    buffer = ReplayBuffer(5)
    buffer.add(random_samples(12, np.random.default_rng(1)))
    assert len(buffer) == 5 and buffer.total_added == 12


# -- self-play -----------------------------------------------------------------


def test_self_play_samples_are_consistent():
    """Samples have valid policies, legal masks and outcomes."""
    config = SelfPlayConfig(
        concurrent_games=3,
        sims=8,
        fast_sims=4,
        full_search_prob=0.5,
        max_plies=80,
    )
    model = CheckersNet(ModelConfig(**TINY_MODEL))
    self_play = SelfPlay(
        NetEvaluator(model),
        config,
        SearchConfig(),
        Position.initial().rules,
        np.random.default_rng(0),
    )
    records, stats = self_play.play(5)
    assert stats.games == 5 == len(records)
    assert stats.black_wins + stats.white_wins + stats.draws == 5
    assert stats.samples == sum(len(r.samples) for r in records) > 0
    # games chosen for logging may still be in progress (they carry over)
    assert 1 <= sum(r.logged for r in records) <= config.saved_games

    for record in records:
        assert record.plies <= 80
        # only games chosen for saving keep a move log
        assert len(record.moves) == (record.plies if record.logged else 0)
        s = record.samples
        if not len(s):
            continue
        mask = s.legal_mask()
        policy = s.policy.astype(np.float32)
        assert np.allclose(policy.sum(1), 1, atol=1e-2)
        assert not policy[~mask].any(), 'visits on an illegal action'
        if record.winner is None:
            assert (s.wdl == DRAW).all()
        else:
            assert set(s.wdl.tolist()) <= {WIN, LOSS}
        json.dumps(record.to_json())  # serialisable for the UI


def test_samples_match_encoding_of_replayed_positions():
    """Replaying a saved game reproduces the recorded legal masks."""
    config = SelfPlayConfig(
        concurrent_games=1,
        sims=8,
        fast_sims=4,
        full_search_prob=1.0,
        max_plies=30,
    )
    model = CheckersNet(ModelConfig(**TINY_MODEL))
    (record,), _ = SelfPlay(
        NetEvaluator(model),
        config,
        SearchConfig(),
        Position.initial().rules,
        np.random.default_rng(3),
    ).play(1)
    pos, searched = Position.initial(), []
    for move in record.moves:
        if move['kind'] == 'full':
            searched.append(pos)
        pos = pos.play(move['action'])
    assert np.array_equal(
        np.packbits(legal_mask(searched), axis=1), record.samples.legal
    )


# -- loss ----------------------------------------------------------------------


class FixedLogits(nn.Module):
    """Stands in for the network: returns fixed outputs."""

    def __init__(self, policy: torch.Tensor, wdl: torch.Tensor) -> None:
        """Remember the outputs to return."""
        super().__init__()
        self.policy, self.wdl = policy, wdl

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the fixed outputs, ignoring the input."""
        return self.policy, self.wdl


def test_policy_loss_ignores_illegal_logits():
    """The policy loss ignores illegal logits and matches a hand count."""
    torch.manual_seed(0)
    mask = torch.zeros(2, ACTION_SIZE, dtype=torch.bool)
    mask[:, [3, 40, 41]] = True
    target = torch.zeros(2, ACTION_SIZE)
    target[:, [3, 40]] = 0.5
    wdl_logits, wdl = torch.randn(2, 3), torch.tensor([0, 2])
    logits = torch.randn(2, ACTION_SIZE)

    def loss(policy: torch.Tensor) -> dict[str, float]:
        """Loss statistics for the given policy logits."""
        model = cast(CheckersNet, FixedLogits(policy, wdl_logits))
        _, stats = compute_loss(model, torch.zeros(2), target, mask, wdl, 1.0)
        return {k: float(v) for k, v in stats.items()}

    base = loss(logits)
    shifted = logits.clone()
    shifted[~mask] += 100.0  # illegal logits must not matter
    assert loss(shifted)['policy_loss'] == pytest.approx(base['policy_loss'])

    legal_logp = torch.log_softmax(logits[:, [3, 40, 41]], dim=1)
    expected = -(0.5 * legal_logp[:, 0] + 0.5 * legal_logp[:, 1]).mean()
    assert base['policy_loss'] == pytest.approx(expected.item(), rel=1e-5)
    assert base['policy_kl'] >= -1e-6


# -- end to end ----------------------------------------------------------------


def test_train_resume_and_play_from_checkpoint(tmp_path):
    """Train, resume past the end, then play with the latest checkpoint."""
    run = tmp_path / 'run'
    Trainer.new(tiny_config(), run).run()
    store = RunStore.open(run)
    assert [r['iteration'] for r in store.read_metrics()] == [0, 1]
    assert [p.name for p in store.list_checkpoints()] == [
        'iter_0000.pt',
        'iter_0001.pt',
    ]
    assert json.loads(store.status_path.read_text())['phase'] == 'finished'
    # games chosen for saving may finish in a later iteration, so only the
    # total is predictable: 2 started per iteration, most finished by now
    saved = [
        line
        for f in sorted((run / 'games').glob('iter_*.jsonl'))
        for line in f.read_text().splitlines()
    ]
    assert 2 <= len(saved) <= 4

    with pytest.raises(FileExistsError):
        Trainer.new(tiny_config(), run)

    trainer = Trainer.resume(run)
    assert trainer.iteration == 2 and len(trainer.buffer) > 0
    trainer.config.train.iterations = 3
    trainer.run()
    rows = store.read_metrics()
    assert [r['iteration'] for r in rows] == [0, 1, 2]
    assert rows[2]['step'] > rows[1]['step'] > 0

    player = make_player(f'mcts:{store.latest_path}:16')
    assert isinstance(player, MCTSPlayer)
    start = Position.initial()
    assert player.choose(start) in start.legal_actions()


def test_flip_table_is_used_consistently_in_targets():
    # Black-to-move targets are stored in the rotated (network) frame.
    """Black-to-move targets are stored in the rotated network frame."""
    pos = Position.initial()
    assert pos.turn == 0
    mask = legal_mask([pos])[0]
    rotated = {int(FLIP_ACTION[a]) for a in pos.legal_actions()}
    assert set(np.flatnonzero(mask).tolist()) == rotated


def test_auto_backend_trial_picks_the_faster(tmp_path, monkeypatch):
    """'auto' warms up on graphs, tries TensorRT then graphs, and keeps
    the faster (TensorRT needs a 5% margin)."""
    trainer = Trainer.new(tiny_config(), tmp_path / 'run')
    evaluator = trainer.evaluator
    settled: list[str] = []
    monkeypatch.setattr(
        evaluator,
        'candidates',
        lambda: [] if settled else ['graphs', 'tensorrt'],
    )
    monkeypatch.setattr(
        evaluator, 'settle', lambda backend, note: settled.append(backend)
    )
    assert trainer._try_backend() is None  # warm-up, not measured
    assert evaluator.backend == 'graphs'
    assert trainer._try_backend() == 'tensorrt'
    trainer._trial_result('tensorrt', 81_000.0)
    assert trainer._try_backend() == 'graphs'
    trainer._trial_result('graphs', 50_000.0)
    assert settled == ['tensorrt']

    # a small TensorRT lead isn't enough
    trainer2 = Trainer.new(tiny_config(), tmp_path / 'run2')
    monkeypatch.setattr(
        trainer2.evaluator,
        'settle',
        lambda backend, note: settled.append(backend),
    )
    trainer2._trial_result('tensorrt', 102_000.0)
    trainer2._trial_result('graphs', 100_000.0)
    assert settled[-1] == 'graphs'


def test_time_budget_stops_and_survives_resume(tmp_path):
    """A time budget ends the run however many iterations are left, the
    learning rate follows the budget, and resuming keeps the time used."""
    config = tiny_config()
    config.train.iterations = 1000
    config.train.time_budget_hours = 1e-9  # used up by the first iteration
    run = tmp_path / 'run'
    trainer = Trainer.new(config, run)
    trainer.run()
    assert trainer.iteration == 1  # the first iteration always runs
    store = RunStore.open(run)
    rows = store.read_metrics()
    assert len(rows) == 1 and rows[0]['elapsed_hours'] >= 0
    assert json.loads(store.status_path.read_text())['phase'] == 'finished'
    assert store.list_checkpoints()  # the last iteration was saved

    resumed = Trainer.resume(run)
    assert resumed.elapsed() > 0
    resumed.run()  # out of time already: nothing more to do
    assert resumed.iteration == 1 and len(store.read_metrics()) == 1

    # half the budget gone = half way down the cosine (past warmup)
    resumed._run_start, resumed._elapsed_before = None, 1800.0
    resumed.config.train.time_budget_hours = 1.0
    resumed.step = 10**6
    t = resumed.config.train
    assert resumed._lr() == pytest.approx((t.lr + t.lr_min) / 2)


def test_step_schedule_steps_down_by_training_hours(tmp_path):
    """Open-ended schedule: lr, halved every 24 h of training, floored."""
    config = tiny_config()
    config.train.lr_schedule = 'step'
    config.train.lr, config.train.lr_min = 0.002, 0.0003
    trainer = Trainer.new(config, tmp_path / 'run')
    trainer.step = 10**6  # past warmup
    for hours, lr in (
        (0, 0.002),
        (23.9, 0.002),
        (24, 0.001),
        (48, 0.0005),
        (200, 0.0003),
    ):
        trainer._run_start, trainer._elapsed_before = None, hours * 3600
        assert trainer._lr() == pytest.approx(lr)


def test_changed_config_reports_and_refuses_shape_changes():
    """Resume overrides list their changes; the network shape is fixed."""
    config, changes = changed_config(
        tiny_config(), ['train.lr=0.0005', 'self_play.sims=16']
    )
    assert config.train.lr == 0.0005
    assert changes == ['self_play.sims: 8 -> 16', 'train.lr: 0.002 -> 0.0005']
    with pytest.raises(ValueError, match='shape'):
        changed_config(tiny_config(), ['model.channels=16'])
    with pytest.raises(ValueError):
        changed_config(tiny_config(), ['train.nonsense=1'])


def test_stop_request_saves_and_stops(tmp_path):
    """control.json {"stop": true} ends the run after the iteration, with a
    checkpoint, and is consumed; a resume carries on from there."""
    config = tiny_config()
    config.train.iterations = 50
    run = tmp_path / 'run'
    trainer = Trainer.new(config, run)
    (run / 'control.json').write_text('{"stop": true}')
    trainer.run()
    store = RunStore.open(run)
    assert trainer.iteration == 1 and not (run / 'control.json').exists()
    assert json.loads(store.status_path.read_text())['phase'] == 'stopped'
    assert store.list_checkpoints()
    resumed = Trainer.resume(run)
    assert resumed.iteration == 1
