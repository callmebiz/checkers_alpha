import json

import numpy as np
import pytest

from core.game import Position, Rules
from core.inference import NetEvaluator, UniformEvaluator
from core.model import CheckersNet, ModelConfig
from core.players import AlphaBetaPlayer, RandomPlayer
from training.arena import Contestant, elo_diff, play_match, random_openings
from training.config import RunConfig
from training.probes import (
    Probe,
    backfill,
    evaluate_probes,
    load_probes,
    merged_results,
    save_probes,
)
from training.run_store import DEFAULT_PROBES, RunStore
from training.trainer import Trainer

RULES = Rules()


# -- arena ---------------------------------------------------------------------


@pytest.mark.parametrize(
    'score, expected', [(0.5, 0.0), (0.75, 190.8), (0.25, -190.8)]
)
def test_elo_diff(score, expected):
    """Scores convert to the standard Elo differences."""
    assert elo_diff(score, 100) == pytest.approx(expected, abs=0.1)


def test_elo_diff_is_finite_for_perfect_scores():
    """A perfect score gives a large, finite, symmetric difference."""
    assert 0 < elo_diff(1.0, 10) < 1000
    assert elo_diff(0.0, 10) == pytest.approx(-elo_diff(1.0, 10))


def test_random_openings_are_distinct_and_playable():
    """Openings are distinct, valid and not already decided."""
    openings = random_openings(10, 4, np.random.default_rng(0), RULES)
    assert len(openings) == 10
    assert len({p.key for p in openings}) == 10
    for pos in openings:
        assert not pos.is_terminal and pos.jump_sq < 0
        Position.from_fen(pos.to_fen())  # a normal, valid position


def test_match_plays_each_opening_with_both_colours():
    """Each opening is played twice, with colours swapped."""
    openings = random_openings(3, 4, np.random.default_rng(1), RULES)
    a = Contestant.of(AlphaBetaPlayer(3, seed=0))
    b = Contestant.of(RandomPlayer(0))
    result = play_match(a, b, openings)
    assert result.total == 6
    assert result.score > 0.8  # alpha-beta crushes random
    starts = [g['start'] for g in result.games]
    assert all(starts.count(fen) == 2 for fen in starts)
    colours = {(g['start'], g['black']) for g in result.games}
    assert len(colours) == 6
    json.dumps(result.games)


def test_network_contestants_are_batched_and_finish():
    """Two network contestants finish their games within the ply cap."""
    openings = random_openings(2, 2, np.random.default_rng(2), RULES)
    model = CheckersNet(ModelConfig(blocks=1, channels=8, value_hidden=16))
    net = Contestant.network('net', NetEvaluator(model), 8)
    uniform = Contestant.network('uniform', UniformEvaluator(), 8)
    result = play_match(net, uniform, openings, max_plies=60)
    assert result.total == 4
    assert all(g['plies'] <= 60 for g in result.games)


def test_search_contestant_beats_random():
    """Even a network-free search beats random play."""
    openings = random_openings(2, 2, np.random.default_rng(3), RULES)
    search = Contestant.network('mcts', UniformEvaluator(), 64)
    result = play_match(search, Contestant.of(RandomPlayer(1)), openings)
    assert result.score >= 0.75


# -- probes --------------------------------------------------------------------


def test_default_probes_are_valid_and_expected_moves_are_right():
    """The shipped probes are valid and alpha-beta agrees with them."""
    probes = load_probes(DEFAULT_PROBES, RULES)
    assert len(probes) >= 5
    for probe in probes:
        if not probe.accepted():
            continue
        pos = Position.from_fen(probe.fen)
        best = AlphaBetaPlayer(8, seed=0).choose(pos)
        assert pos.move_text([best]) in probe.accepted(), probe.name


def test_load_probes_rejects_bad_entries(tmp_path):
    """Bad FENs and duplicate names are refused, naming the probe."""
    path = tmp_path / 'p.json'
    save_probes(path, [Probe('a', 'B:W1:B2')])  # White man on its last row
    with pytest.raises(ValueError, match="'a'"):
        load_probes(path, RULES)
    save_probes(path, [Probe('a', 'B:W21:B1'), Probe('a', 'B:W22:B1')])
    with pytest.raises(ValueError, match='unique'):
        load_probes(path, RULES)


def test_evaluate_probes_reports_prior_and_search():
    """Probe results carry priors, visits and a search verdict."""
    probes = [
        Probe('win', 'B:W29:B21,22', expected='21-25'),
        Probe('over', 'W:W29:B25,22'),  # White is blocked: already lost
    ]
    model = CheckersNet(ModelConfig(blocks=1, channels=8, value_hidden=16))
    results = evaluate_probes(probes, NetEvaluator(model), 64, RULES)
    assert results['over'] == {'terminal': -1}
    win = results['win']
    assert win['best'] == '21-25' and win['correct'] is True
    assert sum(m['visits'] for m in win['moves']) == 64
    assert sum(m['prior'] for m in win['moves']) == pytest.approx(1, abs=1e-3)
    assert win['search_value'] > 0.5
    json.dumps(results)


# -- trainer integration -------------------------------------------------------


def evaluation_config() -> RunConfig:
    """A tiny CPU config with the arena and probes every iteration."""
    return RunConfig.from_dict(
        {
            'model': {'blocks': 1, 'channels': 8, 'value_hidden': 16},
            'self_play': {
                'games_per_iteration': 4,
                'concurrent_games': 4,
                'sims': 8,
                'fast_sims': 4,
                'max_plies': 60,
            },
            'train': {
                'iterations': 2,
                'batch_size': 16,
                'min_buffer': 1,
                'buffer_size': 1000,
                'checkpoint_every': 5,
            },
            'arena': {
                'every': 1,
                'games': 2,
                'sims': 4,
                'opponents': ['random'],
                'max_plies': 60,
            },
            'probes': {'sims': 8},
            'seed': 0,
            'device': 'cpu',
        }
    )


def test_trainer_runs_arena_probes_and_backfill(tmp_path):
    """Training writes arena and probe rows; back-fill and trimming work."""
    Trainer.new(evaluation_config(), tmp_path / 'run').run()
    store = RunStore.open(tmp_path / 'run')

    # arena: every iteration, each after a checkpoint; Elo chains from 0
    arena = store.read_arena()
    assert [r['iteration'] for r in arena] == [0, 1]
    assert arena[0]['elo'] == 0.0 and arena[1]['elo'] is not None
    assert [r['opponent'] for r in arena[1]['results']] == ['random', 'iter 0']
    assert [p.name for p in store.list_checkpoints()] == [
        'iter_0000.pt',
        'iter_0001.pt',
    ]
    assert (store.games / 'arena_0001.jsonl').exists()

    # probes: evaluated every iteration from the run's own probes.json
    probes = load_probes(store.probes_path, RULES)
    rows = store.read_probe_results()
    assert [r['iteration'] for r in rows] == [0, 1]
    assert set(rows[0]['probes']) == {p.name for p in probes}

    # a probe added mid-run can be back-filled for every checkpoint
    probes.append(Probe('new', 'W:WK32:BK1,K5'))
    save_probes(store.probes_path, probes)
    assert backfill(store, ['new']) == 2
    merged = merged_results(store)
    assert all('new' in merged[i] and 'start' in merged[i] for i in (0, 1))

    # resuming from an older state drops the later rows, to be redone
    store.truncate_after(0)
    assert [r['iteration'] for r in store.read_arena()] == [0]
    assert all(r['iteration'] == 0 for r in store.read_probe_results())


def test_broken_probes_file_does_not_stop_training(tmp_path):
    """A corrupt probes file is skipped without stopping training."""
    config = evaluation_config()
    config.train.iterations = 1
    config.arena.every = 0
    trainer = Trainer.new(config, tmp_path / 'run')
    trainer.store.probes_path.write_text('not json', encoding='utf-8')
    trainer.run()
    assert trainer.store.read_probe_results() == []
    assert len(trainer.store.read_metrics()) == 1
    assert trainer.store.read_metrics()[0]['iteration'] == 0
