import json

import pytest

from core.inference import NetEvaluator, graph_batch_size
from core.model import CheckersNet
from training.actors import ActorPool
from training.config import RunConfig
from training.run_store import RunStore
from training.trainer import Trainer


@pytest.mark.parametrize(
    'n, size', [(1, 1), (3, 4), (32, 32), (33, 64), (96, 96), (97, 128)]
)
def test_graph_batch_size_pads_little(n, size):
    """Graph sizes are powers of two up to 32, then multiples of 32."""
    assert graph_batch_size(n) == size


def actor_config(workers: int) -> RunConfig:
    """A tiny CPU config that plays with ``workers`` processes."""
    return RunConfig.from_dict(
        {
            'model': {'blocks': 1, 'channels': 8, 'value_hidden': 16},
            'self_play': {
                'workers': workers,
                'games_per_iteration': 6,
                'concurrent_games': 4,
                'sims': 4,
                'fast_sims': 2,
                'max_plies': 40,
                'saved_games': 3,
            },
            'train': {
                'iterations': 2,
                'batch_size': 16,
                'min_buffer': 1,
                'buffer_size': 1000,
                'checkpoint_every': 1,
            },
            'arena': {'every': 0},
            'probes': {'every': 0},
            'seed': 0,
            'device': 'cpu',
        }
    )


@pytest.mark.slow  # starts worker processes (~10 s)
def test_training_with_worker_processes(tmp_path):
    """Worker processes produce games, samples and saved move logs."""
    Trainer.new(actor_config(workers=2), tmp_path / 'run').run()
    store = RunStore.open(tmp_path / 'run')
    rows = store.read_metrics()
    assert [r['iteration'] for r in rows] == [0, 1]
    for row in rows:
        sp = row['self_play']
        assert sp['games'] >= 6 and row['new_samples'] == sp['samples'] > 0
        assert row['memory'].get('workers_mb', 1) > 0
    saved = [
        json.loads(line)
        for f in sorted(store.games.glob('iter_*.jsonl'))
        for line in f.read_text().splitlines()
    ]
    assert len(saved) >= 1 and all(g['moves'] for g in saved)


@pytest.mark.slow
def test_dead_worker_is_reported_not_waited_on(tmp_path):
    """A killed worker raises an error instead of hanging the trainer."""
    config = actor_config(workers=1)
    pool = ActorPool(config, 1)
    try:
        pool.processes[0].terminate()
        pool.processes[0].join()
        evaluator = NetEvaluator(CheckersNet(config.model))
        with pytest.raises(RuntimeError, match='worker-0'):
            pool.play(evaluator, 4)
    finally:
        pool.close()
    assert not any(p.is_alive() for p in pool.processes)


@pytest.mark.parametrize(
    'logical, physical, workers, expected',
    [
        # hybrid i9-12900K: 8 hyperthreaded P-cores (0-15), 8 E-cores (16-23)
        (24, 16, 4, (0, [2, 4, 6, 8])),
        (24, 16, 9, (0, [2, 4, 6, 8, 10, 12, 14, 16, 17])),
        # ordinary 8-core CPU with hyperthreading: second threads come last
        (16, 8, 8, (0, [2, 4, 6, 8, 10, 12, 14, 3])),
        # no hyperthreading
        (4, 4, 5, (0, [1, 2, 3, 1, 2])),
    ],
)
def test_cpu_plan(monkeypatch, logical, physical, workers, expected):
    """The server gets its own core; workers fill fast cores first."""
    import training.actors as actors

    def cpu_count(logical_cpus: bool = True) -> int:
        """Fake core counts for this case."""
        return logical if logical_cpus else physical

    monkeypatch.setattr(
        actors.psutil, 'cpu_count', lambda logical=True: cpu_count(logical)
    )
    assert actors.cpu_plan(workers) == expected
