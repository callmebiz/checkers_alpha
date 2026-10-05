import json
import math

import numpy as np
import pytest

from experiments.size_study import ORDER, Study, checkpoints
from training import fileio
from training.tournament import (
    Plan,
    bootstrap_intervals,
    fit_ratings,
    score_table,
)


def games_between(a: str, b: str, wins: int, draws: int, losses: int):
    """Game records with ``a``'s results against ``b``, colours alternating."""
    out = []
    results = ['a'] * wins + [None] * draws + ['b'] * losses
    for i, r in enumerate(results):
        black, white = (a, b) if i % 2 == 0 else (b, a)
        winner = None
        if r is not None:
            name = a if r == 'a' else b
            winner = 'black' if name == black else 'white'
        out.append({'black': black, 'white': white, 'winner': winner})
    return out


def test_score_table_counts_both_sides():
    """Each game counts for both players, draws as halves."""
    table = score_table(games_between('x', 'y', 3, 2, 1))
    assert table['x', 'y'] == (4.0, 6)
    assert table['y', 'x'] == (2.0, 6)


def test_fit_ratings_match_elo_formula():
    """A 75% score over many games is ~191 Elo; the anchor is 0."""
    games = games_between('a', 'b', 300, 0, 100)
    ratings = fit_ratings(games, anchor='b')
    assert ratings['b'] == 0
    expected = -400 * math.log10(1 / 0.75 - 1)
    assert ratings['a'] == pytest.approx(expected, abs=5)


def test_fit_ratings_chain_and_unbeaten_player():
    """Ratings are transitive through a chain and finite when unbeaten."""
    games = games_between('a', 'b', 75, 0, 25) + games_between(
        'b', 'c', 75, 0, 25
    )
    games += games_between('top', 'a', 20, 0, 0)
    ratings = fit_ratings(games, anchor='c')
    assert ratings['a'] > ratings['b'] > ratings['c'] == 0
    assert ratings['a'] == pytest.approx(2 * ratings['b'], rel=0.1)
    assert ratings['a'] < ratings['top'] < math.inf


def test_bootstrap_intervals_contain_the_estimate():
    """The 90% interval brackets the point estimate."""
    games = games_between('a', 'b', 30, 10, 20)
    ratings = fit_ratings(games, anchor='b')
    low, high = bootstrap_intervals(games, 'b', rounds=50)['a']
    assert low <= ratings['a'] <= high
    assert high - low > 10


def test_plan_round_trip(tmp_path):
    """A plan written as JSON loads back with entrants and blocks."""
    data = {
        'config': 'c.json',
        'entrants': [{'name': 'n', 'checkpoint': 'x.pt', 'sims': 1}],
        'blocks': [{'name': 'b', 'pairs': [['n', 'ab4']]}],
        'openings': 3,
    }
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(data))
    plan = Plan.load(path)
    assert plan.entrants[0].sims == 1 and plan.entrants[0].spec is None
    assert plan.blocks[0].pairs == [('n', 'ab4')]
    assert plan.openings == 3 and plan.anchor == 'ab4'


def fake_run(root, iterations: int, hours: float, every: int) -> None:
    """A run directory with metrics rows and (empty) checkpoint files."""
    (root / 'checkpoints').mkdir(parents=True)
    (root / 'config.json').write_text('{}')
    for i in range(iterations):
        fileio.append_jsonl(
            root / 'metrics.jsonl',
            {'iteration': i, 'elapsed_hours': hours * (i + 1) / iterations},
        )
        if (i + 1) % every == 0 or i == iterations - 1:
            (root / 'checkpoints' / f'iter_{i:04d}.pt').write_bytes(b'')


def test_study_plan_covers_every_entrant(tmp_path):
    """The tournament plan names real checkpoints, and every pairing is
    between declared entrants."""
    study = Study(tmp_path, hours=4.0, dry=False)
    for size, n in zip(ORDER, (100, 150, 300), strict=True):
        fake_run(study.run_dir(size), n, 4.0, 10)
    points = checkpoints(study.run_dir('6x64'))
    assert points[-1] == (299, 4.0)
    plan = Plan.load(study.plan())
    names = {e.name for e in plan.entrants}
    assert {'10x128', '6x64 raw', 'ab8'} <= names
    assert len([n for n in names if '@' in n]) == 3 * 3
    for e in plan.entrants:
        if e.checkpoint is not None:
            assert (tmp_path / 'runs').as_posix() in e.checkpoint.replace(
                '\\', '/'
            )
    pairs = [p for b in plan.blocks for p in b.pairs]
    assert pairs and all(a in names and b in names for a, b in pairs)
    assert len(set(pairs)) == len(pairs)
    hours = sorted(plan.extra['hours'][n] for n in names if '6x64 @' in n)
    assert hours == pytest.approx([1.0, 2.0, 3.0], abs=0.15)


def test_study_config_has_time_budget(tmp_path):
    """Generated configs differ only in the model and evaluation cadence."""
    study = Study(tmp_path, hours=4.25, dry=False)
    big, small = study.config_for('10x128'), study.config_for('6x64')
    assert big['train']['time_budget_hours'] == 4.25
    assert big['model']['channels'] == 128 and small['model']['blocks'] == 6
    assert big['self_play'] == small['self_play']
    assert np.isclose(
        big['arena']['every'] * 72, small['arena']['every'] * 30, rtol=0.1
    )
