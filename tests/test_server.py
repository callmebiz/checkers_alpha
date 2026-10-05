import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from core.game import Position
from core.model import CheckersNet, ModelConfig, save_model
from server.app import create_app
from server.engine import Analysis, ModelPool, replay, state
from server.review import QUALITIES, classify, move_accuracy

MODEL = 'demo/checkpoints/iter_0003.pt'


@pytest.fixture(scope='module')
def client(tmp_path_factory):
    """A server on a runs directory holding one tiny CPU checkpoint."""
    runs = tmp_path_factory.mktemp('runs')
    run = runs / 'demo'
    (run / 'checkpoints').mkdir(parents=True)
    shape = {'blocks': 1, 'channels': 8, 'value_hidden': 16}
    (run / 'config.json').write_text(json.dumps({'model': shape}))
    (run / 'metrics.jsonl').write_text(
        json.dumps({'iteration': 3, 'elapsed_hours': 0.5}) + '\n'
    )
    save_model(runs / MODEL, CheckersNet(ModelConfig(**shape)))
    with TestClient(create_app(runs, device='cpu', ui=None)) as c:
        yield c


def first_jump_line() -> list[int]:
    """Hops from the start up to a position with a capture to make."""
    pos, actions = Position.initial(), []
    while not any(a & 1 for a in pos.legal_actions()):
        a = pos.legal_actions()[0]
        actions.append(a)
        pos = pos.play(a)
    return actions


def test_state_of_the_start():
    """The start: 24 pieces, Black to move with 7 hops, no turns yet."""
    s = state(None, [])
    assert s['turn'] == 'black' and len(s['pieces']) == 24
    assert len(s['legal']) == 7 and s['result'] is None
    assert {h['from'] for h in s['legal']} == {9, 10, 11, 12}
    assert s['fen'].startswith('B:W21')


def test_turns_group_hops_and_text():
    """Hops are grouped into turns with PDN text, colours alternating."""
    actions = first_jump_line()
    pos = replay(None, actions)[-1]
    jump = next(a for a in pos.legal_actions() if a & 1)
    s = state(None, [*actions, jump])
    assert [t['color'] for t in s['turns'][:2]] == ['black', 'white']
    assert 'x' in s['turns'][-1]['text']
    assert s['last']['captures'] is not None


def test_replay_rejects_illegal_actions():
    """An action that isn't legal is refused, with its ply."""
    with pytest.raises(ValueError, match='ply 1'):
        replay(None, [0])


def test_players_lists_checkpoints(client):
    """The catalog finds the checkpoint, its shape and training hours."""
    data = client.get('/api/players').json()
    assert data['default_model'] == MODEL
    run = data['runs'][0]
    assert run['run'] == 'demo' and run['shape'] == '1x8'
    assert run['checkpoints'][0]['hours'] == 0.5
    assert 'ab:4' in data['engines']


def test_state_endpoint_and_errors(client):
    """Good lines come back as states; bad ones as 400s with a reason."""
    ok = client.post('/api/state', json={'actions': []})
    assert ok.status_code == 200 and len(ok.json()['legal']) == 7
    bad = client.post('/api/state', json={'actions': [0]})
    assert bad.status_code == 400 and 'illegal' in bad.json()['detail']
    fen = client.post('/api/state', json={'start': 'B:W1:B20'})
    assert fen.status_code == 400 and 'crowning' in fen.json()['detail']


def test_analyse_counts_simulations(client):
    """Visits add up to the simulations, each move has a line."""
    r = client.post(
        '/api/analyse', json={'line': {}, 'model': MODEL, 'sims': 40}
    ).json()
    assert r['sims'] >= 40 and r['turn'] == 'black'
    assert sum(m['visits'] for m in r['moves']) == r['sims']
    assert sum(m['prior'] for m in r['moves']) == pytest.approx(1, abs=1e-3)
    assert all(m['line'][0]['action'] == m['action'] for m in r['moves'])
    assert r['best'] in [m['action'] for m in r['moves']]
    assert len(r['net']['wdl']) == 3


def test_models_must_be_checkpoints_inside_runs(client):
    """Paths outside the runs directory, or not checkpoints, are refused."""
    for model in ('../secret.pt', 'demo/config.json', 'demo/none.pt'):
        r = client.post(
            '/api/analyse', json={'line': {}, 'model': model, 'sims': 1}
        )
        assert r.status_code == 400


@pytest.mark.parametrize('player', ['random', 'ab:2', f'model:{MODEL}:16'])
def test_every_kind_of_player_moves_legally(client, player):
    """Each player returns a legal hop (models with their analysis)."""
    r = client.post(
        '/api/move', json={'line': {'actions': []}, 'player': player}
    )
    assert r.status_code == 200
    move = r.json()['move']
    assert move['action'] in Position.initial().legal_actions()
    assert (r.json()['analysis'] is not None) == player.startswith('model')


def test_evals_cover_every_turn(client):
    """A point for the start and each turn, shares adding up to one; a
    finished game ends on its result."""
    actions = first_jump_line()
    r = client.post(
        '/api/evals',
        json={'line': {'actions': actions}, 'model': MODEL, 'sims': 16},
    ).json()['points']
    assert len(r) == len(actions) + 1  # one hop per turn in this line
    assert [p['turn'] for p in r] == list(range(-1, len(actions)))
    for p in r:
        assert p['white'] + p['draw'] + p['black'] == pytest.approx(1, 1e-3)
        assert -1 <= p['q'] <= 1
    over = client.post(
        '/api/evals',
        json={
            'line': {'start': 'B:W15,23:B10', 'actions': [79, 149]},
            'model': MODEL,
        },
    ).json()['points']
    assert len(over) == 2 and over[-1]['black'] == 1 and over[-1]['q'] == -1


def test_review_judges_every_turn(client):
    """One verdict per complete turn, sides summarised; a turn with one
    legal move is forced and left out of accuracy."""
    actions = first_jump_line()
    pos = replay(None, actions)[-1]
    jumps = [a for a in pos.legal_actions() if a & 1]
    line = [*actions, jumps[0]]
    r = client.post(
        '/api/review', json={'line': {'actions': line}, 'sims': 32}
    ).json()
    assert r['judge'] == MODEL  # the only checkpoint
    assert len(r['turns']) == len(line)  # one hop per turn here
    for t in r['turns']:
        assert t['quality'] in QUALITIES and 0 <= t['cost'] <= 1
    if len(jumps) == 1 and len(pos.complete_moves()) == 1:
        assert r['turns'][-1]['quality'] == 'forced'
        assert r['turns'][-1]['accuracy'] is None
    counts = r['sides']['black']['counts']
    blacks = sum(t['color'] == 'black' for t in r['turns'])
    assert sum(counts.values()) == blacks


def test_review_of_nothing_is_empty(client):
    """A line with no moves has no verdicts."""
    r = client.post('/api/review', json={'line': {}}).json()
    assert r['turns'] == [] and r['sides']['white']['accuracy'] is None


def test_unknown_player_is_refused(client):
    """A player spec that means nothing is a 400."""
    r = client.post('/api/move', json={'line': {}, 'player': 'ab:x'})
    assert r.status_code == 400


def test_live_analysis_streams_until_done(client):
    """Snapshots arrive with growing simulations; the last one is done.
    A new request replaces the old one."""
    with client.websocket_connect('/api/live') as ws:
        ws.send_json({'id': 1, 'line': {}, 'model': MODEL, 'sims': 300})
        ws.send_json({'id': 2, 'line': {}, 'model': MODEL, 'sims': 30})
        seen = []
        while True:
            msg = ws.receive_json()
            seen.append(msg)
            if msg['id'] == 2 and msg['done']:
                break
        assert seen[-1]['sims'] >= 30
        ws.send_json({'id': 3, 'line': {}, 'model': 'nope.pt'})
        assert 'error' in ws.receive_json()


def test_classify_follows_the_thresholds():
    """Categories by cost, in the documented order."""
    assert classify(0.0, 0.5, True, True, True) == 'forced'
    assert classify(0.25, 0.5, False, False, False) == 'blunder'
    assert classify(0.16, 0.8, False, False, False) == 'miss'
    assert classify(0.16, 0.5, False, False, False) == 'mistake'
    assert classify(0.06, 0.5, False, False, False) == 'inaccuracy'
    assert classify(0.0, 0.5, True, False, True) == 'great'
    assert classify(0.0, 0.95, True, False, True) == 'best'  # decided
    assert classify(0.0, 0.5, True, False, False) == 'best'
    assert classify(0.01, 0.5, False, False, False) == 'excellent'
    assert classify(0.03, 0.5, False, False, False) == 'good'


def test_move_accuracy_curve():
    """A free move is 100, accuracy falls with cost, never below 0."""
    assert move_accuracy(0) == pytest.approx(100, abs=0.01)
    assert move_accuracy(0.05) < move_accuracy(0.01) < 100
    assert move_accuracy(1.0) == 0


def test_strongest_reads_tournament_ratings(tmp_path):
    """The top-rated searching network wins; raw networks and alpha-beta
    don't count."""
    for name in ('a', 'b'):
        ckpt = tmp_path / name / 'checkpoints' / 'iter_0001.pt'
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b'')
    t = tmp_path / 'study' / 'tournament'
    t.mkdir(parents=True)

    def path(name: str) -> str:
        """Where checkpoint ``name`` lives."""
        return str(tmp_path / name / 'checkpoints' / 'iter_0001.pt')

    (t / 'plan.json').write_text(
        json.dumps(
            {
                'entrants': [
                    {'name': 'a', 'checkpoint': path('a'), 'sims': 200},
                    {'name': 'b', 'checkpoint': path('b'), 'sims': 200},
                    {'name': 'b raw', 'checkpoint': path('b'), 'sims': 1},
                    {'name': 'ab8', 'spec': 'ab:8'},
                ]
            }
        )
    )
    (t / 'ratings.json').write_text(
        json.dumps(
            {
                'ratings': {
                    'a': {'elo': 700},
                    'b': {'elo': 650},
                    'b raw': {'elo': 900},
                    'ab8': {'elo': 1000},
                }
            }
        )
    )
    assert (
        ModelPool(tmp_path, 'cpu').strongest() == 'a/checkpoints/iter_0001.pt'
    )
    assert ModelPool(tmp_path / 'a', 'cpu').strongest() is None


def test_search_wdl_sees_through_a_forced_recapture():
    """Two kings each; White takes one and Black must take back. The
    network alone sees Black a king down and lost; the search has played
    the recapture and sees the draw. Its win minus loss is Q."""
    pool = ModelPool(Path('runs'))
    model = pool.strongest()
    if model is None:
        pytest.skip('no trained checkpoint under runs/')
    pos = Position.from_fen('W:WK14,K15:BK10,K9')
    capture = next(a for a in pos.legal_actions() if a & 1)
    analysis = Analysis(pos.play(capture), pool.evaluator(model))
    analysis.step(400)
    snap = analysis.snapshot()
    win, draw, loss = snap['wdl']
    assert snap['net']['wdl'][2] > 0.8  # the network: Black is lost
    assert draw > 0.8  # the search: a draw
    assert win + draw + loss == pytest.approx(1, abs=1e-3)
    assert win - loss == pytest.approx(snap['q'], abs=1e-3)
