"""Run configuration: one typed dataclass per section, loaded from JSON.

A config file has these sections, each optional (defaults fill in)::

    {
      "rules":     {...},   core.game.Rules          draw rules
      "model":     {...},   core.model.ModelConfig   network size
      "search":    {...},   core.mcts.SearchConfig   PUCT constants, noise
      "self_play": {...},   SelfPlayConfig           games, sims, temperature
      "train":     {...},   TrainConfig              optimiser, buffer, schedule
      "arena":     {...},   ArenaConfig              strength evaluation
      "probes":    {...},   ProbeConfig              probe positions
      "seed": 0,
      "device": "cuda",
      "backend": "auto"
    }

Unknown keys raise an error, so a typo can't silently fall back to a default.
Any value can be overridden from the command line with ``--set
section.key=value`` (the value is parsed as JSON, falling back to a string).
The fully resolved config is saved in each run directory, so a run can always
be reproduced or resumed with exactly the settings it started with.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.game import Rules
from core.mcts import SearchConfig
from core.model import ModelConfig


@dataclass
class SelfPlayConfig:
    games_per_iteration: int = 256
    # Worker processes playing games in parallel (training.actors); 0 plays
    # in the trainer's own process. Each worker needs ~0.5 GB of RAM.
    workers: int = 0
    # Pin the inference server and each worker to its own CPU (see
    # training.actors.cpu_plan). Helps most on hybrid CPUs.
    pin_cpus: bool = True
    # Games in progress at once, in total (split evenly between workers).
    # Keep it near games_per_iteration: a much larger pool makes games span
    # several iterations, so their samples come from older networks.
    concurrent_games: int = 256
    sims: int = 200  # simulations for a full search (recorded for training)
    # Playout cap randomisation (KataGo): each move gets a full search with
    # probability full_search_prob, otherwise a fast one. Only full searches
    # become training samples; fast ones just move the game along, so the
    # same compute yields many more games (better value targets).
    fast_sims: int = 50
    full_search_prob: float = 0.25
    temperature_plies: int = 16  # sample moves for this many opening plies
    temperature: float = 1.0  # ... at this temperature; then play the best
    max_plies: int = 400  # adjudicate a draw after this many plies
    # Evaluations remembered per worker (core.inference.CachedEvaluator), so
    # repeated positions skip the network; 0 turns the cache off.
    eval_cache: int = 100_000
    saved_games: int = 4  # games per iteration kept for replay in the UI


@dataclass
class TrainConfig:
    iterations: int = 100
    batch_size: int = 512
    lr: float = 2e-3  # peak learning rate, reached after warmup
    lr_min: float = 1e-4  # end of the cosine decay over all iterations
    warmup_steps: int = 200
    weight_decay: float = 1e-4
    value_weight: float = 1.0  # value loss weight relative to policy loss
    grad_clip: float = 5.0
    # Training steps per iteration = new samples * sample_reuse / batch_size,
    # so each sample is trained on about sample_reuse times while in buffer.
    sample_reuse: float = 4.0
    buffer_size: int = 300_000
    min_buffer: int = 5_000  # don't train until this many samples exist
    checkpoint_every: int = 5


@dataclass
class ArenaConfig:
    every: int = 5  # iterations between evaluations; 0 turns them off
    # Games per opponent: games / 2 random openings, each played with both
    # colours so an unbalanced opening can't favour either side.
    games: int = 20
    sims: int = 100  # the network's search budget per move
    opponents: list[str] = field(default_factory=lambda: ['ab:2', 'ab:4'])
    vs_previous: bool = True  # also play the last evaluated checkpoint (Elo)
    opening_plies: int = 4  # random plies from the start position
    max_plies: int = 300  # adjudicate a draw after this many plies
    seed: int = 1234  # fixed, so every evaluation uses the same openings


@dataclass
class ProbeConfig:
    every: int = 1  # iterations between probe evaluations; 0 turns them off
    sims: int = 200  # search budget per probe


@dataclass
class RunConfig:
    rules: Rules = field(default_factory=Rules)
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    probes: ProbeConfig = field(default_factory=ProbeConfig)
    seed: int | None = None
    device: str = 'cuda'
    # How the trainer runs the network for self-play, probes and the arena:
    # 'auto' (TensorRT if installed, else CUDA graphs), 'tensorrt',
    # 'graphs' or 'eager'. See core.inference.NetEvaluator.
    backend: str = 'auto'

    def to_dict(self) -> dict[str, Any]:
        """The config as nested plain dicts, ready for JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        """Build a config from nested dicts, rejecting unknown keys."""
        sections = _section_types()
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in sections:
                kwargs[key] = _build(sections[key], value, key)
            elif key in ('seed', 'device', 'backend'):
                kwargs[key] = value
            else:
                raise ValueError(f'Unknown config key {key!r}')
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        """Read a config from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        """Write the config to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + '\n')


def _section_types() -> dict[str, type]:
    """Each config section name and its dataclass."""
    return {
        'rules': Rules,
        'model': ModelConfig,
        'search': SearchConfig,
        'self_play': SelfPlayConfig,
        'train': TrainConfig,
        'arena': ArenaConfig,
        'probes': ProbeConfig,
    }


def _build(cls: type, values: dict[str, Any], section: str) -> Any:
    """Build section ``cls`` from ``values``, rejecting unknown keys."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f'Unknown keys in {section!r}: {sorted(unknown)}')
    return cls(**values)


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> None:
    """Apply ``section.key=value`` strings to a config dict, in place."""
    for item in overrides:
        path, sep, raw = item.partition('=')
        if not sep:
            raise ValueError(f'Override {item!r} is not of the form a.b=c')
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        *parents, leaf = path.strip().split('.')
        node = data
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
