"""Training samples and the replay buffer that holds recent ones.

A sample is one position from self-play with its training targets:

- ``bits``, ``scalars``: the position's compact canonical features
  (see ``core.encoding.features``). Expanded to planes only when a batch is
  trained on, which keeps each sample under 600 bytes.
- ``policy``: the search's visit distribution over the 256 network actions,
  the target for the policy head.
- ``legal``: the legal network actions, bit-packed (256 bits -> 32 bytes).
  The policy loss is taken over legal moves only.
- ``wdl``: the game's final outcome for the side to move in that position:
  0 win, 1 draw, 2 loss. The target for the value head.

The replay buffer is a fixed-size ring: new samples overwrite the oldest, so
training always draws from the most recent ``capacity`` positions. Mixing in
older games smooths training; dropping very old ones keeps the targets close
to the current network's strength.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import numpy.typing as npt

from core.encoding import NUM_BITBOARDS
from core.game import ACTION_SIZE
from training import fileio

WIN, DRAW, LOSS = 0, 1, 2


@dataclass
class Samples:
    """A batch of training samples, one row per position."""

    bits: npt.NDArray[np.uint32]  # (n, 5)
    scalars: npt.NDArray[np.float32]  # (n, 2)
    policy: npt.NDArray[np.float16]  # (n, 256)
    legal: npt.NDArray[np.uint8]  # (n, 32) packed bits
    wdl: npt.NDArray[np.uint8]  # (n,)

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.wdl)

    @classmethod
    def empty(cls, n: int = 0) -> Samples:
        """``n`` all-zero samples."""
        return cls(
            bits=np.zeros((n, NUM_BITBOARDS), np.uint32),
            scalars=np.zeros((n, 2), np.float32),
            policy=np.zeros((n, ACTION_SIZE), np.float16),
            legal=np.zeros((n, ACTION_SIZE // 8), np.uint8),
            wdl=np.zeros(n, np.uint8),
        )

    @classmethod
    def concat(cls, parts: list[Samples]) -> Samples:
        """Join batches of samples end to end."""
        if not parts:
            return cls.empty()
        return cls(
            **{
                f.name: np.concatenate([getattr(p, f.name) for p in parts])
                for f in fields(cls)
            }
        )

    def take(self, index: npt.NDArray[np.intp] | slice) -> Samples:
        """Rows selected by ``index``."""
        return Samples(
            **{f.name: getattr(self, f.name)[index] for f in fields(self)}
        )

    def legal_mask(self) -> npt.NDArray[np.bool_]:
        """Unpacked ``(n, 256)`` bool mask of legal network actions."""
        return np.unpackbits(self.legal, axis=1).astype(np.bool_)


class ReplayBuffer:
    """Ring buffer of the most recent ``capacity`` samples."""

    def __init__(self, capacity: int) -> None:
        """Allocate room for ``capacity`` samples."""
        self.capacity = capacity
        self.data = Samples.empty(capacity)
        self.size = 0  # filled rows
        self.cursor = 0  # next row to write
        self.total_added = 0  # samples ever added, including overwritten

    def __len__(self) -> int:
        """Number of samples currently stored."""
        return self.size

    def add(self, samples: Samples) -> None:
        """Append samples, overwriting the oldest once the buffer is full."""
        n = len(samples)
        if n >= self.capacity:  # only the newest capacity rows can survive
            samples = samples.take(slice(n - self.capacity, n))
            self.total_added += n - self.capacity
            n = self.capacity
        rows = (self.cursor + np.arange(n)) % self.capacity
        for f in fields(Samples):
            getattr(self.data, f.name)[rows] = getattr(samples, f.name)
        self.cursor = (self.cursor + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
        self.total_added += n

    def sample(self, batch_size: int, rng: np.random.Generator) -> Samples:
        """A uniform random batch (with replacement) from the filled rows."""
        return self.data.take(rng.integers(0, self.size, batch_size))

    def ordered(self) -> Samples:
        """All filled rows, oldest first."""
        if self.size < self.capacity:
            return self.data.take(slice(0, self.size))
        order = (self.cursor + np.arange(self.capacity)) % self.capacity
        return self.data.take(order)

    def save(self, path: str | Path) -> None:
        """Write the filled rows to ``path`` (an ``.npz`` file).

        Written as stored, without first reordering into a copy (a full
        buffer can be hundreds of MB); ``cursor`` records where the oldest
        row is. The write goes to a temporary file first, so an interrupted
        save leaves the previous file intact.
        """
        path = Path(path)
        if self.size < self.capacity:
            data, cursor = self.data.take(slice(0, self.size)), 0  # views
        else:
            data, cursor = self.data, self.cursor
        tmp = fileio.tmp_path(path)
        with tmp.open('wb') as f:
            np.savez(
                f,
                total_added=self.total_added,
                cursor=cursor,
                **{x.name: getattr(data, x.name) for x in fields(data)},
            )
        fileio.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path, capacity: int) -> ReplayBuffer:
        """Load a saved buffer; ``capacity`` may differ from when saved."""
        buffer = cls(capacity)
        with np.load(path) as f:
            cursor = int(f['cursor']) if 'cursor' in f else 0
            # rotate so the oldest row comes first
            arrays = {
                x.name: np.roll(f[x.name], -cursor, axis=0)
                for x in fields(Samples)
            }
            buffer.add(Samples(**arrays))
            buffer.total_added = int(f['total_added'])
        return buffer
