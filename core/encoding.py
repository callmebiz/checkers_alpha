"""Network input from positions, and actions between board and net frames.

Canonical view
--------------
The network always sees the board from the side to move, as if it were White:
its own pieces start at the bottom and move up. When Black is to move, the board
is rotated 180 degrees before encoding. That is exactly the board as seen from
Black's chair.

Why a rotation and not a mirror: rotating maps (row, col) to (7 - row, 7 - col),
which keeps row + col odd, so dark squares stay dark. In square numbers it maps
``s`` to ``31 - s`` (PDN 1 <-> 32, 5 <-> 28, ...), and each player's near-left
corner stays dark. A left-right mirror would move pieces onto light squares, so
it is never used.

The rotation is only a view for the network. The game itself never holds a
rotated position: it advances with ``Position.play()`` in absolute coordinates,
and actions are translated between the two frames with ``to_canonical`` /
``from_canonical``. Rotating reverses every direction too, which is what
``FLIP_ACTION`` encodes.

Why bother: the network learns one set of patterns ("my man can capture
up-left") instead of two color-specific copies, and the replay buffer can mix
positions from both sides.

Features and planes
-------------------
``features()`` packs a position into five 32-bit bitboards plus two scalars. The
replay buffer stores this compact form. ``planes()`` expands it to the ``(N,
NUM_PLANES, 8, 8)`` float input:

    0  own men
    1  own kings
    2  opponent men
    3  opponent kings
    4  piece that must continue a multi-jump
    5  no-progress counter / draw limit       (constant plane)
    6  repetitions so far / repetition limit  (constant plane)
    7  all ones: lets convolutions tell the board edge from zero padding
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from core.game import ACTION_SIZE, BLACK, NUM_SQUARES, Position, square_to_rc

NUM_BITBOARDS = 5
NUM_PLANES = NUM_BITBOARDS + 3

# Board coordinates of each square, for scattering 32 squares onto the 8x8 grid
SQUARE_ROW = np.array([square_to_rc(s)[0] for s in range(NUM_SQUARES)])
SQUARE_COL = np.array([square_to_rc(s)[1] for s in range(NUM_SQUARES)])


def _flip_action(a: int) -> int:
    """The same hop seen from the other side.

    Square ``s`` becomes ``31 - s`` and direction ``d`` becomes ``3 - d``.

    Directions are ordered NW, NE, SW, SE, so ``3 - d`` is the opposite
    direction.
    """
    square, direction, jump = a >> 3, (a >> 1) & 3, a & 1
    return (NUM_SQUARES - 1 - square) * 8 + (3 - direction) * 2 + jump


FLIP_ACTION = np.array([_flip_action(a) for a in range(ACTION_SIZE)])
_FLIP = FLIP_ACTION.tolist()  # plain list: faster to index from Python


def reverse_bits(x: int) -> int:
    """Rotate a bitboard 180 degrees: bit s moves to bit 31 - s."""
    return int(f'{x:032b}'[::-1], 2)


def to_canonical(action: int, turn: int) -> int:
    """Board action -> network action for the side ``turn``."""
    return _FLIP[action] if turn == BLACK else action


def from_canonical(action: int, turn: int) -> int:
    """Network action -> board action. The flip is its own inverse."""
    return _FLIP[action] if turn == BLACK else action


def canonical_actions(pos: Position) -> list[int]:
    """Legal actions of ``pos`` as network indices.

    In the same order as ``pos.legal_actions()``.
    """
    legal = pos.legal_actions()
    return [_FLIP[a] for a in legal] if pos.turn == BLACK else list(legal)


def legal_mask(positions: Sequence[Position]) -> npt.NDArray[np.bool_]:
    """``(N, 256)`` bool mask of each position's legal network actions."""
    mask = np.zeros((len(positions), ACTION_SIZE), dtype=np.bool_)
    for i, pos in enumerate(positions):
        mask[i, canonical_actions(pos)] = True
    return mask


def features(
    positions: Sequence[Position],
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.float32]]:
    """Compact canonical features, as stored in the replay buffer.

    Returns ``bits (N, 5) uint32`` and ``scalars (N, 2) float32``.
    """
    n = len(positions)
    bits = np.empty((n, NUM_BITBOARDS), dtype=np.uint32)
    scalars = np.empty((n, 2), dtype=np.float32)
    for i, p in enumerate(positions):
        if p.turn == BLACK:  # rotate so the mover's men move up the board
            own, opp = reverse_bits(p.black), reverse_bits(p.white)
            kings = reverse_bits(p.kings)
            jump = 1 << (NUM_SQUARES - 1 - p.jump_sq) if p.jump_sq >= 0 else 0
        else:
            own, opp, kings = p.white, p.black, p.kings
            jump = 1 << p.jump_sq if p.jump_sq >= 0 else 0
        bits[i] = (own & ~kings, own & kings, opp & ~kings, opp & kings, jump)
        rules = p.rules
        scalars[i, 0] = min(p.no_progress / rules.draw_plies, 1.0)
        scalars[i, 1] = min(
            (p.reps - 1) / max(rules.repetition_limit - 1, 1), 1.0
        )
    return bits, scalars


_SHIFTS = np.arange(NUM_SQUARES, dtype=np.uint32)


def planes(
    bits: npt.NDArray[np.uint32], scalars: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Expand compact features to network input ``(N, NUM_PLANES, 8, 8)``."""
    n = bits.shape[0]
    # (N, 5, 32): each bitboard unpacked to one 0/1 value per square
    on = ((bits[:, :, None] >> _SHIFTS) & 1).astype(np.float32)
    out = np.zeros((n, NUM_PLANES, 8, 8), dtype=np.float32)
    # scatter the 32 squares onto the dark cells of the 8x8 grid
    out[:, :NUM_BITBOARDS, SQUARE_ROW, SQUARE_COL] = on
    out[:, NUM_BITBOARDS] = scalars[:, 0, None, None]
    out[:, NUM_BITBOARDS + 1] = scalars[:, 1, None, None]
    out[:, NUM_BITBOARDS + 2] = 1.0
    return out


def encode(positions: Sequence[Position]) -> npt.NDArray[np.float32]:
    """Network input for ``positions``: ``planes(*features(positions))``."""
    return planes(*features(positions))


class PlaneExpander(nn.Module):
    """``planes`` on the GPU: compact features in, network input out.

    Workers and the replay buffer pass features around (40 bytes per
    position as int64 bitboards plus scalars) instead of planes (2 KB), and
    the expansion runs on the GPU, inside the same CUDA graph as the network.
    """

    square_index: torch.Tensor
    shifts: torch.Tensor

    def __init__(self) -> None:
        """Precompute where each square goes on the flattened 8x8 grid."""
        super().__init__()
        index = torch.as_tensor(SQUARE_ROW * 8 + SQUARE_COL)
        self.register_buffer('square_index', index, persistent=False)
        self.register_buffer(
            'shifts', torch.arange(NUM_SQUARES), persistent=False
        )

    def forward(
        self, bits: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        """``bits (N, 5)`` int64 and ``scalars (N, 2)`` float to planes."""
        n = bits.shape[0]
        # (N, 5, 32): each bitboard unpacked to one 0/1 value per square
        on = ((bits.unsqueeze(-1) >> self.shifts) & 1).to(torch.float32)
        out = torch.zeros(
            (n, NUM_PLANES, 64), dtype=torch.float32, device=bits.device
        )
        out[:, :NUM_BITBOARDS, self.square_index] = on
        out[:, NUM_BITBOARDS] = scalars[:, 0:1]
        out[:, NUM_BITBOARDS + 1] = scalars[:, 1:2]
        out[:, NUM_BITBOARDS + 2] = 1.0
        return out.view(n, NUM_PLANES, 8, 8)
