"""The policy/value network.

Architecture
------------
A standard AlphaZero-style residual network over the 8x8 board::

    planes (N, 8, 8, 8)
      -> stem conv
      -> `blocks` residual blocks, `channels` wide
           |-> policy head -> 256 action logits
           '-> value head  -> 3 logits: win / draw / loss

Both outputs are from the point of view of the side to move (see
``core.encoding``).

Policy head
-----------
The policy head is convolutional. It ends in 8 output channels, one per hop type
(4 directions x step/jump), evaluated on every square. Reading those 8 values at
each of the 32 dark squares gives ``32 * 8 = 256`` logits in the same ``square *
8 + hop`` order as the actions in ``core.game``.

Only 170 of the 256 hops fit on the board: from an edge or corner square, some
directions leave it. Those outputs simply go unused. Search takes a softmax over
the *legal* moves only, and training masks the loss to legal moves too. Nothing
is wasted in learning, and the layout helps generalisation: one set of
convolution weights predicts "capture up-left" on every square. The zero padding
plus the all-ones input plane tell the network where the edges are. A fully
connected head, as in the old model, would have to learn each square separately.

Value head
----------
Predicts win/draw/loss probabilities rather than one number. Draws are common in
checkers, and "50% draw, 50% win" is different from "75% win, 25% loss" even
though both average to +0.5. Search uses the scalar ``value = P(win) -
P(loss)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.encoding import NUM_PLANES, SQUARE_COL, SQUARE_ROW

# 4 directions x (step, jump); matches the action layout in core.game
HOPS_PER_SQUARE = 8


@dataclass
class ModelConfig:
    blocks: int = 6
    channels: int = 64
    value_hidden: int = 128


class ResBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection: ``relu(x + f(x))``."""

    def __init__(self, channels: int) -> None:
        """Two 3x3 convolutions with batch norm, ``channels`` wide."""
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply both convolutions and add the input back (skip connection)."""
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class CheckersNet(nn.Module):
    """Residual network with a per-square policy head and a win/draw/loss head.

    ``forward(x)`` takes ``(N, NUM_PLANES, 8, 8)`` canonical planes and returns
    ``(policy_logits (N, 256), wdl_logits (N, 3))``.
    """

    square_index: torch.Tensor

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Build the stem, the residual tower and both heads from ``config``."""
        super().__init__()
        self.config = config or ModelConfig()
        ch = self.config.channels
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[ResBlock(ch) for _ in range(self.config.blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(
                ch, HOPS_PER_SQUARE, 1
            ),  # 8 hop logits at every cell of the 8x8 grid
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(ch, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 64, self.config.value_hidden),
            nn.ReLU(),
            nn.Linear(self.config.value_hidden, 3),
        )

        # Where the 32 dark squares sit in the flattened 8x8 policy map
        self.register_buffer(
            'square_index',
            torch.as_tensor(SQUARE_ROW * 8 + SQUARE_COL),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Input planes to ``(policy_logits, wdl_logits)``."""
        x = self.blocks(self.stem(x))
        hop_map = self.policy_head(x)  # (N, 8, 8, 8): hop type x row x col
        # (N, 8, 32): keep the dark squares only. A gather with one fixed
        # index (rather than indexing rows and columns separately) compiles
        # to a static-shape TensorRT engine.
        hops = hop_map.flatten(2).index_select(2, self.square_index)
        # (N, 256), ordered square * 8 + hop like the actions
        policy = hops.transpose(1, 2).reshape(x.shape[0], -1)
        return policy, self.value_head(x)


def save_model(path: str | Path, model: CheckersNet, **extra: Any) -> None:
    """Save weights and config so ``load_model`` can rebuild the net.

    ``extra`` keys are stored alongside (the trainer adds optimizer state,
    iteration, etc.).
    """
    torch.save(
        {
            'model_config': asdict(model.config),
            'model_state': model.state_dict(),
            **extra,
        },
        path,
    )


def load_model(
    path: str | Path, device: str | torch.device = 'cpu'
) -> tuple[CheckersNet, dict[str, Any]]:
    """Load a checkpoint written by ``save_model`` or the trainer.

    Returns ``(model in eval mode, full checkpoint dict)``.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CheckersNet(ModelConfig(**ckpt['model_config']))
    model.load_state_dict(ckpt['model_state'])
    return model.to(device).eval(), ckpt
