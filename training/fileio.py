"""Safe file writes for files that other processes may be reading.

Atomic writes: a file is written to ``<name>.tmp`` and then renamed over the
target, so a reader sees either the old file or the new one, never half of
one, and an interrupted write leaves the old file intact.

Windows caveat: renaming over a file fails while another process has it open
(say, the UI reading ``status.json`` or loading ``latest.pt``). ``replace``
therefore retries for a while. For files that don't matter much (the status
heartbeat), ``required=False`` gives up quietly instead of stopping training.

``.jsonl`` helpers: one JSON object per line, appended as results arrive.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

RETRY_SECONDS = 10.0


def replace(tmp: Path, path: Path, required: bool = True) -> bool:
    """Rename ``tmp`` over ``path``, retrying while ``path`` is in use.

    Returns whether it succeeded. If ``required``, raises after
    ``RETRY_SECONDS`` of failures; otherwise deletes ``tmp`` and returns
    False.
    """
    deadline = time.monotonic() + RETRY_SECONDS
    delay = 0.01
    while True:
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            if time.monotonic() > deadline:
                if required:
                    raise
                tmp.unlink(missing_ok=True)
                return False
            time.sleep(delay)
            delay = min(delay * 2, 0.5)


def tmp_path(path: Path) -> Path:
    """The temporary file used while writing ``path``."""
    return path.with_name(path.name + '.tmp')


def write_text(path: Path, text: str, required: bool = True) -> bool:
    """Write ``text`` to ``path`` atomically; returns whether it worked."""
    tmp = tmp_path(path)
    tmp.write_text(text, encoding='utf-8')
    return replace(tmp, path, required)


def torch_save(obj: Any, path: Path) -> None:
    """``torch.save`` ``obj`` to ``path`` atomically."""
    tmp = tmp_path(path)
    torch.save(obj, tmp)
    replace(tmp, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON row to a ``.jsonl`` file."""
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row) + '\n')


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """All rows of a ``.jsonl`` file; a half-written last line is skipped."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line still being written by another process
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Replace a ``.jsonl`` file with ``rows``, atomically."""
    write_text(path, ''.join(json.dumps(r) + '\n' for r in rows))
