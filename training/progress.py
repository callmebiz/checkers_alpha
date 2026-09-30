"""Console progress bars, logging that coexists with them, and memory stats.

The console stays compact during a long run:

- An outer bar counts iterations; its postfix shows the latest losses,
  draw rate, speed and Elo.
- An inner bar follows the current phase (self-play games, training steps,
  arena games, ...) and disappears when the phase ends.
- Only notable events (start, arena results, warnings) are printed, above
  the bars. Everything else goes to ``train.log`` in detail.

When the output isn't an interactive terminal (a redirected or background
run), bars are turned off and a one-line summary per iteration is printed
instead, so a log file of the console output stays readable.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import torch
from tqdm import tqdm

try:
    import psutil
except ImportError:  # optional: process memory is then not reported
    psutil = None


def interactive() -> bool:
    """Whether stdout is a terminal that can show progress bars."""
    return sys.stdout.isatty()


class TqdmHandler(logging.Handler):
    """Logging handler that prints above any active progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        """Print the log entry above the progress bars."""
        try:
            tqdm.write(self.format(record), file=sys.stdout)
            sys.stdout.flush()  # show promptly even when redirected to a file
        except Exception:
            self.handleError(record)


class Progress:
    """An outer bar over iterations and one inner bar per phase."""

    def __init__(self, total: int, initial: int, enabled: bool) -> None:
        """Create the run bar; nothing is drawn unless ``enabled``."""
        self.enabled = enabled
        self.outer = tqdm(
            total=total,
            initial=initial,
            desc='run',
            unit='it',
            position=0,
            dynamic_ncols=True,
            disable=not enabled,
        )
        self.inner: tqdm | None = None
        # tracked here too, since disabled bars don't count
        self.done = 0
        self.total = 0

    def phase(self, name: str, total: int, unit: str) -> None:
        """Start the inner bar for a new phase (closing the previous one)."""
        self.end_phase()
        self.done, self.total = 0, total
        self.inner = tqdm(
            total=total,
            desc=f'  {name}',
            unit=unit,
            position=1,
            leave=False,
            dynamic_ncols=True,
            disable=not self.enabled,
        )

    def advance(self, n: int = 1) -> None:
        """Count ``n`` more units of the current phase."""
        self.done += n
        if self.inner is not None:
            self.inner.update(n)

    def end_phase(self) -> None:
        """Close the phase bar, if one is open."""
        if self.inner is not None:
            self.inner.close()
            self.inner = None

    def iteration_done(self, postfix: dict[str, Any]) -> None:
        """Close the phase bar and advance the run bar with new stats."""
        self.end_phase()
        self.outer.set_postfix(postfix, refresh=False)
        self.outer.update(1)

    def close(self) -> None:
        """Close all bars."""
        self.end_phase()
        self.outer.close()


def memory_snapshot(device: torch.device) -> dict[str, float]:
    """Current memory use in MB: process RSS (and the workers' combined
    RSS), and CUDA allocated / peak / reserved in this process. Resets the
    CUDA peak, so each snapshot covers the time since the previous one.
    Useful to spot leaks over a long run."""
    stats: dict[str, float] = {}
    if psutil is not None:
        process = psutil.Process()
        stats['process_mb'] = round(process.memory_info().rss / 2**20)
        children = process.children()  # self-play workers, if any
        if children:
            rss = 0
            for child in children:
                try:
                    rss += child.memory_info().rss
                except psutil.Error:
                    pass  # exited meanwhile
            stats['workers_mb'] = round(rss / 2**20)
    if device.type == 'cuda':
        stats['cuda_mb'] = round(torch.cuda.memory_allocated(device) / 2**20)
        peak = torch.cuda.max_memory_allocated(device)
        stats['cuda_peak_mb'] = round(peak / 2**20)
        reserved = torch.cuda.memory_reserved(device)
        stats['cuda_reserved_mb'] = round(reserved / 2**20)
        torch.cuda.reset_peak_memory_stats(device)
    return stats
