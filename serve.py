#!/usr/bin/env python3
"""Serve the web UI and its API.

Examples:
  # play and analyse with every checkpoint under runs/
  python serve.py
  # another runs directory, reachable from other machines
  python serve.py --runs E:/old_runs --host 0.0.0.0 --port 8080

Then open http://127.0.0.1:8020. Build the UI first (``npm run build`` in
``ui/``), or run its dev server (``npm run dev``), which forwards ``/api``
here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from server.app import ROOT, create_app


def main() -> None:
    """Parse the arguments and run the server."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--runs', type=Path, default=Path('runs'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8020)
    parser.add_argument('--device', help='cuda or cpu (default: cuda if any)')
    args = parser.parse_args()
    app = create_app(args.runs, args.device)
    if not (ROOT / 'ui' / 'dist').is_dir():
        print('No built UI: run "npm install && npm run build" in ui/.')
    print(f'Serving on http://{args.host}:{args.port} (Ctrl+C to stop)')
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
