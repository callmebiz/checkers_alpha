"""Web server for the UI: game state, analysis and engine moves over HTTP.

``server.engine`` does the work (positions, searches, models); ``server.app``
exposes it as a FastAPI app and serves the built UI. Start it with
``python serve.py``.
"""
