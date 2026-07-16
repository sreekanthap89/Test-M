"""
utils.py — Shared utilities for the EASY6 prediction learning project.
"""

import os
from datetime import datetime

# ── Session folder (shared across all steps when run via run_all.py) ──────────
# When run_all.py calls set_session_folder() once, every subsequent call to
# get_run_folder() returns THAT same folder instead of creating a new one.
# When a script is run standalone, _SESSION_FOLDER stays None and a fresh
# timestamped folder is created automatically.
_SESSION_FOLDER: str | None = None


def set_session_folder(path: str) -> None:
    """Called once by run_all.py to pin all scripts to a shared output folder."""
    global _SESSION_FOLDER
    _SESSION_FOLDER = path
    os.makedirs(path, exist_ok=True)


def get_run_folder(base: str = "runs") -> str:
    """
    Return the output folder for this run's charts.

    • When called from run_all.py  → returns the shared session folder.
    • When called from a standalone script → creates & returns a new
      timestamped folder:  runs/YYYY-MM-DD_HH-MM-SS/
    """
    if _SESSION_FOLDER is not None:
        return _SESSION_FOLDER
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = os.path.join(base, timestamp)
    os.makedirs(run_folder, exist_ok=True)
    return run_folder
