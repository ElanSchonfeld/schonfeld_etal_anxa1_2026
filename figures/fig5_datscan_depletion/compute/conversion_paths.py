"""Location of the participant-level intermediates of the conversion scripts."""
import os
from pathlib import Path

_root = os.environ.get("PPMI_ROOT")
if not _root:
    raise RuntimeError("Set PPMI_ROOT to the authorized local PPMI data directory.")
PRIVATE = Path(_root).expanduser().resolve() / "derived" / "fig5_conversion"
PRIVATE.mkdir(parents=True, exist_ok=True)
