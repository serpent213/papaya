"""Compatibility entrypoint for running Papaya via `python main.py`."""

import sys
from pathlib import Path

# Add src to path for development mode
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from papaya.__main__ import main  # noqa: F401


if __name__ == "__main__":
    main()
