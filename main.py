"""Compatibility entrypoint for running Papaya via `python main.py`."""

from papaya.__main__ import main  # noqa: F401


if __name__ == "__main__":
    main()
