"""``python -m agent_tools`` entrypoint; the console surface lives in ``cli``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
