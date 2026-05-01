#!/usr/bin/env python3
"""Project-root entry point. Equivalent to ``python -m skillops.cli``.

Usage::

    python run_skillops.py --task "place a clean apple in the fridge" \
        --library examples/library/
"""
import sys

if __name__ == "__main__":
    from skillops.cli import main
    sys.exit(main(sys.argv[1:]))
