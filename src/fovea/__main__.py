"""Allow ``python -m fovea`` to invoke the process boundary."""

from __future__ import annotations

import sys

from fovea.cli import main

if __name__ == "__main__":
    sys.exit(main())
