#!/usr/bin/env python3
"""Entry-point so the lint package can be invoked as ``python -m scripts.lint``.

Delegates entirely to lint_repo.main().
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(REPO_ROOT)

from lint_repo import main  # noqa: E402 – needs cwd set first

sys.exit(main())
