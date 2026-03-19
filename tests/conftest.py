"""pytest configuration and fixtures.

This file ensures the project `src/` package is importable when running tests from
the repository root.
"""

import os
import sys

# Make sure the repository root (containing src/) is on sys.path so tests can
# import `src.*` modules reliably.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
