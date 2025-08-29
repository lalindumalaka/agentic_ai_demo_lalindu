
"""Markers for tests ."""

import os
from importlib.util import find_spec

import pytest


require_run_all = pytest.mark.skipif(not os.getenv("RUN_ALL"), reason="requires RUN_ALL environment variable")
require_soundfile = pytest.mark.skipif(find_spec("soundfile") is None, reason="requires soundfile")
require_torch = pytest.mark.skipif(find_spec("torch") is None, reason="requires torch")
