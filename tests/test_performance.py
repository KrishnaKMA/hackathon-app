"""Hardware dependent runs are explicit and produce JSON, without timing assertions."""

import subprocess
import sys

import pytest


@pytest.mark.performance
def test_reproducible_benchmark():
    subprocess.run(
        [sys.executable, "scripts/benchmark.py", "--runs", "30", "--threads", "4"],
        check=True,
    )
