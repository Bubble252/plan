from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_core_import_does_not_load_simulator_ros_or_torch() -> None:
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")
    code = (
        "import sys\n"
        "import auv_planner\n"
        "forbidden = {'holoocean', 'rclpy', 'torch'}\n"
        "loaded = forbidden.intersection(sys.modules)\n"
        "if loaded:\n"
        "    raise SystemExit(sorted(loaded))\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
