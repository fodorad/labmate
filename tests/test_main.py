import subprocess
import sys


def test_module_entrypoint_shows_help():
    out = subprocess.run(
        [sys.executable, "-m", "labmate", "--help"], capture_output=True, text=True
    )
    assert out.returncode == 0
    assert "probe" in out.stdout and "paper2flow" in out.stdout
