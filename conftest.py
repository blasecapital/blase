from pathlib import Path
import shutil

def pytest_sessionfinish(session, exitstatus):
    # ONLY remove test-created dirs in repo root — be explicit.
    for p in [Path("data"), Path("runs"), Path("restore")]:
        try:
            shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass