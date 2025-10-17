from typing import Any, Dict, Optional
from pathlib import Path
from datetime import datetime

from blase.training.protocols import Tracker


class DefaultTracker(Tracker):
    def __init__(self, run_dir: Optional[str]):
        self.run_dir = Path(run_dir or "runs") / datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def start_step(self, fqn: str, params: Dict[str, Any]):
        # minimal no-op context manager; integrate with your Track later
        class Ctx:
            def __enter__(inner):
                return inner

            def __exit__(inner, exc_type, exc, tb):
                return False

            def log_metric(inner, name, value):
                pass

            def log_artifact(inner, name, path):
                pass

            @property
            def params(inner):
                return params

        return Ctx()

    def log_metric(self, name: str, value: Any):
        pass

    def log_artifact(self, name: str, path: str):
        pass

    def end_step(self, status: str = "ok"):
        pass
