from typing import Any, Dict, Iterable
from blase.types import SinkResult
from blase.preparing.interfaces import ManifestBatch, Splits


def write(
    manifest_iter: Iterable[ManifestBatch],
    splits: Splits,
    cfg: Dict[str, Any],
) -> SinkResult: ...
