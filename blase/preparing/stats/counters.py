from typing import Dict, Any, Iterable
from blase.preparing.interfaces import ManifestBatch, Stats


def compute(manifest_iter: Iterable[ManifestBatch], cfg: Dict[str, Any]) -> Stats: ...
