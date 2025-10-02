from typing import Dict, Any, Iterable
from blase.preparing.interfaces import ManifestBatch, Splits


def split(manifest_iter: Iterable[ManifestBatch], cfg: Dict[str, Any]) -> Splits: ...
