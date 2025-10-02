from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence
from blase.types import Batch


def align_stream(
    label_iters: Sequence[Iterable[Dict[str, Any]]],
    kv_index: Mapping[str, Dict[str, Any]],
    rg_index: Any,
    *,
    join_cfg: Dict[str, Any],
    box_cfg: Dict[str, Any],
    class_cfg: Dict[str, Any],
    scale_cfg: Dict[str, Any],
) -> Iterator[Batch[Sequence[Dict[str, Any]], Dict[str, Any]]]: ...
