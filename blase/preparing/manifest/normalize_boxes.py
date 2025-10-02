from typing import Tuple, Sequence, List


def convert_box(
    box: Sequence[float],
    image_w: int,
    image_h: int,
    *,
    coord_in: str,
    coord_out: str,
    clamp: bool,
    drop_oob: bool,
) -> Tuple[bool, List[float]]: ...
