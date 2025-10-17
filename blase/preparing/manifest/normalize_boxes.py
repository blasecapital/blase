from typing import Tuple, Sequence, List


def xywh_abs_to_xyxy_rel(
    xywh: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    clamp: bool = True,
) -> Tuple[float, float, float, float]:
    x, y, w, h = xywh
    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    # normalize
    rxmin = xmin / max(img_w, 1)
    rymin = ymin / max(img_h, 1)
    rxmax = xmax / max(img_w, 1)
    rymax = ymax / max(img_h, 1)
    if clamp:
        rxmin = max(0.0, min(1.0, rxmin))
        rymin = max(0.0, min(1.0, rymin))
        rxmax = max(0.0, min(1.0, rxmax))
        rymax = max(0.0, min(1.0, rymax))
    return (rxmin, rymin, rxmax, rymax)


def xyxy_abs_to_xyxy_rel(
    xyxy: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    clamp: bool = True,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    rx1 = x1 / max(img_w, 1)
    ry1 = y1 / max(img_h, 1)
    rx2 = x2 / max(img_w, 1)
    ry2 = y2 / max(img_h, 1)
    if clamp:
        rx1 = max(0.0, min(1.0, rx1))
        ry1 = max(0.0, min(1.0, ry1))
        rx2 = max(0.0, min(1.0, rx2))
        ry2 = max(0.0, min(1.0, ry2))
    return (rx1, ry1, rx2, ry2)


def convert_box(
    box: Sequence[float],
    image_w: int,
    image_h: int,
    *,
    coord_in: str,
    coord_out: str,
    clamp: bool,
    drop_oob: bool,
) -> Tuple[bool, List[float]]:
    """Return (keep, [xmin,ymin,xmax,ymax] in xyxy_rel)."""
    if image_w <= 0 or image_h <= 0:
        return False, []

    # Normalize input to xyxy_abs first
    if coord_in == "xywh_abs":
        x, y, w, h = map(float, box)
        x1, y1, x2, y2 = x, y, x + w, y + h
    elif coord_in == "xyxy_abs":
        x1, y1, x2, y2 = map(float, box)
    else:
        # unsupported → treat as xywh_abs
        x, y, w, h = map(float, box)
        x1, y1, x2, y2 = x, y, x + w, h + y

    # Validate geometry
    if x2 <= x1 or y2 <= y1:
        return False, []

    # Drop if fully outside and drop_oob
    if drop_oob and (x2 < 0 or y2 < 0 or x1 > image_w or y1 > image_h):
        return False, []

    # Convert to xyxy_rel
    if coord_out != "xyxy_rel":
        # For now we only output xyxy_rel across pipeline
        # Extend later if needed.
        pass

    if coord_in == "xywh_abs":
        rx1, ry1, rx2, ry2 = xywh_abs_to_xyxy_rel(
            (x1, y1, x2 - x1, y2 - y1), image_w, image_h, clamp
        )
    else:
        rx1, ry1, rx2, ry2 = xyxy_abs_to_xyxy_rel(
            (x1, y1, x2, y2), image_w, image_h, clamp
        )

    # Optionally drop degenerate after normalization
    if rx2 <= rx1 or ry2 <= ry1:
        return False, []

    return True, [rx1, ry1, rx2, ry2]
