from blase.preparing.manifest.normalize_boxes import convert_box


def test_convert_xyxy_abs_to_rel_with_clamp():
    ok, out = convert_box(
        box=[-1, 0.5, 3, 2.5],  # partially oob
        image_w=4,
        image_h=4,
        coord_in="xyxy_abs",
        coord_out="xyxy_rel",
        clamp=True,
        drop_oob=False,
    )
    assert ok
    # clamped then normalized to [0,1]
    assert 0.0 <= out[0] <= 1.0 and 0.0 <= out[2] <= 1.0
    assert out[0] <= out[2] and out[1] <= out[3]


def test_drop_oob_when_enabled():
    ok, _ = convert_box(
        box=[-5, -5, -1, -1],
        image_w=10,
        image_h=10,
        coord_in="xyxy_abs",
        coord_out="xyxy_rel",
        clamp=False,
        drop_oob=True,
    )
    assert ok is False
