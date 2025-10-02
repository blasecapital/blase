from blase.preparing.interfaces import ManifestBatch
from blase.preparing.stats import counters as counters_mod


# Fake ManifestBatch generator
def _b(rows, last=False):
    return ManifestBatch(
        data=rows, is_last=last, meta={"class_map": {"weed": 0, "radish": 1}}
    )


def _manifest_iter():
    yield _b(
        [
            {
                "image_id": "a",
                "det": [
                    {"xmin": 0.1, "ymin": 0.1, "xmax": 0.2, "ymax": 0.2, "label": 0}
                ],
                "cls": [1],
            }
        ]
    )
    yield _b([{"image_id": "b", "det": [], "cls": [0]}], last=True)


def test_counters_has_class_histogram():
    stats = counters_mod.compute(_manifest_iter(), cfg={})
    assert "class_hist" in stats and sum(stats["class_hist"].values()) >= 2
