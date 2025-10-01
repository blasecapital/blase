# tests/unit/examin/test_parquet_adapter.py
from typing import Union
from types import SimpleNamespace
from pathlib import Path
import pytest
import re

from PIL import Image

# Adjust to your actual module path
from blase.examining import preview_image_backend as mod
from blase.examining.preview_image_backend import (
    parquet_adapter,
    _has_any_nonempty_bytes,
    parquet_bytes_available,
    sample_parquet_bytes,
    _extract_cell_bytes,
    maybe_materialize_grid,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
PIL = pytest.importorskip("PIL")


def _write_parquet(dirpath: Path, colname: str, values, extra=None) -> Path:
    extra = extra or {}
    data = {colname: pa.array(values)}
    for k, v in extra.items():
        data[k] = pa.array(v)
    table = pa.table(data)
    p = dirpath / "paths.parquet"
    pq.write_table(table, p)
    return p


def _write_parquet_simple(
    path: Path, cols: dict, row_group_size: Union[int, None] = None
) -> Path:
    table = pa.table(cols)
    pq.write_table(table, path, row_group_size=row_group_size)
    return path


def _write_parquet_sample(
    path: Path, n: int, row_group_size: Union[int, None] = None
) -> Path:
    # bytes col alternates empty/non-empty
    data = [b"" if i % 2 == 0 else b"x" for i in range(n)]
    tbl = pa.table({"bytes": pa.array(data, type=pa.binary())})
    pq.write_table(tbl, path, row_group_size=row_group_size)
    return path


def _mk_img(path: Path, size=(20, 12), color=(30, 60, 90), fmt="PNG") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    img.save(path, fmt)
    return path


def _item(path: Union[Path, str], ok=True):
    # Minimal stand-in for PreviewItem
    return SimpleNamespace(path=str(path), ok=ok)


# ==========
# parquet_bytes_available tests
# ==========
def test_binary_true(tmp_path: Path):
    p = tmp_path / "bin.parquet"
    _write_parquet_simple(p, {"bytes": pa.array([b"", None, b"x"], type=pa.binary())})
    assert parquet_bytes_available(str(p), "bytes") is True


def test_binary_false_all_empty(tmp_path: Path):
    p = tmp_path / "bin_empty.parquet"
    _write_parquet_simple(p, {"bytes": pa.array([b"", None, b""], type=pa.binary())})
    assert parquet_bytes_available(str(p), "bytes") is False


def test_fixed_size_binary_true(tmp_path: Path):
    p = tmp_path / "fsb.parquet"
    _write_parquet_simple(
        p, {"bytes": pa.array([None, b"\x00\x00\x00\x00"], type=pa.binary(4))}
    )
    assert parquet_bytes_available(str(p), "bytes") is True


def test_list_uint8_true(tmp_path: Path):
    p = tmp_path / "list.parquet"
    _write_parquet_simple(
        p, {"bytes": pa.array([[], None, [1, 2]], type=pa.list_(pa.uint8()))}
    )
    assert parquet_bytes_available(str(p), "bytes") is True


def test_large_list_uint8_false(tmp_path: Path):
    p = tmp_path / "llist_empty.parquet"
    _write_parquet_simple(
        p, {"bytes": pa.array([[], None], type=pa.large_list(pa.uint8()))}
    )
    assert parquet_bytes_available(str(p), "bytes") is False


def test_string_allowed(tmp_path: Path):
    p = tmp_path / "str.parquet"
    _write_parquet_simple(p, {"bytes": pa.array(["", None, "x"], type=pa.string())})
    assert parquet_bytes_available(str(p), "bytes") is True


def test_invalid_type_rejected(tmp_path: Path):
    p = tmp_path / "int.parquet"
    _write_parquet_simple(p, {"bytes": pa.array([0, 1, 2], type=pa.int32())})
    assert parquet_bytes_available(str(p), "bytes") is False


def test_missing_column(tmp_path: Path):
    p = tmp_path / "missing.parquet"
    _write_parquet_simple(p, {"other": pa.array([b"x"], type=pa.binary())})
    assert parquet_bytes_available(str(p), "bytes") is False


def test_not_parquet_path(tmp_path: Path):
    p = tmp_path / "not.parquet"
    p.write_text("hello")
    assert parquet_bytes_available(str(p), "bytes") is False


def test_multiple_row_groups_scan_limit_hits_second_rg(tmp_path: Path):
    # Two row groups. First RG empty. Second RG has data.
    p = tmp_path / "multi_rg.parquet"
    data = pa.array([b"", b"x"], type=pa.binary())
    _write_parquet_simple(p, {"bytes": data}, row_group_size=1)
    assert parquet_bytes_available(str(p), "bytes") is True


def test_reads_first_256_only_in_fallback(tmp_path: Path, monkeypatch):
    # Force fallback path by faking num_row_groups as 0 and implementing .read
    p = tmp_path / "fallback.parquet"
    arr = pa.array([b""] * 256 + [b"x"], type=pa.binary())
    _write_parquet_simple(p, {"bytes": arr})

    real_pf = pq.ParquetFile

    class PFShim:
        def __init__(self, path):
            self._pf = real_pf(path)
            self.schema_arrow = self._pf.schema_arrow
            self.num_row_groups = 0  # trigger fallback

        def read(self, columns=None):
            return self._pf.read(columns=columns)

    monkeypatch.setattr(pq, "ParquetFile", PFShim, raising=True)
    # Only first 256 rows are scanned. Last non-empty is ignored -> False.
    assert parquet_bytes_available(str(p), "bytes") is False


def test_row_group_read_exception_returns_false(tmp_path: Path, monkeypatch):
    p = tmp_path / "boom.parquet"
    data = pa.array([b"", b"x"], type=pa.binary())
    _write_parquet_simple(p, {"bytes": data}, row_group_size=1)

    real_pf = pq.ParquetFile

    class PFShim:
        def __init__(self, path):
            self._pf = real_pf(path)
            self.schema_arrow = self._pf.schema_arrow
            self.num_row_groups = self._pf.num_row_groups

        def read_row_group(self, rg, columns=None):
            raise RuntimeError("RG read failed")

    monkeypatch.setattr(pq, "ParquetFile", PFShim, raising=True)
    assert parquet_bytes_available(str(p), "bytes") is False


# ==========
# parquet_adapter tests
# ==========


def test_parquet_adapter_with_explicit_path_col_and_root_join(tmp_path: Path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True, exist_ok=True)
    # Create some placeholder files (not required, but realistic)
    (root / "a.jpg").write_bytes(b"x")
    (root / "sub" / "b.jpg").write_bytes(b"y")
    abs_p = tmp_path / "abs.jpg"
    abs_p.write_bytes(b"z")

    values = [
        "a.jpg",  # relative
        "sub/b.jpg",  # relative nested
        str(abs_p),  # absolute
        None,  # null
        "",  # empty
        " None ",  # string "None" after strip
        "   ",  # whitespace
    ]
    src = _write_parquet(tmp_path, "pcol", values)

    pop, meta = parquet_adapter(
        {"source": str(src), "path_col": "pcol", "path_root": str(root)}
    )

    # Expect only three valid paths
    assert pop.size() == 3
    got = [pop.path_at(i) for i in range(pop.size())]
    # All must be absolute and resolved
    assert all(Path(g).is_absolute() for g in got)

    # First two should resolve under root, third is the absolute one untouched
    assert Path(got[0]) == (root / "a.jpg").resolve()
    assert Path(got[1]) == (root / "sub" / "b.jpg").resolve()
    assert Path(got[2]) == abs_p.resolve()

    # Descriptor sanity
    d = pop.descriptor()
    assert d["kind"] == "parquet"
    assert d["source"] == str(Path(src).resolve())
    assert d["path_col"] == "pcol"
    assert d["rows"] == 3
    assert d["path_root"] == str(root.resolve())

    # Meta counts
    assert meta["rows_total"] == len(values)
    assert meta["rows_with_paths"] == 3
    assert meta["rows_null_paths"] == len(values) - 3
    assert meta["path_root"] == str(root.resolve())


@pytest.mark.parametrize("colname", ["path", "image_path", "filepath"])
def test_parquet_adapter_guesses_common_colnames(tmp_path: Path, colname: str):
    f1 = tmp_path / "f1.jpg"
    f1.write_bytes(b"x")
    f2 = tmp_path / "f2.jpg"
    f2.write_bytes(b"y")
    src = _write_parquet(tmp_path, colname, [str(f1), str(f2), None])

    pop, meta = parquet_adapter({"source": str(src), "path_root": None})
    assert pop.size() == 2
    assert pop.descriptor()["path_col"] == colname
    assert meta["rows_total"] == 3
    assert meta["rows_with_paths"] == 2
    assert meta["rows_null_paths"] == 1


def test_parquet_adapter_raises_without_path_col(tmp_path: Path):
    # No recognizable path column
    src = _write_parquet(tmp_path, "other_col", ["a", "b", "c"])
    with pytest.raises(ValueError) as e:
        parquet_adapter({"source": str(src)})
    assert "parquet_path_col" in str(e.value).lower()


def test_parquet_adapter_does_not_join_absolute_paths(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    abs_p = tmp_path / "abs.jpg"
    abs_p.write_bytes(b"z")
    rel_p = "rel.jpg"
    (root / rel_p).write_bytes(b"x")

    src = _write_parquet(tmp_path, "path", [str(abs_p), rel_p])

    pop, _ = parquet_adapter({"source": str(src), "path_root": str(root)})
    assert pop.size() == 2
    paths = [Path(pop.path_at(i)) for i in range(2)]
    # Absolute stays as is
    assert paths[0] == abs_p.resolve()
    # Relative is joined to path_root
    assert paths[1] == (root / rel_p).resolve()


def test_parquet_adapter_indexing_and_bounds(tmp_path: Path):
    p1 = tmp_path / "a.jpg"
    p1.write_bytes(b"a")
    p2 = tmp_path / "b.jpg"
    p2.write_bytes(b"b")
    src = _write_parquet(tmp_path, "path", [str(p1), str(p2)])

    pop, _ = parquet_adapter({"source": str(src)})
    assert pop.size() == 2
    assert pop.path_at(0) == str(p1.resolve())
    assert pop.path_at(1) == str(p2.resolve())
    with pytest.raises(IndexError):
        _ = pop.path_at(2)


# ==========
# _has_any_nonempty_bytes tests
# ==========
def test_binary_true_when_any_nonempty():
    arr = pa.array([b"", None, b"\x00", b"abc"], type=pa.binary())
    assert _has_any_nonempty_bytes(arr) is True


def test_binary_false_when_all_empty_or_null():
    arr = pa.array([b"", None, b""], type=pa.binary())
    assert _has_any_nonempty_bytes(arr) is False


def test_large_binary_and_fixed_size_binary():
    lb = pa.array([None, b""], type=pa.large_binary())
    fsb = pa.array([b"\x00\x00\x00\x00", None], type=pa.binary(4))
    assert _has_any_nonempty_bytes(lb) is False
    assert _has_any_nonempty_bytes(fsb) is True  # length==4 > 0


def test_string_types():
    s = pa.array(["", None, " "], type=pa.string())
    ls = pa.array([None, ""], type=pa.large_string())
    assert _has_any_nonempty_bytes(s) is True  # " " length > 0
    assert _has_any_nonempty_bytes(ls) is False


def test_list_uint8_types():
    list_reg = pa.array([[1, 2, 3], [], None], type=pa.list_(pa.uint8()))
    list_large = pa.array([[], None], type=pa.large_list(pa.uint8()))
    assert _has_any_nonempty_bytes(list_reg) is True
    assert _has_any_nonempty_bytes(list_large) is False


def test_all_null_chunk_skipped_by_null_count():
    arr = pa.array([None, None, None], type=pa.large_binary())
    assert _has_any_nonempty_bytes(arr) is False


def test_chunked_array_across_chunks():
    a1 = pa.array([b"", None], type=pa.binary())
    a2 = pa.array([b""], type=pa.binary())
    a3 = pa.array([b"x"], type=pa.binary())
    chunked = pa.chunked_array([a1, a2, a3])
    assert _has_any_nonempty_bytes(chunked) is True


def test_chunked_array_all_empty():
    a1 = pa.array([b"", None], type=pa.binary())
    a2 = pa.array(["", None], type=pa.binary())
    chunked = pa.chunked_array([a1, a2])
    assert _has_any_nonempty_bytes(chunked) is False


def test_scan_limit_256_elements_per_chunk():
    # 257 elements with only the last one non-empty -> should be False due to 256 scan cap per chunk
    data = [b""] * 256 + [b"x"]
    arr = pa.array(data, type=pa.binary())
    assert _has_any_nonempty_bytes(arr) is False

    # But if split into two chunks, second chunk’s first element is scanned -> True
    ch1 = pa.array([b""] * 256, type=pa.binary())
    ch2 = pa.array([b"x"], type=pa.binary())
    chunked = pa.chunked_array([ch1, ch2])
    assert _has_any_nonempty_bytes(chunked) is True


def test_fallback_when_is_valid_raises_uses_py_fallback():
    class _FakeItem:
        def __init__(self, data: bytes):
            self._d = data

        def as_buffer(self):  # force inner try to fail
            raise RuntimeError("no buffer")

        def as_py(self):
            return self._d

    class _FakeBinaryChunk:
        def __init__(self, items):
            self._items = items
            self.type = pa.binary()
            self.null_count = 0  # not all null

        def __len__(self):
            return len(self._items)

        def __getitem__(self, i):
            return self._items[i]

        def is_valid(self):  # make .is_valid() fail
            raise RuntimeError("boom")

    # Two elements, second is non-empty -> should return True via as_py() fallback
    fake = _FakeBinaryChunk([_FakeItem(b""), _FakeItem(b"x")])
    assert _has_any_nonempty_bytes(fake) is True


def test_string_none_literal_is_nonempty():
    arr = pa.array(["None", ""], type=pa.string())
    assert _has_any_nonempty_bytes(arr) is True


# ==========
# sample_parquet_bytes tests
# ==========


@pytest.fixture
def monkeypatch_extract_probe(monkeypatch):
    # Make helpers cheap and deterministic; expose rg/row via pseudo_path
    def fake_extract(col, i):
        # return raw bytes so fake_probe can ignore image decoding
        return col[i].as_py()

    def fake_probe(b, *, pseudo_path, **_):
        m = re.search(r"rg=(\d+):row=(\d+)", pseudo_path) or re.search(
            r"row=(\d+)$", pseudo_path
        )
        if m and len(m.groups()) == 2:
            rg, row = map(int, m.groups())
            return {
                "kind": "thumb",
                "rg": rg,
                "row": row,
                "len": 0 if b is None else len(b),
            }
        elif m:
            row = int(m.group(1))
            return {
                "kind": "thumb",
                "rg": None,
                "row": row,
                "len": 0 if b is None else len(b),
            }
        return {
            "kind": "thumb",
            "rg": None,
            "row": None,
            "len": 0 if b is None else len(b),
        }

    monkeypatch.setattr(mod, "_extract_cell_bytes", fake_extract, raising=True)
    monkeypatch.setattr(mod, "_probe_image_record_bytes", fake_probe, raising=True)


def test_missing_column_returns_empty(tmp_path: Path, monkeypatch_extract_probe):
    p = tmp_path / "t.parquet"
    # write a different column
    tbl = pa.table({"other": pa.array([b"x"], type=pa.binary())})
    pq.write_table(tbl, p)
    idx, items = sample_parquet_bytes(
        source=str(p),
        bytes_col="bytes",
        k=3,
        seed=42,
        head_read_bytes=32,
        thumb_size=64,
        max_side=256,
    )
    assert idx == []
    assert items == []


def test_deterministic_indices_and_length(tmp_path: Path, monkeypatch_extract_probe):
    p = tmp_path / "t.parquet"
    _write_parquet_sample(p, n=25, row_group_size=5)

    args = dict(
        source=str(p),
        bytes_col="bytes",
        k=7,
        seed=1337,
        head_read_bytes=16,
        thumb_size=32,
        max_side=256,
    )
    idx1, items1 = sample_parquet_bytes(**args)
    idx2, items2 = sample_parquet_bytes(**args)

    assert idx1 == sorted(idx1)
    assert idx2 == sorted(idx2)
    assert len(idx1) == 7
    assert len(items1) == 7
    assert idx1 == idx2
    # Items are in the same order as idx
    assert len(items1) == len(items2)


def test_seed_changes_sampling(tmp_path: Path, monkeypatch_extract_probe):
    p = tmp_path / "t.parquet"
    _write_parquet_sample(p, n=30, row_group_size=10)

    idx1, _ = sample_parquet_bytes(
        source=str(p),
        bytes_col="bytes",
        k=8,
        seed=1,
        head_read_bytes=8,
        thumb_size=16,
        max_side=128,
    )
    idx2, _ = sample_parquet_bytes(
        source=str(p),
        bytes_col="bytes",
        k=8,
        seed=2,
        head_read_bytes=8,
        thumb_size=16,
        max_side=128,
    )
    assert idx1 != idx2


def test_reads_only_needed_row_groups(
    tmp_path: Path, monkeypatch, monkeypatch_extract_probe
):
    p = tmp_path / "t.parquet"
    # 4 row groups of 5 rows each = 20 rows
    _write_parquet_sample(p, n=20, row_group_size=5)

    real_pf = pq.ParquetFile
    calls = {"rgs": set(), "count": 0}

    class PFShim:
        def __init__(self, path):
            self._pf = real_pf(path)
            self.schema_arrow = self._pf.schema_arrow
            self.metadata = self._pf.metadata
            self.num_row_groups = self._pf.num_row_groups

        def read_row_group(self, rg, columns=None):
            calls["rgs"].add(rg)
            calls["count"] += 1
            return self._pf.read_row_group(rg, columns=columns)

    monkeypatch.setattr(pq, "ParquetFile", PFShim, raising=True)

    idx, items = sample_parquet_bytes(
        source=str(p),
        bytes_col="bytes",
        k=6,
        seed=7,
        head_read_bytes=8,
        thumb_size=16,
        max_side=128,
    )

    assert len(idx) == 6
    assert len(items) == 6
    # Only row groups that contain sampled indices should be read
    # Compute expected RGs from idx and RG size = 5
    expected_rgs = {i // 5 for i in idx}
    assert calls["rgs"] == expected_rgs
    # And items embed rg/local info from the shimmed pseudo_path
    for g, rec in zip(idx, items):
        assert rec["kind"] == "thumb"
        assert rec["rg"] == g // 5
        assert rec["row"] == g % 5


def test_fallback_whole_read_when_no_row_groups(
    tmp_path: Path, monkeypatch, monkeypatch_extract_probe
):
    p = tmp_path / "t.parquet"
    _write_parquet_sample(p, n=12, row_group_size=4)

    real_pf = pq.ParquetFile

    class PFShim:
        def __init__(self, path):
            self._pf = real_pf(path)
            self.schema_arrow = self._pf.schema_arrow
            self.metadata = self._pf.metadata
            self.num_row_groups = 0  # trigger fallback branch

        def read(self, columns=None):
            return self._pf.read(columns=columns)

    monkeypatch.setattr(pq, "ParquetFile", PFShim, raising=True)

    idx, items = sample_parquet_bytes(
        source=str(p),
        bytes_col="bytes",
        k=5,
        seed=99,
        head_read_bytes=8,
        thumb_size=16,
        max_side=128,
    )
    assert len(idx) == 5
    assert len(items) == 5
    # In fallback, rg is None and row is present
    assert all(rec["rg"] is None for rec in items)
    assert all(isinstance(rec["row"], int) for rec in items)


def test_k_or_total_zero_returns_empty(tmp_path: Path, monkeypatch_extract_probe):
    p = tmp_path / "t.parquet"
    _write_parquet_sample(p, n=0)  # empty file

    idx, items = sample_parquet_bytes(
        source=str(p),
        bytes_col="bytes",
        k=5,
        seed=1,
        head_read_bytes=8,
        thumb_size=16,
        max_side=128,
    )
    assert idx == [] and items == []

    # k == 0
    _write_parquet_sample(p, n=10)
    idx, items = sample_parquet_bytes(
        source=str(p),
        bytes_col="bytes",
        k=0,
        seed=1,
        head_read_bytes=8,
        thumb_size=16,
        max_side=128,
    )
    assert idx == [] and items == []


# ==========
# _extract_cell_bytes tests
# ==========
def test_binary_variants():
    arr = pa.array([b"", None, b"\x01\x02"], type=pa.binary())
    assert _extract_cell_bytes(arr, 0) == b""  # empty bytes preserved
    assert _extract_cell_bytes(arr, 1) is None  # null -> None
    assert _extract_cell_bytes(arr, 2) == b"\x01\x02"

    lb = pa.array([None, b"x"], type=pa.large_binary())
    assert _extract_cell_bytes(lb, 0) is None
    assert _extract_cell_bytes(lb, 1) == b"x"

    fsb = pa.array([b"\x00\x00\x00\x00"], type=pa.binary(4))
    assert _extract_cell_bytes(fsb, 0) == b"\x00\x00\x00\x00"


def test_string_base64_and_plain_utf8():
    s = pa.array(["", None, "YWI=", "abc"], type=pa.string())
    assert _extract_cell_bytes(s, 0) is None  # empty string -> None
    assert _extract_cell_bytes(s, 1) is None
    assert _extract_cell_bytes(s, 2) == b"ab"  # base64 decode
    assert _extract_cell_bytes(s, 3) == b"abc"  # utf-8 fallback

    ls = pa.array(["4pi6", ""], type=pa.large_string())
    assert _extract_cell_bytes(ls, 0) == b"\xe2\x98\xba"  # decodes base64
    assert _extract_cell_bytes(ls, 1) is None


def test_list_uint8_and_large_list():
    la = pa.array([[1, 2, 3], [], None], type=pa.list_(pa.uint8()))
    assert _extract_cell_bytes(la, 0) == b"\x01\x02\x03"
    assert _extract_cell_bytes(la, 1) is None
    assert _extract_cell_bytes(la, 2) is None

    lla = pa.array([[255], None], type=pa.large_list(pa.uint8()))
    assert _extract_cell_bytes(lla, 0) == b"\xff"
    assert _extract_cell_bytes(lla, 1) is None


def test_chunked_array_indexing_across_chunks():
    a1 = pa.array([b"a"], type=pa.binary())
    a2 = pa.array([b"", b"bc"], type=pa.binary())
    ca = pa.chunked_array([a1, a2])
    assert _extract_cell_bytes(ca, 0) == b"a"
    assert _extract_cell_bytes(ca, 1) == b""  # empty bytes preserved
    assert _extract_cell_bytes(ca, 2) == b"bc"


def test_validity_mask_fallback_path_with_fake_chunk():
    # Build a fake chunk object to force .is_valid() to raise.
    class _FakeCell:
        def __init__(self, b: bytes):
            self._b = b

        def as_buffer(self):  # force primary branch to raise
            raise RuntimeError("no buffer")

        def as_py(self):
            return self._b

    class _FakeBinaryChunk:
        def __init__(self):
            self.type = pa.binary()
            self.null_count = 0
            self._items = [_FakeCell(b""), _FakeCell(b"x")]

        def __len__(self):
            return len(self._items)

        def __getitem__(self, i):
            return self._items[i]

        def is_valid(self):  # make validity path raise so scalar fallback runs
            raise RuntimeError("boom")

    fake = _FakeBinaryChunk()
    # Index 0 -> empty bytes -> None; index 1 -> b"x"
    assert _extract_cell_bytes(fake, 0) is None
    assert _extract_cell_bytes(fake, 1) == b"x"


# ==========
# maybe_materialize_grid tests
# ==========
def test_returns_none_when_no_ok_items(tmp_path: Path):
    out = tmp_path / "grid.png"
    items = [
        _item(tmp_path / "missing.jpg", ok=True),  # does not exist
        _item("parquet:file#rg=0:row=1", ok=True),  # pseudo path, not a file
        _item(tmp_path / "also_missing.png", ok=False),
    ]
    assert (
        maybe_materialize_grid(
            items, out, fmt="png", jpeg_quality=80, thumb_size=32, cols=3
        )
        is None
    )
    assert not out.exists()


def test_builds_png_grid_two_images_cols_clamped(tmp_path: Path):
    a = _mk_img(tmp_path / "a.png", size=(40, 20))
    b = _mk_img(tmp_path / "b.png", size=(10, 30))
    out = tmp_path / "grid.png"

    p = maybe_materialize_grid(
        [_item(a), _item(b)],
        out,
        fmt="png",
        jpeg_quality=85,
        thumb_size=32,
        cols=3,  # > number of thumbs → clamp to 2
    )
    assert p == str(out.resolve())
    assert out.exists()

    with Image.open(out) as im:
        # 2 thumbs → cols=2, rows=1 → (2*32, 1*32)
        assert im.size == (64, 32)
        assert im.mode == "RGB"


def test_builds_jpeg_grid_and_format(tmp_path: Path):
    a = _mk_img(tmp_path / "a.jpg", size=(100, 80), fmt="JPEG")
    out = tmp_path / "grid.jpg"

    p = maybe_materialize_grid(
        [_item(a)], out, fmt="jpg", jpeg_quality=60, thumb_size=32, cols=3
    )
    assert p == str(out.resolve())
    with Image.open(out) as im:
        assert im.format == "JPEG"
        assert im.size == (32, 32)  # 1x1 grid


def test_skips_unopenable_images_but_writes_if_any_valid(tmp_path: Path):
    good = _mk_img(tmp_path / "good.png", size=(16, 16))
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")

    out = tmp_path / "grid.png"
    p = maybe_materialize_grid(
        [_item(bad), _item(good)],
        out,
        fmt="png",
        jpeg_quality=80,
        thumb_size=32,
        cols=3,
    )
    assert p == str(out.resolve())
    with Image.open(out) as im:
        assert im.size == (32, 32)  # only one valid thumb


def test_cols_clamped_to_min_1_and_rows_compute(tmp_path: Path):
    a = _mk_img(tmp_path / "a.png")
    b = _mk_img(tmp_path / "b.png")
    c = _mk_img(tmp_path / "c.png")

    out = tmp_path / "grid.png"
    p = maybe_materialize_grid(
        [_item(a), _item(b), _item(c)],
        out,
        fmt="png",
        jpeg_quality=80,
        thumb_size=20,
        cols=0,
    )
    assert p == str(out.resolve())
    with Image.open(out) as im:
        # cols becomes 1, rows=3 → (1*20, 3*20)
        assert im.size == (20, 60)
