from pathlib import Path
import sys
import types
import pytest
from types import SimpleNamespace

from blase.utils.hashing import Hash
from blase.extracting.parquet_backend import (
    _normalize_source_list,
    _scan_parquet_manifest_stub,
    _ensure_parquet_content_hashes,
    _compute_manifest_root_hash_stub,
    _plan_parquet_batches_stub,
    _read_parquet_table_stub,
    _compute_batch_hash_stub,
    _auto_detect_image_cols,
    _validate_image_schema,
    _iter_images_from_table,
)


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def _mk_entry(path, rows=10, rgs=1, size=100, mtime=123456):
    return {
        "path": str(path),
        "rows": rows,
        "row_groups": rgs,
        "size": size,
        "mtime": mtime,
    }


def _entry_with_rgs(path, rg_rows, rg_bytes=None, size=None, rows=None):
    """Helper to build an entry dict with row-group detail."""
    rows_total = rows if rows is not None else sum(rg_rows)
    d = {
        "path": str(path),
        "rows": int(rows_total),
        "row_groups": len(rg_rows),
        "rg_rows": list(map(int, rg_rows)),
    }
    if rg_bytes is not None:
        d["rg_uncompressed_bytes"] = list(map(int, rg_bytes))
    if size is not None:
        d["size"] = int(size)
    return d


class _FakeField:
    def __init__(self, name, typ="int64"):
        self.name, self.type = name, typ


class _FakeSchema:
    def __init__(self, names=("a", "b"), metadata=None):
        self._names = list(names)
        self.metadata = {k.encode("utf-8"): b"" for k in (metadata or {})}

    def __iter__(self):
        return iter([_FakeField(n) for n in self._names])

    @property
    def names(self):
        return list(self._names)


class _FakePFMeta:
    def __init__(self, rgs, rows):
        self.num_row_groups, self.num_rows = rgs, rows


class _FakeParquetFile:
    def __init__(self, path):
        p = Path(path)
        rows, rgs = (100, 5) if p.name.endswith("_big.parquet") else (10, 1)
        self.metadata = _FakePFMeta(rgs, rows)
        self.schema_arrow = _FakeSchema(("c1", "c2", "c3"), metadata={"k1": ""})


class _FakeIPCFile:
    def __init__(self, rows=7):
        self.schema = _FakeSchema(("f1", "f2"), metadata={"m": ""})
        self.num_rows = rows


def _install_fake_pyarrow(
    monkeypatch, *, feather_rows=7, dataset_schema_names=("d1", "d2")
):
    pa = types.ModuleType("pyarrow")

    ipc_mod = types.ModuleType("pyarrow.ipc")
    ipc_mod.open_file = lambda path: _FakeIPCFile(rows=feather_rows)
    pa.ipc = ipc_mod

    pq_mod = types.ModuleType("pyarrow.parquet")
    pq_mod.ParquetFile = _FakeParquetFile

    class _FakeDatasetObj:
        def __init__(self, files):
            # IMPORTANT: keep Path objects so f.exists() works in SUT
            self.files = files
            self.schema = _FakeSchema(dataset_schema_names, metadata={"ds": ""})

    def _expand_to_files(paths):
        out = []
        for p in paths if isinstance(paths, list) else [paths]:
            p = Path(p)
            if p.is_dir():
                out.extend(sorted(p.rglob("*.parquet")))
            else:
                out.append(p)
        return out

    ds_mod = types.ModuleType("pyarrow.dataset")
    ds_mod.dataset = lambda paths, format=None: _FakeDatasetObj(_expand_to_files(paths))

    sys.modules["pyarrow"] = pa
    sys.modules["pyarrow.ipc"] = ipc_mod
    sys.modules["pyarrow.parquet"] = pq_mod
    sys.modules["pyarrow.dataset"] = ds_mod

    monkeypatch.setitem(sys.modules, "pyarrow", pa)
    monkeypatch.setitem(sys.modules, "pyarrow.ipc", ipc_mod)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", pq_mod)
    monkeypatch.setitem(sys.modules, "pyarrow.dataset", ds_mod)

    assert hasattr(pa, "ipc") and hasattr(pa.ipc, "open_file")
    assert hasattr(ds_mod, "dataset")


class _FA_Table:
    def __init__(self, cols, rows):
        self._cols = list(cols)
        self._rows = int(rows)

    def slice(self, off, cnt):
        return _FA_Table(self._cols, cnt)

    def to_pandas(self, **kw):
        return {
            "__kind__": "pandas.DataFrame",
            "rows": self._rows,
            "columns": self._cols,
        }

    # convenience for assertions
    @property
    def _rows_count(self):
        return self._rows

    @property
    def _columns(self):
        return list(self._cols)


def _install_fake_arrow_read(monkeypatch, total_rows=10, rg_rows_map=None):
    pa = types.ModuleType("pyarrow")

    # add null() so pa.null() works in pa.table({c: pa.array([], type=pa.null())})
    pa.null = lambda: None

    def _pa_array(_, type=None):
        return []

    def _pa_table(dct):
        # rows=0 when building empty table
        return _FA_Table(list(dct.keys()), rows=0)

    pa.array = _pa_array
    pa.table = _pa_table

    pq = types.ModuleType("pyarrow.parquet")

    class _PF:
        def __init__(self, frag, memory_map=True):
            pass

        def read(self, columns=None, use_threads=True):
            return _FA_Table(list(columns or []), total_rows)

        def read_row_group(self, idx, columns=None, use_threads=True):
            rows = (rg_rows_map or {}).get(int(idx), total_rows)
            return _FA_Table(list(columns or []), rows)

    pq.ParquetFile = _PF

    sys.modules["pyarrow"] = pa
    sys.modules["pyarrow.parquet"] = pq
    monkeypatch.setitem(sys.modules, "pyarrow", pa)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", pq)


class _FakePATable:
    """Minimal pyarrow.Table stand-in exposing column_names."""

    def __init__(self, cols):
        self.column_names = list(cols)


class _PandasLikeDF:
    def __init__(self, cols, dtypes_map):
        self.columns = list(cols)
        # emulate pandas .dtypes[...] lookup
        self.dtypes = dtypes_map


def _install_fake_pyarrow_types(monkeypatch):
    pa = types.ModuleType("pyarrow")

    types_mod = types.ModuleType("pyarrow.types")

    class _Int:
        pass

    class _Bin:
        pass

    class _LBin:
        pass

    class _Str:
        pass

    class _LStr:
        pass

    class _Bool:
        pass

    class _List:
        pass

    class _Dict:
        pass

    class _Null:
        pass

    class _Flt:
        pass

    class _Bad:
        pass  # unsupported label/path

    def _isinstance_factory(T):
        return lambda x: isinstance(x, T)

    types_mod.is_integer = _isinstance_factory(_Int)
    types_mod.is_binary = _isinstance_factory(_Bin)
    types_mod.is_large_binary = _isinstance_factory(_LBin)
    types_mod.is_string = _isinstance_factory(_Str)
    types_mod.is_large_string = _isinstance_factory(_LStr)
    types_mod.is_boolean = _isinstance_factory(_Bool)
    types_mod.is_list = _isinstance_factory(_List)
    types_mod.is_dictionary = _isinstance_factory(_Dict)
    types_mod.is_null = _isinstance_factory(_Null)
    types_mod.is_floating = _isinstance_factory(_Flt)

    pa.types = types_mod

    sys.modules["pyarrow"] = pa
    sys.modules["pyarrow.types"] = types_mod
    monkeypatch.setitem(sys.modules, "pyarrow", pa)
    monkeypatch.setitem(sys.modules, "pyarrow.types", types_mod)

    return SimpleNamespace(
        Int=_Int,
        Bin=_Bin,
        LBin=_LBin,
        Str=_Str,
        LStr=_LStr,
        Bool=_Bool,
        List=_List,
        Dict=_Dict,
        Null=_Null,
        Flt=_Flt,
        Bad=_Bad,
    )


class _ArrowSchema:
    def __init__(self, type_map):  # name -> type instance
        self._map = dict(type_map)

    def field(self, name):
        return SimpleNamespace(type=self._map[name])


class _ArrowTableLike:
    def __init__(self, cols, schema):
        self.column_names = list(cols)
        self.schema = schema


class _ArrowLikeTable:
    def __init__(self, data: dict):
        # data: name -> list
        self._data = {k: list(v) for k, v in data.items()}
        self.column_names = list(self._data.keys())
        self.num_rows = len(next(iter(self._data.values()))) if self._data else 0

    def __getitem__(self, col):
        return _ArrowCol(self._data[col])


class _ArrowCol:
    def __init__(self, seq):
        self._seq = list(seq)

    def combine_chunks(self):
        return self

    def to_pylist(self):
        return list(self._seq)


class _SeriesStub(list):
    def tolist(self):
        return list(self)


class _PDLikeDF:
    def __init__(self, data: dict):
        self._data = {k: list(v) for k, v in data.items()}
        self.columns = list(self._data.keys())

    def __len__(self):
        return len(next(iter(self._data.values()))) if self._data else 0

    def __getitem__(self, col):
        return _SeriesStub(self._data[col])


# ----- _normalize_source_list test -----
def test_single_file_path(tmp_path):
    f = tmp_path / "one.parquet"
    _touch(f)
    out = _normalize_source_list(str(f))
    assert out == [f]


def test_directory_recursive_default(tmp_path):
    d = tmp_path / "data"
    f1 = d / "a.parquet"
    f2 = d / "sub" / "b.parquet"
    f3 = d / "sub" / "c.feather"  # should be ignored by default pattern
    for p in (f1, f2, f3):
        _touch(p)

    out = _normalize_source_list(d)  # default pattern "**/*.parquet"
    assert out == sorted([f1, f2], key=lambda q: str(q))


def test_directory_non_recursive_with_simple_pattern(tmp_path):
    d = tmp_path / "root"
    top = d / "top.parquet"
    sub = d / "sub" / "deep.parquet"
    for p in (top, sub):
        _touch(p)

    out = _normalize_source_list(d, pattern="*.parquet", recursive=False)
    assert out == [top]


def test_list_of_mixed_sources(tmp_path):
    d = tmp_path / "mix"
    inside = d / "x.parquet"
    outside = tmp_path / "y.parquet"
    other = d / "z.txt"
    for p in (inside, outside, other):
        _touch(p)

    out = _normalize_source_list([d, outside])
    assert out == sorted([inside, outside], key=lambda q: str(q))


def test_limit_files_truncates_after_sort(tmp_path):
    d = tmp_path / "many"
    files = [d / f"f{i}.parquet" for i in range(5)]
    for p in files:
        _touch(p)

    out = _normalize_source_list(d, limit_files=2)
    expect = sorted(files, key=lambda q: str(q))[:2]
    assert out == expect


@pytest.mark.parametrize("limit", [0, -1, -5])
def test_limit_files_zero_or_negative_returns_empty(tmp_path, limit):
    d = tmp_path / "lim"
    for name in ("a.parquet", "b.parquet"):
        _touch(d / name)

    out = _normalize_source_list(d, limit_files=limit)
    assert out == []


# ----- _scan_parquet_manifest_stub test -----
def test_scan_manifest_single_files_parquet(monkeypatch, tmp_path):
    _install_fake_pyarrow(monkeypatch)

    f1 = tmp_path / "a.parquet"
    f2 = tmp_path / "b_big.parquet"
    for p in (f1, f2):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")  # size/mtime read by _file_stats

    out = _scan_parquet_manifest_stub(
        sources=[f1, f2],
        fmt="auto",
        columns=None,
        filters=None,
        schema=None,
    )

    assert out["kind"] == "parquet"
    assert out["sources"] == [str(f1), str(f2)]
    # rows from fake PF: 10 + 100
    assert out["rows"] == 110
    # row_groups from fake PF: 1 + 5
    assert out["row_groups"] == 6
    # schema names from fakePF schema_arrow
    assert out["columns"] == ["c1", "c2", "c3"]
    assert out["projected_columns"] is None
    assert out["filters_present"] is False
    assert isinstance(out["schema_fp"], str) and out["schema_fp"].startswith("schema:")
    # entries contain per-file stats
    paths = {e["path"] for e in out["entries"]}
    assert paths == {str(f1), str(f2)}


def test_scan_manifest_dataset_dir(monkeypatch, tmp_path):
    # In dataset path we pass a directory in sources.
    _install_fake_pyarrow(monkeypatch, dataset_schema_names=("dx", "dy", "dz"))

    d = tmp_path / "ds"
    d.mkdir()
    # Files the dataset would report; they need to exist for stat() and optional PF calls
    files = [d / f"part-{i}.parquet" for i in range(3)]
    for p in files:
        p.write_bytes(b"")

    out = _scan_parquet_manifest_stub(
        sources=[d],  # directory triggers dataset branch
        fmt="auto",
        columns=None,
        filters=None,
        schema=None,
    )

    assert out["kind"] == "dataset"
    assert out["sources"] == [str(d)]
    # schema from fake dataset
    assert out["columns"] == ["dx", "dy", "dz"]
    assert isinstance(out["schema_fp"], str) and out["schema_fp"].startswith("schema:")
    # entries reflect each file
    assert {Path(e["path"]).name for e in out["entries"]} == {p.name for p in files}
    # totals from fake PF defaults: each part has rows=10 rgs=1 -> 30 rows, 3 groups
    assert out["rows"] == 30
    assert out["row_groups"] == 3


def test_scan_manifest_feather_files_with_projection_and_filters(monkeypatch, tmp_path):
    _install_fake_pyarrow(monkeypatch, feather_rows=42)

    f = tmp_path / "table.feather"
    f.write_bytes(b"")

    out = _scan_parquet_manifest_stub(
        sources=[f],
        fmt="feather",  # force feather path
        columns=["f1"],
        filters=[("x", ">", 0)],  # any truthy object marks filters_present
        schema=None,
    )

    assert out["kind"] == "feather"
    assert out["rows"] == 42
    assert out["row_groups"] == 1
    assert out["columns"] == ["f1", "f2"]  # from fake IPC schema names
    assert out["projected_columns"] == ["f1"]
    assert out["filters_present"] is True
    assert isinstance(out["schema_fp"], str) and out["schema_fp"].startswith("schema:")
    assert len(out["entries"]) == 1
    e0 = out["entries"][0]
    assert e0["path"] == str(f)
    assert e0["rows"] == 42
    assert e0["row_groups"] == 1


# ----- _ensure_parquet_content_hashes test -----
def test_ensure_parquet_content_hashes_happy_path(tmp_path):
    f1 = tmp_path / "a.parquet"
    f2 = tmp_path / "b.parquet"
    f1.write_bytes(b"a")
    f2.write_bytes(b"bb")

    h1 = Hash().hash_file(f1)
    h2 = Hash().hash_file(f2)

    manifest = {
        "entries": [
            {"path": str(f1)},  # fill
            {"path": str(f2), "hash_content": None},  # overwrite None
        ]
    }
    out = _ensure_parquet_content_hashes(manifest)

    assert out["entries"][0]["hash_content"] == h1
    assert out["entries"][1]["hash_content"] == h2


def test_ensure_parquet_content_hashes_preserves_existing(tmp_path):
    f = tmp_path / "c.parquet"
    f.write_bytes(b"ccc")
    preset = "deadbeef" * 8  # 64 hex chars

    manifest = {"entries": [{"path": str(f), "hash_content": preset}]}
    out = _ensure_parquet_content_hashes(manifest)
    # unchanged
    assert out["entries"][0]["hash_content"] == preset


def test_ensure_parquet_content_hashes_missing_path_and_nonexistent(tmp_path):
    missing = tmp_path / "nope.parquet"
    manifest = {"entries": [{"path": None}, {"path": str(missing)}]}
    out = _ensure_parquet_content_hashes(manifest)

    assert out["entries"][0]["hash_content"] is None
    # nonexistent file → None (hash_file raises -> caught)
    assert out["entries"][1]["hash_content"] is None


def test_ensure_parquet_content_hashes_empty_entries_list():
    out = _ensure_parquet_content_hashes({"entries": []})
    assert out["entries"] == []


# ----- _compute_manifest_root_hash_stub test -----
def test_compute_manifest_root_hash_order_independent_sources_and_entries(tmp_path):
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    a.write_bytes(b"a")
    b.write_bytes(b"b")

    man1 = {
        "kind": "parquet",
        "schema_fp": "schema:abc",
        "sources": [str(b), str(a)],  # out of order
        "columns": ["c1", "c2"],
        "projected_columns": ["c2"],
        "filters_present": True,
        "entries": [
            _mk_entry(b, rows=100, rgs=5),
            _mk_entry(a, rows=10, rgs=1),
        ],  # out of order
    }
    man2 = {
        "kind": "parquet",
        "schema_fp": "schema:abc",
        "sources": [str(a), str(b)],  # sorted
        "columns": ["c1", "c2"],
        "projected_columns": ["c2"],
        "filters_present": True,
        "entries": [
            _mk_entry(a, rows=10, rgs=1),
            _mk_entry(b, rows=100, rgs=5),
        ],  # sorted
    }

    h1 = _compute_manifest_root_hash_stub(man1)
    h2 = _compute_manifest_root_hash_stub(man2)
    assert isinstance(h1, str) and isinstance(h2, str)
    assert h1 == h2  # order independent


def test_compute_manifest_root_hash_changes_on_entry_diff(tmp_path):
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    a.write_bytes(b"a")
    b.write_bytes(b"b")

    base = {
        "kind": "parquet",
        "schema_fp": "schema:abc",
        "sources": [str(a), str(b)],
        "columns": ["c1", "c2"],
        "projected_columns": ["c2"],
        "filters_present": True,
        "entries": [_mk_entry(a, rows=10, rgs=1), _mk_entry(b, rows=100, rgs=5)],
    }
    h_base = _compute_manifest_root_hash_stub(base)

    # Change rows for one file
    mod = dict(base)
    mod["entries"] = [_mk_entry(a, rows=11, rgs=1), _mk_entry(b, rows=100, rgs=5)]
    h_mod = _compute_manifest_root_hash_stub(mod)

    assert h_base != h_mod


def test_compute_manifest_root_hash_filters_repr_precedence(tmp_path):
    a = tmp_path / "a.parquet"
    a.write_bytes(b"a")

    man_repr = {
        "kind": "parquet",
        "schema_fp": "schema:abc",
        "sources": [str(a)],
        "columns": ["c1"],
        "projected_columns": [],
        "filters_present": True,  # present but string should take precedence
        "filters_repr": "x > 0",  # canonical string
        "entries": [_mk_entry(a, rows=1, rgs=1)],
    }
    man_present_only = dict(man_repr)
    man_present_only.pop("filters_repr")
    # same everything else
    h_repr = _compute_manifest_root_hash_stub(man_repr)
    h_present = _compute_manifest_root_hash_stub(man_present_only)
    assert h_repr != h_present  # string vs boolean presence


def test_compute_manifest_root_hash_columns_and_projection_affect_hash(tmp_path):
    a = tmp_path / "a.parquet"
    a.write_bytes(b"a")

    base = {
        "kind": "parquet",
        "schema_fp": "schema:abc",
        "sources": [str(a)],
        "columns": ["c1", "c2"],
        "projected_columns": ["c2"],
        "filters_present": False,
        "entries": [_mk_entry(a, rows=10, rgs=1)],
    }
    h_base = _compute_manifest_root_hash_stub(base)

    # Change columns order -> different identity (function preserves list order)
    cols_reordered = dict(base)
    cols_reordered["columns"] = ["c2", "c1"]
    h_cols = _compute_manifest_root_hash_stub(cols_reordered)
    assert h_cols != h_base

    # Change projection -> different identity
    proj_changed = dict(base)
    proj_changed["projected_columns"] = ["c1"]
    h_proj = _compute_manifest_root_hash_stub(proj_changed)
    assert h_proj != h_base


# ----- _plan_parquet_batches_stub test -----
def _entry_with_rgs(path, rg_rows, rg_bytes=None, size=None, rows=None):
    """Helper to build an entry dict with row-group detail."""
    rows_total = rows if rows is not None else sum(rg_rows)
    d = {
        "path": str(path),
        "rows": int(rows_total),
        "row_groups": len(rg_rows),
        "rg_rows": list(map(int, rg_rows)),
    }
    if rg_bytes is not None:
        d["rg_uncompressed_bytes"] = list(map(int, rg_bytes))
    if size is not None:
        d["size"] = int(size)
    return d


def test_plan_parquet_batches_returns_empty_when_max_rows_zero(tmp_path):
    a = tmp_path / "a.parquet"
    a.write_bytes(b"x")
    manifest = {
        "rows": 10,
        "entries": [_entry_with_rgs(a, [5, 5], rg_bytes=[1000, 1000])],
    }
    plan = _plan_parquet_batches_stub(manifest, batch_size=None, max_rows=0)
    # No cap applied; expect two plan rows (one per RG), last True on the second.
    assert len(plan) == 2
    assert plan[0] == {
        "fragment": str(a),
        "row_group": 0,
        "offset": 0,
        "count": 5,
        "is_last": False,
    }
    assert plan[1] == {
        "fragment": str(a),
        "row_group": 1,
        "offset": 0,
        "count": 5,
        "is_last": True,
    }


def test_plan_parquet_batches_manual_batch_size_with_rg_slicing(tmp_path):
    a = tmp_path / "a.parquet"
    a.write_bytes(b"x")
    # Single RG with 5 rows forces slicing when batch_size=3
    manifest = {"rows": 5, "entries": [_entry_with_rgs(a, [5], rg_bytes=[5000])]}
    plan = _plan_parquet_batches_stub(manifest, batch_size=3)
    # Expect two slices of the same RG: 0..3 and 3..5
    assert len(plan) == 2
    assert plan[0] == {
        "fragment": str(a),
        "row_group": 0,
        "offset": 0,
        "count": 3,
        "is_last": False,
    }
    assert plan[1] == {
        "fragment": str(a),
        "row_group": 0,
        "offset": 3,
        "count": 2,
        "is_last": True,
    }


def test_plan_parquet_batches_manual_batch_size_across_multiple_rgs(tmp_path):
    a = tmp_path / "a.parquet"
    a.write_bytes(b"x")
    manifest = {
        "rows": 5,
        "entries": [_entry_with_rgs(a, [3, 2], rg_bytes=[1000, 1000])],
    }
    plan = _plan_parquet_batches_stub(manifest, batch_size=4)

    # Expect three plan rows: 3 from RG0, then 1 from RG1 (fills batch), then remaining 1 from RG1.
    assert len(plan) == 3
    assert plan[0] == {
        "fragment": str(a),
        "row_group": 0,
        "offset": 0,
        "count": 3,
        "is_last": False,
    }
    assert plan[1] == {
        "fragment": str(a),
        "row_group": 1,
        "offset": 0,
        "count": 1,
        "is_last": False,
    }
    assert plan[2] == {
        "fragment": str(a),
        "row_group": 1,
        "offset": 1,
        "count": 1,
        "is_last": True,
    }


def test_plan_parquet_batches_bytes_greedy_respects_cap(tmp_path):
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    # Two RGs with known sizes: 50 and 60. Set cap≈85 (target=100, margin=0.15)
    manifest = {
        "rows": 2,
        "entries": [
            _entry_with_rgs(a, [1], rg_bytes=[50]),
            _entry_with_rgs(b, [1], rg_bytes=[60]),
        ],
    }
    plan = _plan_parquet_batches_stub(
        manifest, target_batch_bytes=100, safety_margin=0.15
    )
    # 50 fits first batch, adding 60 would exceed → flush; then second batch
    assert len(plan) == 2
    assert plan[0]["fragment"] == str(a)
    assert plan[1]["fragment"] == str(b)
    assert plan[-1]["is_last"] is True


def test_plan_parquet_batches_bytes_greedy_huge_rg_gets_own_batch(tmp_path):
    a = tmp_path / "a.parquet"
    a.write_bytes(b"x")
    # Single RG way over cap → emitted alone and flushed
    manifest = {"rows": 1, "entries": [_entry_with_rgs(a, [1], rg_bytes=[200])]}
    plan = _plan_parquet_batches_stub(
        manifest, target_batch_bytes=100, safety_margin=0.15
    )
    assert len(plan) == 1
    assert plan[0]["fragment"] == str(a)
    assert plan[0]["row_group"] == 0
    assert plan[0]["count"] == 1
    assert plan[0]["is_last"] is True


def test_plan_parquet_batches_no_rg_info_synthesizes_one_chunk_per_file(tmp_path):
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    a.write_bytes(b"x")
    b.write_bytes(b"yy")
    # No rg_rows/rg_uncompressed_bytes → one synthetic chunk per entry
    manifest = {
        "entries": [
            {"path": str(a), "rows": 3, "row_groups": 0, "size": 300},
            {"path": str(b), "rows": 2, "row_groups": 0, "size": 200},
        ]
    }
    plan = _plan_parquet_batches_stub(manifest, batch_size=None)
    assert len(plan) == 2
    assert {p["fragment"] for p in plan} == {str(a), str(b)}
    # Offsets are 0 for synthesized chunks
    assert all(p["offset"] == 0 for p in plan)
    assert plan[-1]["is_last"] is True


def test_plan_parquet_batches_images_vs_table_default_estimates_affect_packing(
    tmp_path,
):
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    # No sizes provided, so planner uses per-row defaults:
    #  images: 64000 * rows * 1.8
    #  table :  8192 * rows * 1.2
    # With target ≈ 20000*(1-0.15)=17000 cap:
    #  table: two entries 9830+9830 < 17000 → combined into one batch
    #  images: each ~115200 > 17000 → each is its own batch
    manifest = {
        "rows": 2,
        "entries": [
            {"path": str(a), "rows": 1, "row_groups": 0},  # no size -> defaults
            {"path": str(b), "rows": 1, "row_groups": 0},
        ],
    }
    table_plan = _plan_parquet_batches_stub(
        manifest, mode="table", target_batch_bytes=20000, safety_margin=0.15
    )
    images_plan = _plan_parquet_batches_stub(
        manifest, mode="images", target_batch_bytes=20000, safety_margin=0.15
    )

    assert (
        len(table_plan) == 2 or len(table_plan) == 1
    )  # planner emits one part per entry; combined into one batch across two parts
    # Both parts should be in a single batch sequence (two plan rows, last=True on second)
    assert table_plan[-1]["is_last"] is True
    # Images path produces separate batches due to huge est size
    assert len(images_plan) == 2
    assert images_plan[-1]["is_last"] is True


def test_plan_parquet_batches_respects_max_rows_limit(tmp_path):
    a = tmp_path / "a.parquet"
    a.write_bytes(b"x")
    # RGs: 4, 4, 4 rows. Limit to 9 rows → 4+4+1
    manifest = {
        "rows": 12,
        "entries": [_entry_with_rgs(a, [4, 4, 4], rg_bytes=[1000, 1000, 1000])],
    }
    plan = _plan_parquet_batches_stub(manifest, batch_size=5, max_rows=9)
    # Expect 2 plans for first RG and second RG fully, then one more for 1 row
    total = sum(p["count"] for p in plan)
    assert total == 9
    assert plan[-1]["is_last"] is True


# ----- _read_parquet_table_stub test -----
def test_read_parquet_table_stub_fallback_table_mode(tmp_path, monkeypatch):
    # Ensure import fails → fallback path
    monkeypatch.delitem(sys.modules, "pyarrow", raising=False)
    monkeypatch.delitem(sys.modules, "pyarrow.parquet", raising=False)
    # manifest columns used when no explicit projection
    manifest = {"columns": ["c1", "c2"]}
    plan = {
        "fragment": str(tmp_path / "a.parquet"),
        "row_group": 0,
        "offset": 0,
        "count": 3,
    }
    out = _read_parquet_table_stub(manifest, plan, return_type="arrow", columns=None)
    assert out["__kind__"] == "pyarrow.Table"
    assert out["rows"] == 3
    assert out["columns"] == ["c1", "c2"]


def test_read_parquet_table_stub_fallback_images_projection(monkeypatch):
    monkeypatch.delitem(sys.modules, "pyarrow", raising=False)
    monkeypatch.delitem(sys.modules, "pyarrow.parquet", raising=False)
    manifest = {
        "columns": ["foo", "img_bytes", "height", "width", "channels", "label", "path"]
    }
    # explicit columns merged with required images cols
    cols = ["foo", "extra"]
    plan = {"fragment": "/x.parquet", "row_group": 0, "offset": 0, "count": 2}
    out = _read_parquet_table_stub(
        manifest,
        plan,
        return_type="arrow",
        columns=cols,
        mode="images",
        bytes_col="img_bytes",
        dims_cols=("height", "width", "channels"),
        label_col="label",
        path_col="path",
    )
    want = set(
        ["foo", "extra", "img_bytes", "height", "width", "channels", "label", "path"]
    )
    assert set(out["columns"]) == want


def test_read_parquet_table_stub_fallback_pandas(monkeypatch):
    monkeypatch.delitem(sys.modules, "pyarrow", raising=False)
    monkeypatch.delitem(sys.modules, "pyarrow.parquet", raising=False)
    manifest = {"columns": ["a", "b"]}
    plan = {"fragment": "/x.parquet", "row_group": 0, "offset": 0, "count": 1}
    out = _read_parquet_table_stub(manifest, plan, return_type="pandas", columns=None)
    assert out["__kind__"] == "pandas.DataFrame"
    assert out["rows"] == 1
    assert out["columns"] == ["a", "b"]


def test_read_parquet_table_stub_arrow_rowgroup_and_slice(monkeypatch, tmp_path):
    _install_fake_arrow_read(monkeypatch, total_rows=8, rg_rows_map={0: 5, 1: 3})
    manifest = {"columns": ["x", "y", "z"]}
    plan = {
        "fragment": str(tmp_path / "file.parquet"),
        "row_group": 0,
        "offset": 1,
        "count": 4,
    }
    tbl = _read_parquet_table_stub(
        manifest, plan, return_type="arrow", columns=["y", "z"]
    )
    assert isinstance(tbl, _FA_Table)
    assert tbl._rows_count == 4  # sliced
    assert tbl._columns == ["y", "z"]


def test_read_parquet_table_stub_arrow_to_pandas(monkeypatch, tmp_path):
    _install_fake_arrow_read(monkeypatch, total_rows=6, rg_rows_map={0: 6})
    plan = {
        "fragment": str(tmp_path / "f.parquet"),
        "row_group": 0,
        "offset": 2,
        "count": 3,
    }
    out = _read_parquet_table_stub(
        manifest={"columns": ["a", "b"]},
        plan=plan,
        return_type="pandas",
        columns=["a"],
    )
    assert out["__kind__"] == "pandas.DataFrame"
    assert out["rows"] == 3
    assert out["columns"] == ["a"]


def test_read_parquet_table_stub_arrow_empty_projection_when_count_zero(monkeypatch):
    _install_fake_arrow_read(monkeypatch, total_rows=0)
    manifest = {"columns": ["c1", "c2", "c3"]}
    # cnt==0 triggers empty-table construction using pa.table with projected columns
    plan = {"fragment": "/any.parquet", "row_group": None, "offset": 0, "count": 0}
    tbl = _read_parquet_table_stub(manifest, plan, return_type="arrow", columns=["c2"])
    assert isinstance(tbl, _FA_Table)
    assert tbl._rows_count == 0
    assert tbl._columns == ["c2"]


def test_read_parquet_table_stub_arrow_images_projection(monkeypatch, tmp_path):
    _install_fake_arrow_read(monkeypatch, total_rows=5, rg_rows_map={0: 5})
    manifest = {"columns": ["foo"]}
    plan = {
        "fragment": str(tmp_path / "g.parquet"),
        "row_group": 0,
        "offset": 0,
        "count": 5,
    }
    tbl = _read_parquet_table_stub(
        manifest,
        plan,
        return_type="arrow",
        columns=["foo"],
        mode="images",
        bytes_col="img_bytes",
        dims_cols=("height", "width", "channels"),
        label_col="label",
        path_col="path",
    )
    req = {"foo", "img_bytes", "height", "width", "channels", "label", "path"}
    assert set(tbl._columns) == req


# ----- _compute_batch_hash_stub test -----
def test_compute_batch_hash_happy_and_stable(tmp_path):
    f = tmp_path / "a.parquet"
    f.write_bytes(b"x")
    plan = {"fragment": str(f), "row_group": 0, "offset": 0, "count": 10}
    h1 = _compute_batch_hash_stub("rootHASH", plan)
    h2 = _compute_batch_hash_stub("rootHASH", plan)
    assert isinstance(h1, str) and h1
    assert h1 == h2  # deterministic


def test_compute_batch_hash_changes_when_fields_change(tmp_path):
    f1 = tmp_path / "a.parquet"
    f2 = tmp_path / "b.parquet"
    f1.write_bytes(b"x")
    f2.write_bytes(b"y")
    base = {"fragment": str(f1), "row_group": 0, "offset": 0, "count": 10}
    h_base = _compute_batch_hash_stub("rootHASH", base)

    h_rg = _compute_batch_hash_stub("rootHASH", {**base, "row_group": 1})
    h_off = _compute_batch_hash_stub("rootHASH", {**base, "offset": 1})
    h_cnt = _compute_batch_hash_stub("rootHASH", {**base, "count": 11})
    h_frag = _compute_batch_hash_stub("rootHASH", {**base, "fragment": str(f2)})

    assert len({h_base, h_rg, h_off, h_cnt, h_frag}) == 5  # all distinct


def test_compute_batch_hash_normalizes_fragment_path(tmp_path, monkeypatch):
    f = tmp_path / "a.parquet"
    f.write_bytes(b"x")
    # use relative path by chdir into tmp_path
    monkeypatch.chdir(tmp_path)
    rel_plan = {"fragment": "./a.parquet", "row_group": 0, "offset": 2, "count": 3}
    abs_plan = {"fragment": str(f), "row_group": 0, "offset": 2, "count": 3}
    h_rel = _compute_batch_hash_stub("rootHASH", rel_plan)
    h_abs = _compute_batch_hash_stub("rootHASH", abs_plan)
    assert h_rel == h_abs  # Path().resolve() makes them identical


def test_compute_batch_hash_allows_no_fragment_when_no_row_group():
    # Feather-like single chunk without path info
    h = _compute_batch_hash_stub(
        "rootHASH", {"fragment": None, "row_group": None, "offset": 0, "count": 0}
    )
    assert isinstance(h, str) and h


def test_compute_batch_hash_errors():
    with pytest.raises(ValueError):
        _compute_batch_hash_stub(
            "", {"fragment": None, "row_group": None, "offset": 0, "count": 0}
        )
    with pytest.raises(ValueError):
        _compute_batch_hash_stub(
            "rootHASH", {"fragment": None, "row_group": 0, "offset": 0, "count": 1}
        )
    with pytest.raises(ValueError):
        _compute_batch_hash_stub(
            "rootHASH",
            {"fragment": "/x.parquet", "row_group": 0, "offset": -1, "count": 1},
        )
    with pytest.raises(ValueError):
        _compute_batch_hash_stub(
            "rootHASH",
            {"fragment": "/x.parquet", "row_group": 0, "offset": 0, "count": -5},
        )


# ----- _auto_detect_image_cols test -----
def test_auto_detect_image_cols_exact_names_pyarrow_table():
    tbl = _FakePATable(["img_bytes", "height", "width", "channels", "label", "path"])
    b, (h, w, c), lab, pth = _auto_detect_image_cols(
        tbl,
        bytes_col="img_bytes",
        dims_cols=("height", "width", "channels"),
        label_col="label",
        path_col="path",
    )
    assert (b, h, w, c, lab, pth) == (
        "img_bytes",
        "height",
        "width",
        "channels",
        "label",
        "path",
    )


def test_auto_detect_image_cols_case_insensitive_and_aliases_pandas_like():
    class DF:
        columns = ["Image", "H", "W", "C", "Category", "FilePath"]

    # Request different casing and rely on aliases for others
    b, (h, w, c), lab, pth = _auto_detect_image_cols(
        DF(),
        bytes_col="image",
        dims_cols=("height", "width", "channels"),
        label_col=None,
        path_col=None,
    )
    # image -> Image, height->H, width->W, channels->C; optional label/path via aliases
    assert b == "Image"
    assert (h, w, c) == ("H", "W", "C")
    assert lab == "Category"
    assert pth == "FilePath"


def test_auto_detect_image_cols_works_with_dict_stub_columns():
    stub = {"columns": ["bytes", "h", "w", "n_channels", "y", "rel_path"]}
    b, (h, w, c), lab, pth = _auto_detect_image_cols(
        stub,
        bytes_col=None,
        dims_cols=("height", "width", "channels"),
        label_col=None,
        path_col=None,
    )
    assert (b, h, w, c, lab, pth) == ("bytes", "h", "w", "n_channels", "y", "rel_path")


def test_auto_detect_image_cols_missing_bytes_raises():
    tbl = _FakePATable(["height", "width", "channels"])
    with pytest.raises(ValueError):
        _auto_detect_image_cols(
            tbl,
            bytes_col=None,
            dims_cols=("height", "width", "channels"),
            label_col=None,
            path_col=None,
        )


@pytest.mark.parametrize(
    "cols_missing",
    [
        (["img_bytes", "width", "channels"]),  # height missing
        (["img_bytes", "height", "channels"]),  # width missing
        (["img_bytes", "height", "width"]),  # channels missing
    ],
)
def test_auto_detect_image_cols_missing_any_dim_raises(cols_missing):
    tbl = _FakePATable(cols_missing)
    with pytest.raises(ValueError):
        _auto_detect_image_cols(
            tbl,
            bytes_col="img_bytes",
            dims_cols=("height", "width", "channels"),
            label_col=None,
            path_col=None,
        )


def test_auto_detect_image_cols_respects_explicit_label_and_path_when_present():
    tbl = _FakePATable(["img_bytes", "height", "width", "channels", "LBL", "P"])
    b, (h, w, c), lab, pth = _auto_detect_image_cols(
        tbl,
        bytes_col="img_bytes",
        dims_cols=("height", "width", "channels"),
        label_col="lbl",
        path_col="p",  # case-insensitive exact
    )
    assert (b, h, w, c) == ("img_bytes", "height", "width", "channels")
    assert lab == "LBL" and pth == "P"


def test_auto_detect_image_cols_when_label_path_not_present_returns_alias_if_available():
    tbl = _FakePATable(
        ["img_bytes", "height", "width", "channels", "target", "file_path"]
    )
    # label_col/path_col None -> find via aliases
    _, _, lab, pth = _auto_detect_image_cols(
        tbl,
        bytes_col="img_bytes",
        dims_cols=("height", "width", "channels"),
        label_col=None,
        path_col=None,
    )
    assert lab == "target"
    assert pth == "file_path"


# ----- _validate_image_schema test -----
def test_validate_image_schema_pandas_ok():
    df = _PandasLikeDF(
        ["img_bytes", "height", "width", "channels", "label", "path"],
        {
            "img_bytes": "object",  # acceptable binary-like
            "height": "int64",
            "width": "Int32",  # nullable int accepted
            "channels": "int32",
            "label": "object",  # allowed
            "path": "string",  # string-like
        },
    )
    _validate_image_schema(
        df,
        bytes_col="img_bytes",
        dims_cols=("height", "width", "channels"),
        label_col="label",
        path_col="path",
    )  # no exception


def test_validate_image_schema_pandas_bad_types_raise():
    # bytes not binary-like
    df = _PandasLikeDF(
        ["img_bytes", "height", "width", "channels"],
        {
            "img_bytes": "float64",
            "height": "int64",
            "width": "int64",
            "channels": "int64",
        },
    )
    with pytest.raises(ValueError):
        _validate_image_schema(
            df, "img_bytes", ("height", "width", "channels"), None, None
        )

    # height not int-like
    df2 = _PandasLikeDF(
        ["img_bytes", "height", "width", "channels"],
        {
            "img_bytes": "object",
            "height": "float64",
            "width": "int64",
            "channels": "int64",
        },
    )
    with pytest.raises(ValueError):
        _validate_image_schema(
            df2, "img_bytes", ("height", "width", "channels"), None, None
        )

    # path not string-like
    df3 = _PandasLikeDF(
        ["img_bytes", "height", "width", "channels", "path"],
        {
            "img_bytes": "object",
            "height": "int64",
            "width": "int64",
            "channels": "int64",
            "path": "float64",
        },
    )
    with pytest.raises(ValueError):
        _validate_image_schema(
            df3, "img_bytes", ("height", "width", "channels"), None, "path"
        )


def test_validate_image_schema_presence_errors_stub_dict():
    stub = {"columns": ["img_bytes", "height", "width"]}  # missing channels
    with pytest.raises(ValueError):
        _validate_image_schema(
            stub, "img_bytes", ("height", "width", "channels"), None, None
        )


def test_validate_image_schema_arrow_ok(monkeypatch):
    T = _install_fake_pyarrow_types(monkeypatch)
    cols = ["img", "h", "w", "c", "lab", "pth"]
    schema = _ArrowSchema(
        {
            "img": T.Bin(),
            "h": T.Int(),
            "w": T.Int(),
            "c": T.Int(),
            "lab": T.Str(),
            "pth": T.LStr(),
        }
    )
    tbl = _ArrowTableLike(cols, schema)
    _validate_image_schema(
        tbl, bytes_col="img", dims_cols=("h", "w", "c"), label_col="lab", path_col="pth"
    )  # no exception


def test_validate_image_schema_arrow_label_bad_type_raises(monkeypatch):
    T = _install_fake_pyarrow_types(monkeypatch)
    cols = ["img", "h", "w", "c", "lab"]
    schema = _ArrowSchema(
        {
            "img": T.LBin(),
            "h": T.Int(),
            "w": T.Int(),
            "c": T.Int(),
            "lab": T.Bad(),  # unsupported
        }
    )
    tbl = _ArrowTableLike(cols, schema)
    with pytest.raises(ValueError):
        _validate_image_schema(tbl, "img", ("h", "w", "c"), "lab", None)


def test_validate_image_schema_arrow_path_not_string_like(monkeypatch):
    T = _install_fake_pyarrow_types(monkeypatch)
    cols = ["img", "h", "w", "c", "pth"]
    schema = _ArrowSchema(
        {
            "img": T.Bin(),
            "h": T.Int(),
            "w": T.Int(),
            "c": T.Int(),
            "pth": T.Int(),  # not string-like
        }
    )
    tbl = _ArrowTableLike(cols, schema)
    with pytest.raises(ValueError):
        _validate_image_schema(tbl, "img", ("h", "w", "c"), None, "pth")


# ----- _iter_images_from_table test -----
def test_iter_images_stub_dict_yields_bytes_and_none_labels_paths():
    stub = {
        "columns": ["img_bytes", "height", "width", "channels"],  # minimal
        "rows": 3,
    }
    gen = _iter_images_from_table(
        table_like=stub,
        return_type="arrow",  # stub branch allowed
        bytes_col="img_bytes",
        dims_cols=("height", "width", "channels"),
        label_col=None,
        path_col=None,
        decode=None,
        images_return="bytes",
    )
    outs = list(gen)
    assert len(outs) == 3
    for imgs, labels, paths in outs:
        assert imgs == [b""]
        assert labels is None
        assert paths == [None]


def test_iter_images_arrow_basic_and_memoryview_paths_and_labels():
    data = {
        "img_bytes": [memoryview(b"abc"), b"z"],  # memoryview -> bytes
        "height": [10, 20],
        "width": [30, 40],
        "channels": [3, 3],
        "label": ["L0", "L1"],
        "path": ["/p/0.jpg", "/p/1.jpg"],
    }
    tbl = _ArrowLikeTable(data)
    gen = _iter_images_from_table(
        table_like=tbl,
        return_type="arrow",
        bytes_col="img_bytes",
        dims_cols=("height", "width", "channels"),
        label_col="label",
        path_col="path",
        decode=None,
        images_return="bytes",
    )
    out = list(gen)
    assert out[0][0] == [b"abc"] and out[0][1] == ["L0"] and out[0][2] == ["/p/0.jpg"]
    assert out[1][0] == [b"z"] and out[1][1] == ["L1"] and out[1][2] == ["/p/1.jpg"]


def test_iter_images_pandas_non_bytes_raises():
    df = _PDLikeDF(
        {
            "img": ["not-bytes"],  # invalid
            "h": [1],
            "w": [2],
            "c": [3],
        }
    )
    with pytest.raises(ValueError):
        list(
            _iter_images_from_table(
                table_like=df,
                return_type="pandas",
                bytes_col="img",
                dims_cols=("h", "w", "c"),
                label_col=None,
                path_col=None,
                decode=None,
                images_return="bytes",
            )
        )


def test_iter_images_handles_missing_dims_values_and_short_paths():
    data = {
        "img_bytes": [b"a", b"b", b"c"],
        "height": [None, 2, 3],  # None allowed
        "width": [4, None, 6],
        "channels": [3, 3, None],
        "path": ["/a", "/b"],  # shorter than rows
    }
    tbl = _ArrowLikeTable(data)
    gen = _iter_images_from_table(
        table_like=tbl,
        return_type="arrow",
        bytes_col="img_bytes",
        dims_cols=("height", "width", "channels"),
        label_col=None,
        path_col="path",
        decode=None,
        images_return="bytes",
    )
    outs = list(gen)
    assert outs[0][2] == ["/a"]
    assert outs[1][2] == ["/b"]
    assert outs[2][2] == [None]  # padded
