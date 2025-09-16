from pathlib import Path
import types
import sys
import importlib.util
import pytest

from blase.utils.hashing import Hash
from blase.extracting.image_backend import (
    auto_target_bytes_from_system,
    scan_manifest_headers,
    shuffle_manifest,
    ensure_item_content_hashes,
    compute_manifest_root_hash,
    plan_image_batches,
    decode_batch_pil,
    decode_batch_cv2,
    iter_batches_from_plan,
    compute_batch_hash
)

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")
cv2 = pytest.importorskip("cv2")

from PIL import Image  # noqa: E402
Image.init()

import PIL.JpegImagePlugin  # noqa: F401
import PIL.PngImagePlugin   # noqa: F401

# Helpers to create dummy modules
def make_module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m

@pytest.fixture(autouse=True)
def clean_modules(monkeypatch):
    # Ensure clean import space each test
    to_clear = ["torch", "tensorflow", "pynvml", "psutil"]
    snap = {k: sys.modules.get(k) for k in to_clear}
    for k in to_clear:
        sys.modules.pop(k, None)
    yield
    # restore
    for k, v in snap.items():
        if v is not None:
            sys.modules[k] = v

def mock_find_spec(monkeypatch, present):
    """present: set[str] of module names to pretend are importable"""
    def _find_spec(name, package=None):
        return object() if name in present else None
    monkeypatch.setattr(importlib.util, "find_spec", _find_spec, raising=True)

def install_fake_pillow(monkeypatch, mapping):
    """
    mapping: dict[str, dict] keyed by filename, value:
      {"size": (w,h), "mode": "RGB"/"L"/..., "exif": {0x0112: int}, "raise": bool}
    """
    class UnidentifiedImageError(Exception): ...
    class _Img:
        def __init__(self, p: Path):
            self._p = Path(p)
            spec = mapping.get(self._p.name, {})
            if spec.get("raise"):
                raise UnidentifiedImageError("bad image")
            self.size = spec.get("size", (0,0))
            self.mode = spec.get("mode", None)
            self._exif = spec.get("exif", {})
        def getexif(self):
            return dict(self._exif)
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class Image:
        @staticmethod
        def open(p): return _Img(Path(p))

    PIL = make_module("PIL", Image=Image, UnidentifiedImageError=UnidentifiedImageError)
    sys.modules["PIL"] = PIL

    # Pretend importlib can find it
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, package=None: object() if name=="PIL" else None, raising=True)
    return PIL

def patch_hash_in_module(monkeypatch):
    # Replace the Hash class with a fake that exposes hash_bytes(bytes)->hexstr
    import blase.extracting.image_backend as mod  # adjust to your module path
    class FakeHash:
        @staticmethod
        def hash_bytes(b: bytes) -> str:
            import hashlib
            return hashlib.sha256(b).hexdigest()
    monkeypatch.setattr(mod, "Hash", FakeHash, raising=True)

def _mk_manifest(n=5):
    return [{"rel_path": f"{i}.jpg", "i": i} for i in range(n)]

def _content_items():
    return [
        {"rel_path": "a.jpg", "hash_content": "aa"*32},
        {"rel_path": "b.jpg", "hash_content": "bb"*32},
    ]


def _pms_items():
    return [
        {"rel_path": "a.jpg", "hash_path_mtime": "h1"},
        {"rel_path": "b.jpg", "hash_path_mtime": "h2"},
    ]

def rec(w=None, h=None, c=None, est=None, name=None):
    d = {"width": w, "height": h, "channels": c}
    if est is not None:
        d["estimated_decoded_bytes"] = est
    if name is not None:
        d["rel_path"] = name
    return d


def est_bytes(w, h, c):  # convenience
    return int((w or 0) * (h or 0) * (c or 0))

def _make_img(path: Path, size=(32, 24), mode="RGB", color=(10, 20, 30)):
    pytest.importorskip("PIL")
    from PIL import Image, PpmImagePlugin

    Image.init()
    # Ensure reader/writer are registered
    Image.register_save("PPM", PpmImagePlugin._save)
    Image.register_open(PpmImagePlugin.PpmImageFile.format, PpmImagePlugin.PpmImageFile)
    Image.register_extension("PPM", ".ppm")

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new(mode, size, color if mode != "L" else 128)

    # Write PPM bytes regardless of filename extension. Pillow opens by header.
    img.save(path, format="PPM")
    return path

def _rgb_png(path: Path, w=6, h=4, rgb=(30, 20, 10)):
    """Create a lossless PNG with known RGB content by writing BGR via OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = (rgb[2], rgb[1], rgb[0])
    img = np.full((h, w, 3), bgr, dtype=np.uint8)  # BGR for OpenCV
    ok = cv2.imwrite(str(path), img)  # PNG by extension
    assert ok, "cv2.imwrite failed"


def _gray_png(path: Path, w=7, h=5, val=123):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((h, w), val, dtype=np.uint8)
    ok = cv2.imwrite(str(path), img)
    assert ok, "cv2.imwrite failed"

def _hc(text: str) -> str:
    return Hash().hash_bytes(text.encode("utf-8"))


# ----- auto_target_bytes_from_system test -----

def test_tensor_prefers_torch_cuda(monkeypatch):
    # torch present with CUDA available
    class DummyCUDA:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def mem_get_info(): return (1_000_000_000, 2_000_000_000)  # free, total
    torch = make_module("torch", cuda=DummyCUDA)
    sys.modules["torch"] = torch
    mock_find_spec(monkeypatch, {"torch"})
    got = auto_target_bytes_from_system("tensor")
    assert got == int(1_000_000_000 * 0.70)
    assert 128 * 2**20 <= got <= 2 * 2**30

def test_tensor_torch_raises_falls_to_cpu(monkeypatch):
    class BadCUDA:
        @staticmethod
        def is_available(): return True
        @staticmethod
        def mem_get_info(): raise RuntimeError("boom")
    sys.modules["torch"] = make_module("torch", cuda=BadCUDA)
    # psutil fallback
    class VMem: 
        available = 10 * 2**30  # 10 GiB
    psutil = make_module("psutil", virtual_memory=lambda: VMem)
    sys.modules["psutil"] = psutil
    mock_find_spec(monkeypatch, {"torch", "psutil"})
    got = auto_target_bytes_from_system("tensor")
    # 50% of 10 GiB = 5 GiB, clamped at 2 GiB
    assert got == 2 * 2**30

def test_tensor_tf_uses_pynvml(monkeypatch):
    # torch absent, TF present with GPU, pynvml present
    class TFConfig:
        @staticmethod
        def list_physical_devices(kind): return ["GPU0"] if kind == "GPU" else []
        class experimental:
            pass
    tf = make_module("tensorflow", config=TFConfig)
    sys.modules["tensorflow"] = tf

    class NVMLInfo: 
        free = 800 * 2**20  # 800 MiB
    class NVML:
        @staticmethod
        def nvmlInit(): pass
        @staticmethod
        def nvmlShutdown(): pass
        @staticmethod
        def nvmlDeviceGetHandleByIndex(idx): return object()
        @staticmethod
        def nvmlDeviceGetMemoryInfo(handle): return NVMLInfo
    sys.modules["pynvml"] = make_module("pynvml", **{k: getattr(NVML, k) for k in dir(NVML) if not k.startswith("_")})

    mock_find_spec(monkeypatch, {"tensorflow", "pynvml"})
    got = auto_target_bytes_from_system("tensor")
    assert got == int(0.70 * 800 * 2**20)  # 70% of 800 MiB
    assert got >= 128 * 2**20

def test_tensor_tf_virtual_device_when_no_pynvml(monkeypatch):
    # TF present with GPU and virtual device limit; no pynvml
    class VCfg: 
        memory_limit = 1024  # MiB
    class TFExperimental:
        @staticmethod
        def get_virtual_device_configuration(_gpu): return [VCfg]
        @staticmethod
        def get_memory_info(_): return {"current": 100 * 2**20}  # 100 MiB
    class TFConfig:
        @staticmethod
        def list_physical_devices(kind): return ["GPU0"] if kind == "GPU" else []
        experimental = TFExperimental
    tf = make_module("tensorflow", config=TFConfig)
    sys.modules["tensorflow"] = tf

    mock_find_spec(monkeypatch, {"tensorflow"})
    got = auto_target_bytes_from_system("tensor")
    # total=1024 MiB, current=100 MiB → free=924 MiB → 70%
    assert got == int(0.70 * 924 * 2**20)
    assert got >= 128 * 2**20

def test_cpu_path_with_psutil(monkeypatch):
    class V: available = 6 * 2**30  # 6 GiB
    psutil = make_module("psutil", virtual_memory=lambda: V)
    sys.modules["psutil"] = psutil
    mock_find_spec(monkeypatch, {"psutil"})
    got = auto_target_bytes_from_system("np")
    # 50% of 6 GiB = 3 GiB, clamped at 2 GiB
    assert got == 2 * 2**30

def test_cpu_min_clamp(monkeypatch):
    class V: available = 32 * 2**20  # 32 MiB
    psutil = make_module("psutil", virtual_memory=lambda: V)
    sys.modules["psutil"] = psutil
    mock_find_spec(monkeypatch, {"psutil"})
    got = auto_target_bytes_from_system("pil")
    assert got == 128 * 2**20  # min clamp

def test_fallback_none_when_unprobeable(monkeypatch):
    mock_find_spec(monkeypatch, set())  # nothing importable
    assert auto_target_bytes_from_system("tensor") is None
    assert auto_target_bytes_from_system("np") is None

# ------ scan_manifest_headers test -----

def test_happy_path_headers_sorted_and_fields(tmp_path, monkeypatch):
    # Arrange: files a.jpg, b.png, c.gif (corrupt)
    a = tmp_path / "a.jpg"; a.write_bytes(b"X"*123); Path(a).touch()
    b = tmp_path / "b.png"; b.write_bytes(b"Y"*45);  Path(b).touch()
    c = tmp_path / "c.gif"; c.write_bytes(b"Z"*1);   Path(c).touch()

    # Fake PIL behavior for each file
    mapping = {
        "a.jpg": {"size": (10, 20), "mode": "RGB", "exif": {0x0112: 6}},
        "b.png": {"size": (5, 5), "mode": "L"},
        "c.gif": {"raise": True},
    }
    install_fake_pillow(monkeypatch, mapping)
    patch_hash_in_module(monkeypatch)

    # Act
    items = scan_manifest_headers(str(tmp_path), "**/*", recursive=False, filename_filter=None)

    # Assert: sorted by rel_path
    rels = [it["rel_path"] for it in items]
    assert rels == ["a.jpg", "b.png", "c.gif"]

    a_rec = items[0]; b_rec = items[1]; c_rec = items[2]

    assert a_rec["ext"] == "jpg" and b_rec["ext"] == "png"
    assert a_rec["width"] == 10 and a_rec["height"] == 20 and a_rec["mode"] == "RGB"
    assert a_rec["channels"] == 3 and a_rec["exif_orientation"] == 6
    assert a_rec["is_corrupt"] is False
    assert a_rec["estimated_decoded_bytes"] == 10*20*3

    assert b_rec["channels"] == 1 and b_rec["exif_orientation"] in (None, 1)
    assert c_rec["is_corrupt"] is True
    assert isinstance(a_rec["hash_path_mtime"], str) and len(a_rec["hash_path_mtime"]) == 64

def test_filename_filter_excludes(tmp_path, monkeypatch):
    f1 = tmp_path / "keep.jpg"; f1.write_bytes(b"1")
    f2 = tmp_path / "skip.jpg"; f2.write_bytes(b"2")
    install_fake_pillow(monkeypatch, {"keep.jpg": {"size": (1,1), "mode": "RGB"},
                                      "skip.jpg": {"size": (1,1), "mode": "RGB"}})
    patch_hash_in_module(monkeypatch)

    items = scan_manifest_headers(str(tmp_path), "*.jpg", recursive=False, filename_filter=lambda p: not p.endswith("skip.jpg"))
    assert [it["rel_path"] for it in items] == ["keep.jpg"]

def test_recursive_double_star_stripped_when_recursive_false(tmp_path, monkeypatch):
    sub = tmp_path / "sub"; sub.mkdir()
    (tmp_path / "top.jpg").write_bytes(b"x")
    (sub / "deep.jpg").write_bytes(b"y")

    install_fake_pillow(monkeypatch, {"top.jpg": {"size": (1,1), "mode": "RGB"},
                                      "deep.jpg": {"size": (1,1), "mode": "RGB"}})
    patch_hash_in_module(monkeypatch)

    items_no_rec = scan_manifest_headers(str(tmp_path), "**/*.jpg", recursive=False, filename_filter=None)
    assert [it["rel_path"] for it in items_no_rec] == ["top.jpg"]

    items_rec = scan_manifest_headers(str(tmp_path), "**/*.jpg", recursive=True, filename_filter=None)
    assert sorted([it["rel_path"] for it in items_rec]) == ["sub/deep.jpg", "top.jpg"]

def test_dependency_missing_raises(monkeypatch, tmp_path):
    # Ensure PIL not importable
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, package=None: None, raising=True)
    with pytest.raises(RuntimeError):
        scan_manifest_headers(str(tmp_path), "*.jpg", recursive=False, filename_filter=None)

def test_bad_directory_raises(monkeypatch, tmp_path):
    # Make PIL present to reach path check
    install_fake_pillow(monkeypatch, {})
    with pytest.raises(NotADirectoryError):
        scan_manifest_headers(str(tmp_path / "nope"), "*.jpg", recursive=False, filename_filter=None)

def test_deterministic_ordering(tmp_path, monkeypatch):
    names = ["z2.jpg", "a0.jpg", "m1.jpg"]
    for n in names:
        (tmp_path / n).write_bytes(b"x")
    install_fake_pillow(monkeypatch, {n: {"size": (1,1), "mode": "RGB"} for n in names})
    patch_hash_in_module(monkeypatch)

    items = scan_manifest_headers(str(tmp_path), "*.jpg", recursive=False, filename_filter=None)
    assert [it["rel_path"] for it in items] == sorted(names)

# ----- shuffle_manifest test -----
def test_seed_none_returns_same_object_no_reorder():
    m = _mk_manifest(4)
    out = shuffle_manifest(m, seed=None)
    assert out is m
    assert [r["i"] for r in out] == [0,1,2,3]

def test_deterministic_same_seed_same_order():
    m = _mk_manifest(6)
    a = shuffle_manifest(m, seed=42)
    b = shuffle_manifest(m, seed=42)
    assert a is not m and b is not m
    assert [r["i"] for r in a] == [r["i"] for r in b]
    # Still a permutation of input
    assert sorted(r["i"] for r in a) == list(range(6))

def test_different_seeds_change_order():
    m = _mk_manifest(6)
    a = shuffle_manifest(m, seed=1)
    c = shuffle_manifest(m, seed=2)
    assert [r["i"] for r in a] != [r["i"] for r in c]

def test_not_in_place_when_seed_given_and_items_preserved_by_identity():
    # Ensure dict objects are the same instances in new order (shallow copy)
    m = _mk_manifest(5)
    ids_in = [id(x) for x in m]
    out = shuffle_manifest(m, seed=7)
    assert out is not m
    assert sorted(id(x) for x in out) == sorted(ids_in)
    # Original order unchanged
    assert [r["i"] for r in m] == [0,1,2,3,4]

def test_empty_and_singleton_are_handled():
    assert shuffle_manifest([], seed=123) == []
    one = [{"rel_path":"a.jpg"}]
    out = shuffle_manifest(one, seed=123)
    assert out == one and out is not one  # copied list, same item

# ----- ensure_item_content_hashes test -----
def test_hashes_files_and_mutates_in_place(tmp_path):
    a = tmp_path / "a.bin"; a.write_bytes(b"hello")
    b = tmp_path / "b.bin"; b.write_bytes(b"\x00\x01\x02")

    items = [
        {"abs_path": str(a), "rel_path": "a.bin", "is_corrupt": False},
        {"abs_path": str(b), "rel_path": "b.bin", "is_corrupt": True},  # still hashable
    ]
    out = ensure_item_content_hashes(items)
    assert out is items
    assert items[0]["hash_content"] == Hash().hash_bytes(b"hello")
    assert items[1]["hash_content"] == Hash().hash_bytes(b"\x00\x01\x02")


def test_preserves_existing_nonempty_hash(tmp_path):
    p = tmp_path / "c.bin"; p.write_bytes(b"data")
    items = [{"abs_path": str(p), "rel_path": "c.bin", "is_corrupt": False, "hash_content": "already"}]
    out = ensure_item_content_hashes(items)
    assert out is items
    assert items[0]["hash_content"] == "already"


def test_sets_none_when_abs_path_missing():
    items = [{"rel_path": "nope.bin", "is_corrupt": False}]
    ensure_item_content_hashes(items)
    assert items[0]["hash_content"] is None


def test_sets_none_on_hash_error(tmp_path, monkeypatch):
    p = tmp_path / "d.bin"; p.write_bytes(b"x")
    items = [{"abs_path": str(p), "rel_path": "d.bin", "is_corrupt": False}]

    import blase.extracting.image_backend as mod

    def boom(self, _path):
        raise IOError("boom")

    # Patch the method used by ensure_item_content_hashes
    monkeypatch.setattr(mod.Hash, "hash_file", boom, raising=True)

    from blase.extracting.image_backend import ensure_item_content_hashes
    ensure_item_content_hashes(items)

    assert items[0]["hash_content"] is None

# ----- compute_manifest_root_hash test -----
def test_invalid_mode_raises(tmp_path):
    with pytest.raises(ValueError):
        compute_manifest_root_hash([], str(tmp_path), "**/*.jpg", True, 42, "bogus")


def test_content_mode_requires_hash_content(tmp_path):
    bad = [{"rel_path": "a.jpg"}]
    with pytest.raises(ValueError):
        compute_manifest_root_hash(bad, str(tmp_path), "**/*.jpg", True, 0, "content")


def test_pms_mode_requires_field(tmp_path):
    bad = [{"rel_path": "a.jpg"}]
    with pytest.raises(ValueError):
        compute_manifest_root_hash(bad, str(tmp_path), "**/*.jpg", True, 0, "path_mtime_size")


def test_ordering_independent_of_input(tmp_path):
    m1 = _content_items()  # a,b
    m2 = list(reversed(_content_items()))  # b,a
    h1 = compute_manifest_root_hash(m1, str(tmp_path), "**/*.jpg", True, 123, "content")
    h2 = compute_manifest_root_hash(m2, str(tmp_path), "**/*.jpg", True, 123, "content")
    assert h1 == h2  # sorted by rel_path inside the function


def test_scan_params_affect_hash_each(tmp_path):
    m = _content_items()
    base = compute_manifest_root_hash(m, str(tmp_path), "**/*.jpg", True, 42, "content")

    # different directory
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    assert compute_manifest_root_hash(m, str(other_dir), "**/*.jpg", True, 42, "content") != base

    # different pattern
    assert compute_manifest_root_hash(m, str(tmp_path), "*.jpg", True, 42, "content") != base

    # different recursive flag
    assert compute_manifest_root_hash(m, str(tmp_path), "**/*.jpg", False, 42, "content") != base

    # different seed
    assert compute_manifest_root_hash(m, str(tmp_path), "**/*.jpg", True, 7, "content") != base

    # different mode (requires other fields)
    m_pms = _pms_items()
    assert compute_manifest_root_hash(m_pms, str(tmp_path), "**/*.jpg", True, 42, "path_mtime_size") != base


def test_seed_none_vs_same_seed(tmp_path):
    m = _content_items()
    h_none = compute_manifest_root_hash(m, str(tmp_path), "**/*.jpg", True, None, "content")
    h_42a  = compute_manifest_root_hash(m, str(tmp_path), "**/*.jpg", True, 42, "content")
    h_42b  = compute_manifest_root_hash(m, str(tmp_path), "**/*.jpg", True, 42, "content")
    assert h_42a == h_42b
    assert h_none != h_42a  # None is distinct from any int


def test_ignores_extra_fields(tmp_path):
    m1 = _content_items()
    m2 = [dict(x, width=100, height=200, junk="x") for x in _content_items()]
    h1 = compute_manifest_root_hash(m1, str(tmp_path), "**/*.jpg", True, 1, "content")
    h2 = compute_manifest_root_hash(m2, str(tmp_path), "**/*.jpg", True, 1, "content")
    assert h1 == h2


def test_empty_manifest_hashes_on_scan_identity_only(tmp_path):
    h1 = compute_manifest_root_hash([], str(tmp_path), "**/*.jpg", True, None, "content")
    h2 = compute_manifest_root_hash([], str(tmp_path), "**/*.jpg", True, None, "content")
    assert h1 == h2
    # Changing any scan parameter changes the root
    h3 = compute_manifest_root_hash([], str(tmp_path), "*.png", True, None, "content")
    assert h3 != h1

# ----- plan_image_batches test -----
def test_manual_fixed_chunks_and_is_last_flags():
    m = [rec(10, 10, 3), rec(5, 5, 3), rec(2, 2, 3), rec(1, 1, 3), rec(4, 4, 3)]
    plan = plan_image_batches(
        manifest=m,
        mode="manual",
        batch_size=2,
        target_batch_bytes=None,
        safety_margin=0.15,
        max_item_decoded_bytes=None,
        max_side=None,
    )
    # 3 batches: [0,1], [2,3], [4]
    assert [len(b["items"]) for b in plan] == [2, 2, 1]
    totals = [b["est_decoded_bytes"] for b in plan]
    assert totals[0] == est_bytes(10,10,3) + est_bytes(5,5,3)
    assert totals[1] == est_bytes(2,2,3) + est_bytes(1,1,3)
    assert totals[2] == est_bytes(4,4,3)
    assert [b["is_last"] for b in plan] == [False, False, True]


def test_auto_greedy_packing_respects_cap_and_flushes():
    # Choose sizes to illustrate flush: cap = 90
    # items with estimates 30, 50, 40 → batches: [30,50]=80 then [40]
    m = [
        rec(10, 1, 3),   # 30
        rec(10, 5, 1),   # 50
        rec(8, 5, 1),    # 40
    ]
    plan = plan_image_batches(
        manifest=m,
        mode="auto",
        batch_size=None,
        target_batch_bytes=100,
        safety_margin=0.10,   # cap=90
        max_item_decoded_bytes=None,
        max_side=None,
    )
    assert [len(b["items"]) for b in plan] == [2, 1]
    assert [b["est_decoded_bytes"] for b in plan] == [80, 40]
    assert [b["is_last"] for b in plan] == [False, True]


def test_auto_single_oversized_item_forms_own_batch():
    # cap=90, item est=120 > cap, should be own batch
    big = rec(40, 1, 3)  # 120
    small = rec(10, 1, 3)  # 30
    plan = plan_image_batches(
        manifest=[big, small],
        mode="auto",
        batch_size=None,
        target_batch_bytes=100,
        safety_margin=0.10,  # cap=90
        max_item_decoded_bytes=None,
        max_side=None,
    )
    assert [len(b["items"]) for b in plan] == [1, 1]
    assert [b["est_decoded_bytes"] for b in plan] == [120, 30]
    assert [b["is_last"] for b in plan] == [False, True]


def test_auto_hard_count_ceiling_applies():
    # Even if bytes allow, batch_size limits count
    m = [rec(10, 1, 3) for _ in range(5)]  # each 30 bytes
    plan = plan_image_batches(
        manifest=m,
        mode="auto",
        batch_size=2,          # ceiling
        target_batch_bytes=1000,
        safety_margin=0.0,
        max_item_decoded_bytes=None,
        max_side=None,
    )
    assert [len(b["items"]) for b in plan] == [2, 2, 1]
    assert plan[-1]["is_last"] is True
    assert all(b["is_last"] is False for b in plan[:-1])


def test_max_side_downscale_affects_estimate():
    # 4000x2000 RGB with max_side=1000 → scale=0.25 → 1000x500x3 = 1_500_000
    m = [rec(4000, 2000, 3)]
    plan = plan_image_batches(
        manifest=m,
        mode="auto",
        batch_size=None,
        target_batch_bytes=10_000_000,
        safety_margin=0.0,
        max_item_decoded_bytes=None,
        max_side=1000,
    )
    assert plan[0]["est_decoded_bytes"] == 1_500_000


def test_max_item_decoded_bytes_caps_estimate():
    # Raw est 10_000_000 → capped to 1_000_000
    m = [rec(2000, 2000, 3)]
    plan = plan_image_batches(
        manifest=m,
        mode="auto",
        batch_size=None,
        target_batch_bytes=10_000_000,
        safety_margin=0.0,
        max_item_decoded_bytes=1_000_000,
        max_side=None,
    )
    assert plan[0]["est_decoded_bytes"] == 1_000_000


def test_missing_dims_use_precomputed_estimate():
    m = [rec(None, None, None, est=1234)]
    plan = plan_image_batches(
        manifest=m,
        mode="auto",
        batch_size=None,
        target_batch_bytes=10_000_000,
        safety_margin=0.0,
        max_item_decoded_bytes=None,
        max_side=None,
    )
    assert plan[0]["est_decoded_bytes"] == 1234


def test_empty_manifest_returns_empty():
    plan = plan_image_batches(
        manifest=[],
        mode="auto",
        batch_size=None,
        target_batch_bytes=1_000_000,
        safety_margin=0.15,
        max_item_decoded_bytes=None,
        max_side=None,
    )
    assert plan == []


def test_input_validation_errors():
    m = [rec(1, 1, 1)]
    with pytest.raises(ValueError):
        plan_image_batches(m, mode="weird", batch_size=None, target_batch_bytes=None,
                           safety_margin=0.1, max_item_decoded_bytes=None, max_side=None)
    with pytest.raises(ValueError):
        plan_image_batches(m, mode="manual", batch_size=0, target_batch_bytes=None,
                           safety_margin=0.1, max_item_decoded_bytes=None, max_side=None)
    with pytest.raises(ValueError):
        plan_image_batches(m, mode="auto", batch_size=None, target_batch_bytes=0,
                           safety_margin=0.1, max_item_decoded_bytes=None, max_side=None)
    with pytest.raises(ValueError):
        plan_image_batches(m, mode="auto", batch_size=None, target_batch_bytes=100,
                           safety_margin=1.0, max_item_decoded_bytes=None, max_side=None)
        
# ----- decode_batch_pil test -----
def test_raises_if_pillow_missing(monkeypatch, tmp_path):
    # Ensure PIL is not imported anywhere in this module
    sys.modules.pop("PIL", None)
    sys.modules.pop("PIL.Image", None)
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, package=None: None if name == "PIL" else object(),
                        raising=True)
    with pytest.raises(RuntimeError):
        decode_batch_pil([{"abs_path": str(tmp_path / "x.jpg")}], return_type="np")


def test_np_rgb_basic(tmp_path):
    p = tmp_path / "a.jpg"
    _make_img(p, size=(10, 6), mode="RGB", color=(5, 6, 7))
    arr = decode_batch_pil([{"abs_path": str(p)}], return_type="np", color="rgb")[0]
    assert arr is not None and arr.shape == (6, 10, 3) and arr.dtype == np.uint8

def test_np_gray_conversion(tmp_path):
    p = tmp_path / "b.png"
    _make_img(p, size=(7, 9), mode="RGB", color=(200, 100, 50))
    arr = decode_batch_pil([{"abs_path": str(p)}], return_type="np", color="gray")[0]
    assert arr.ndim == 2 and arr.shape == (9, 7) and arr.dtype == np.uint8

def test_max_side_downscale(tmp_path):
    p = tmp_path / "c.jpg"
    _make_img(p, size=(400, 100), mode="RGB", color=(0, 0, 0))
    arr = decode_batch_pil([{"abs_path": str(p)}], return_type="np", color="rgb", max_side=200)[0]
    assert arr.shape == (50, 200, 3)

def test_missing_path_yields_none(tmp_path):
    pytest.importorskip("PIL")
    assert decode_batch_pil([{"abs_path": ""}], return_type="np") == [None]

def test_corrupt_file_returns_none(tmp_path):
    pytest.importorskip("PIL")
    p = tmp_path / "not_image.bin"
    p.write_bytes(b"\x00\x01\x02\x03")
    assert decode_batch_pil([{"abs_path": str(p)}], return_type="np") == [None]

def test_tensor_requires_torch(monkeypatch, tmp_path):
    orig_find = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, package=None: None if name == "torch" else orig_find(name, package),
                        raising=True)
    p = tmp_path / "e.jpg"
    _make_img(p, size=(5, 4), mode="RGB", color=(9, 9, 9))
    with pytest.raises(RuntimeError):
        decode_batch_pil([{"abs_path": str(p)}], return_type="tensor")

@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_tensor_happy_path(tmp_path):
    import torch  # noqa: F401
    p = tmp_path / "f.jpg"
    _make_img(p, size=(3, 2), mode="RGB", color=(7, 7, 7))
    t = decode_batch_pil([{"abs_path": str(p)}], return_type="tensor")[0]
    import torch as _torch
    assert isinstance(t, _torch.Tensor) and t.dtype == _torch.uint8 and t.shape[-1] in (1, 3)

# ----- decode_batch_cv2 test -----
def test_rgb_basic_shape_and_color(tmp_path):
    p = tmp_path / "a.png"
    _rgb_png(p, w=3, h=2, rgb=(30, 20, 10))
    out = decode_batch_cv2([{"abs_path": str(p)}], return_type="np", color="rgb", max_side=None)
    arr = out[0]
    assert arr is not None and arr.shape == (2, 3, 3) and arr.dtype == np.uint8
    # First pixel should be RGB (30,20,10) after BGR->RGB conversion
    assert tuple(int(x) for x in arr[0, 0]) == (30, 20, 10)


def test_gray_basic(tmp_path):
    p = tmp_path / "b.png"
    _gray_png(p, w=4, h=3, val=77)
    out = decode_batch_cv2([{"abs_path": str(p)}], return_type="np", color="gray")
    arr = out[0]
    assert arr.ndim == 2 and arr.shape == (3, 4) and int(arr[0, 0]) == 77


def test_max_cv2_side_downscale(tmp_path):
    p = tmp_path / "c.png"
    _rgb_png(p, w=400, h=100, rgb=(1, 2, 3))
    out = decode_batch_cv2([{"abs_path": str(p)}], return_type="np", color="rgb", max_side=200)
    arr = out[0]
    assert arr.shape == (50, 200, 3)  # 0.5 scale on max side


def test_missing_cv2_path_yields_none(tmp_path):
    assert decode_batch_cv2([{"abs_path": ""}], return_type="np") == [None]


def test_corrupt_cv2_file_returns_none(tmp_path):
    p = tmp_path / "not_image.bin"
    p.write_bytes(b"\x00\x01\x02\x03")
    assert decode_batch_cv2([{"abs_path": str(p)}], return_type="np") == [None]


def test_pil_cv2_return_type(tmp_path):
    pytest.importorskip("PIL")
    p = tmp_path / "d.png"
    _rgb_png(p, w=8, h=8, rgb=(9, 8, 7))
    out = decode_batch_cv2([{"abs_path": str(p)}], return_type="pil", color="rgb")
    import PIL.Image as PImage
    assert isinstance(out[0], PImage.Image)


def test_tensor_cv2_requires_torch(monkeypatch, tmp_path):
    orig_find = importlib.util.find_spec

    def fake_find(name, package=None):
        if name == "torch":
            return None
        return orig_find(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find, raising=True)
    p = tmp_path / "e.png"
    _rgb_png(p, w=5, h=4, rgb=(1, 1, 1))
    with pytest.raises(RuntimeError):
        decode_batch_cv2([{"abs_path": str(p)}], return_type="tensor")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_tensor_cv2_happy_path(tmp_path):
    import torch  # noqa: F401
    p = tmp_path / "f.png"
    _rgb_png(p, w=3, h=2, rgb=(7, 7, 7))
    t = decode_batch_cv2([{"abs_path": str(p)}], return_type="tensor")[0]
    import torch as _torch
    assert isinstance(t, _torch.Tensor) and t.dtype == _torch.uint8 and t.shape[-1] in (1, 3)

# ----- iter_batches_from_plan test -----
def test_empty_plan_yields_nothing():
    assert list(iter_batches_from_plan([])) == []

def test_marks_only_last_true_when_all_false():
    plan = [
        {"items": [1], "est_decoded_bytes": 10, "is_last": False},
        {"items": [2], "est_decoded_bytes": 20, "is_last": False},
        {"items": [3], "est_decoded_bytes": 30, "is_last": False},
    ]
    out = list(iter_batches_from_plan(plan))
    assert [b["is_last"] for b in out] == [False, False, True]

def test_corrects_multiple_true_flags():
    plan = [
        {"items": [1], "est_decoded_bytes": 10, "is_last": True},
        {"items": [2], "est_decoded_bytes": 20, "is_last": True},
    ]
    out = list(iter_batches_from_plan(plan))
    assert [b["is_last"] for b in out] == [False, True]

def test_missing_key_raises():
    bad = [{"items": [], "est_decoded_bytes": 0}]  # no is_last
    with pytest.raises(ValueError):
        list(iter_batches_from_plan(bad))

def test_wrong_types_raises_items_not_list():
    bad = [{"items": "x", "est_decoded_bytes": 0, "is_last": False}]
    with pytest.raises(ValueError):
        list(iter_batches_from_plan(bad))

def test_wrong_types_raises_est_not_int():
    bad = [{"items": [], "est_decoded_bytes": "0", "is_last": False}]
    with pytest.raises(ValueError):
        list(iter_batches_from_plan(bad))

def test_shallow_copy_dict_but_same_items_list():
    items0 = [{"rel_path": "a.jpg"}]
    plan = [
        {"items": items0, "est_decoded_bytes": 1, "is_last": False},
        {"items": [{"rel_path": "b.jpg"}], "est_decoded_bytes": 2, "is_last": True},
    ]
    out = list(iter_batches_from_plan(plan))
    # dict objects are new
    assert out[0] is not plan[0]
    # items list is the same object (shallow copy semantics)
    assert out[0]["items"] is items0
    # normalized flags: only last True
    assert [b["is_last"] for b in out] == [False, True]

# ----- compute_batch_hash test -----
def test_deterministic_and_order_sensitive():
    root = _hc("root")
    items_ab = [{"rel_path": "a.jpg", "hash_content": _hc("A")},
                {"rel_path": "b.jpg", "hash_content": _hc("B")}]
    items_ba = [{"rel_path": "b.jpg", "hash_content": _hc("B")},
                {"rel_path": "a.jpg", "hash_content": _hc("A")}]

    h1 = compute_batch_hash(root, items_ab)
    h2 = compute_batch_hash(root, items_ab)
    h3 = compute_batch_hash(root, items_ba)

    assert isinstance(h1, str) and len(h1) > 0
    assert h1 == h2
    assert h1 != h3  # order matters


def test_empty_items_allowed_and_stable():
    root = _hc("root")
    h1 = compute_batch_hash(root, [])
    h2 = compute_batch_hash(root, [])
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) > 0


def test_missing_hash_content_raises():
    root = _hc("root")
    items = [{"rel_path": "a.jpg"}]  # no 'hash_content'
    with pytest.raises(ValueError):
        compute_batch_hash(root, items)


def test_invalid_root_hash_raises():
    with pytest.raises(ValueError):
        compute_batch_hash("", [])
    with pytest.raises(ValueError):
        compute_batch_hash(None, [])  # type: ignore[arg-type]


def test_rel_path_affects_hash():
    root = _hc("root")
    hc_same = _hc("BYTES")
    h1 = compute_batch_hash(root, [{"rel_path": "x/a.jpg", "hash_content": hc_same}])
    h2 = compute_batch_hash(root, [{"rel_path": "y/a.jpg", "hash_content": hc_same}])
    assert h1 != h2


def test_hash_content_affects_hash():
    root = _hc("root")
    h1 = compute_batch_hash(root, [{"rel_path": "a.jpg", "hash_content": _hc("A")}])
    h2 = compute_batch_hash(root, [{"rel_path": "a.jpg", "hash_content": _hc("B")}])
    assert h1 != h2