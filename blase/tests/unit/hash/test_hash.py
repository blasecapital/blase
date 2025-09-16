import os
import re
import sys
import stat
import pytest
from pathlib import Path

# Assume your class is exposed as blase.hashing.Hash (adjust imports as needed)
from blase.utils.hashing import Hash

HEX_RE = re.compile(r"^[0-9a-f]+$")


@pytest.fixture()
def hasher():
    return Hash()


@pytest.fixture()
def tmp_text_file(tmp_path: Path):
    p = tmp_path / "hello.txt"
    p.write_text("hello world\n", encoding="utf-8")
    return p


@pytest.fixture()
def tmp_bin_file(tmp_path: Path):
    p = tmp_path / "blob.bin"
    p.write_bytes(b"\x00\x01\x02\xff" * 1024)  # 4KB
    return p


def test_hash_bytes_deterministic(hasher):
    data = b"same"
    h1 = hasher.hash_bytes(data)
    h2 = hasher.hash_bytes(data)
    assert h1 == h2
    assert isinstance(h1, str)
    assert HEX_RE.match(h1)


def test_hash_bytes_changes_with_content(hasher):
    assert hasher.hash_bytes(b"a") != hasher.hash_bytes(b"b")
    assert hasher.hash_bytes(b"") != hasher.hash_bytes(b"a")  # empty is distinct


def test_hash_bytes_large_payload(hasher):
    data = b"abc123" * 1_000_00  # ~600 KB, cheap but non-trivial
    h = hasher.hash_bytes(data)
    assert isinstance(h, str) and HEX_RE.match(h)


def test_hash_file_deterministic(hasher, tmp_text_file):
    h1 = hasher.hash_file(tmp_text_file)
    h2 = hasher.hash_file(str(tmp_text_file))  # str vs Path
    assert h1 == h2
    assert HEX_RE.match(h1)


def test_hash_file_binary(hasher, tmp_bin_file):
    h1 = hasher.hash_file(tmp_bin_file)
    assert HEX_RE.match(h1)


def test_hash_file_matches_hash_bytes_when_same_content(hasher, tmp_path):
    data = b"content-123"
    p = tmp_path / "x.bin"
    p.write_bytes(data)
    assert hasher.hash_file(p) == hasher.hash_bytes(data)


def test_hash_file_missing_raises(hasher, tmp_path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(Exception):
        hasher.hash_file(missing)


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Windows chmod semantics differ; test not meaningful.",
)
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="Running as root; permission checks won’t fail under root.",
)
def test_hash_file_permission_error(hasher, tmp_path: Path):
    p = tmp_path / "secret.txt"
    p.write_text("top secret")
    os.chmod(p, stat.S_IWUSR)  # owner write-only
    try:
        with pytest.raises(Exception):
            hasher.hash_file(p)
    finally:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)


def test_hash_file_directory_raises(hasher, tmp_path: Path):
    # Hashing a directory should fail clearly
    with pytest.raises(Exception):
        hasher.hash_file(tmp_path)


def test_hash_file_symlink_same_as_target(hasher, tmp_text_file, tmp_path):
    link = tmp_path / "link.txt"
    # Some CI (esp. Windows) may not support symlinks without admin—skip if not supported
    try:
        link.symlink_to(tmp_text_file)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    assert hasher.hash_file(link) == hasher.hash_file(tmp_text_file)


def test_line_endings_change_hash(hasher, tmp_path):
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(b"one\ntwo\n")
    crlf.write_bytes(b"one\r\ntwo\r\n")
    # Content bytes differ → hashes should differ
    assert hasher.hash_file(lf) != hasher.hash_file(crlf)


def test_non_ascii_and_null_bytes(hasher, tmp_path):
    p = tmp_path / "utf8.bin"
    data = "π≈3.14159".encode("utf-8") + b"\x00\x00"
    p.write_bytes(data)
    assert hasher.hash_file(p) == hasher.hash_bytes(data)


@pytest.mark.parametrize(
    "payloads",
    [
        [b"a", b"b", b"c", b"d"],
        [b"\x00" * 32, b"\x00" * 31 + b"\x01"],
    ],
)
def test_basic_collision_sanity(hasher, payloads):
    # Not a proof of collision resistance—just a sanity net against regressions
    hashes = {hasher.hash_bytes(x) for x in payloads}
    assert len(hashes) == len(payloads)
