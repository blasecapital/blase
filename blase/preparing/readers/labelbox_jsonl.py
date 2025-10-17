import json
import gzip
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator
from urllib.parse import urlparse, unquote


def _basename(p: str) -> str:
    s = unquote(p)
    # handle URL or path
    if "://" in s:
        s = urlparse(s).path
    return Path(s).name


def _stem(p: str) -> str:
    return Path(_basename(p)).stem


def _image_id(rec: Dict[str, Any], id_from: str) -> str:
    dr = rec.get("data_row", {})
    if id_from == "external_id":
        return str(dr.get("external_id") or _basename(dr.get("row_data", "")))
    if id_from == "external_stem":
        return _stem(str(dr.get("external_id") or dr.get("row_data", "")))
    if id_from == "row_data_basename":
        return _basename(str(dr.get("row_data", "")))
    if id_from == "row_data_stem":
        return _stem(str(dr.get("row_data", "")))
    if id_from == "data_row_id":
        return str(dr.get("id"))
    raise ValueError(f"id_from={id_from} unsupported")


def _iter_records(uri: str) -> Iterator[Dict[str, Any]]:
    p = Path(uri)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif p.suffix.lower() in {".jsonl", ".ndjson"}:
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        # single JSON file or one JSON per line; try both
        txt = p.read_text(encoding="utf-8").strip()
        if txt.startswith("{") and txt.endswith("}"):
            yield json.loads(txt)
        else:
            for line in txt.splitlines():
                line = line.strip()
                if line:
                    yield json.loads(line)


def read(src: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    src["options"]:
      id_from: ...
      project_id: ...
      normalize_names: bool = True
      mode: "auto" | "detection" | "classification" = "auto"
    """
    opts = dict(src.get("options") or {})
    id_from = opts.get("id_from", "external_stem")
    project_id_filter = opts.get("project_id")
    normalize = bool(opts.get("normalize_names", True))
    mode = opts.get("mode", "auto")

    def want_det() -> bool:
        return mode in ("auto", "detection")

    def want_cls() -> bool:
        return mode in ("auto", "classification")

    def norm(s: str) -> str:
        return s.strip().lower().replace(" ", "_") if normalize else s

    for rec in _iter_records(src["uri"]):
        iid = _image_id(rec, id_from=id_from)
        projects = rec.get("projects") or {}
        for proj_id, proj in projects.items():
            if project_id_filter and proj_id != project_id_filter:
                continue
            for lab in proj.get("labels") or []:
                ann = lab.get("annotations") or {}

                # detection objects
                if want_det():
                    for obj in ann.get("objects") or []:
                        bb = obj.get("bounding_box")
                        if not bb:
                            continue
                        yield {
                            "image_id": iid,
                            "bbox": [
                                float(bb["left"]),
                                float(bb["top"]),
                                float(bb["width"]),
                                float(bb["height"]),
                            ],  # xywh_abs
                            "class": norm(
                                str(obj.get("value") or obj.get("name") or "unknown")
                            ),
                            "iscrowd": 0,
                        }

                # image-level classifications
                if want_cls():
                    for c in ann.get("classifications") or []:
                        ans = c.get("answer") or {}
                        if "value" in ans:  # Radio
                            yield {"image_id": iid, "class": norm(str(ans["value"]))}
                            continue
                        for a in c.get("answers") or []:  # Checklist
                            if "value" in a:
                                yield {"image_id": iid, "class": norm(str(a["value"]))}
