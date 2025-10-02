from pathlib import Path
import json
import gzip
from typing import Union


def open_writer(path: Path, compression: str):
    try:
        import tensorflow as tf
    except ImportError:

        class _JSONLWriter:
            def __init__(self, p: Path, comp: str):
                self._fh = (
                    gzip.open(p, "wt", encoding="utf-8")
                    if comp == "GZIP"
                    else open(p, "w", encoding="utf-8")
                )

            def write(self, example_bytes: Union[bytes, dict]):
                if isinstance(example_bytes, (bytes, bytearray)):
                    # allow make_example JSON fallback (rare)
                    self._fh.write(json.dumps({"_raw": len(example_bytes)}) + "\n")
                else:
                    self._fh.write(json.dumps(example_bytes) + "\n")

            def close(self):
                self._fh.close()

        path.parent.mkdir(parents=True, exist_ok=True)
        return _JSONLWriter(path, compression)

    # TensorFlow TFRecordWriter
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = tf.io.TFRecordOptions(
        compression_type=("GZIP" if compression == "GZIP" else None)
    )

    class _TFWriter:
        def __init__(self, p: Path):
            self._w = tf.io.TFRecordWriter(str(p), options=opts)

        def write(self, example_bytes: bytes):
            self._w.write(example_bytes)

        def close(self):
            self._w.close()

    return _TFWriter(path)
