import hashlib
import pickle
import inspect
from pathlib import Path
from typing import Any, Union


class Hash:
    """
    Utility class for generating unique hashes of code, data, files, and directories
    to support reproducibility, change detection, and rollback within the `blase` framework.

    `Hash` enables the `Track` system to uniquely identify and reference everything that contributes
    to a machine learning pipeline — from custom functions and config files to model weights and
    preprocessed data artifacts.

    This allows users to:
    - Verify whether pipeline components have changed.
    - Reproduce exact versions of past experiments.
    - Avoid redundant computation by detecting hash collisions.
    - Audit model dependencies across disjointed runs.

    Features:
    ---------
    - Hashes Python functions and classes using source code introspection.
    - Hashes file contents (e.g., CSVs, model files, preprocessed batches) using SHA-256.
    - Hashes serialized objects (e.g., NumPy arrays, configs, dictionaries) for consistency tracking.
    - Recursively hashes directories for end-to-end snapshot integrity.
    - Supports hash comparison and validation across pipeline runs.

    Use Cases:
    ----------
    - Log a function's hash before training to track logic changes.
    - Detect changes to `.npy` or `.csv` inputs before reprocessing data.
    - Compare two tracked runs to determine if code or data changed.
    - Link artifacts to specific pipeline steps using their hash digests.

    Integration:
    ------------
    This utility is called automatically by the `Track` module and can also be used manually
    in custom workflows or analysis tools.

    Example:
    --------
    >>> from utils.hashing import Hash
    >>> hasher = Hash()

    >>> def preprocess(data): return data * 2
    >>> hasher.hash_function(preprocess)
    '9a8b3dcb1a43d...'

    >>> hasher.hash_file("data/batch_01.npy")
    '62ddf742f3a5a...'

    >>> hasher.hash_object({"learning_rate": 0.01, "layers": 3})
    'ab42ed3cb2...'

    Notes:
    ------
    - All hashes use SHA-256 unless otherwise specified.
    - Hashes are deterministic and stable across platforms and sessions.
    - Users are not expected to manage hashes directly, but may do so for advanced use.
    """

    def __init__(self, hash_algorithm: str = "sha256"):
        """
        Initialize the hash utility with the desired algorithm.

        Args:
            hash_algorithm (str): Hashing algorithm to use (default is 'sha256').
        """
        self.hash_algorithm = hash_algorithm

    def _hasher(self):
        """
        Create a new hash object using the initialized algorithm.

        Returns:
            hashlib.Hash: A new hash object.
        """
        return hashlib.new(self.hash_algorithm)

    def hash_function(self, func: Any) -> str:
        """
        Generate a hash based on the source code of a Python function or class.

        Args:
            func (Any): Python function or class object.

        Returns:
            str: Hash digest string.
        """
        source = inspect.getsource(func)
        h = self._hasher()
        h.update(source.encode('utf-8'))
        return h.hexdigest()

    def hash_file(self, file_path: Union[str, Path]) -> str:
        """
        Generate a hash for the contents of a file.

        Args:
            file_path (Union[str, Path]): Path to the file.

        Returns:
            str: Hash digest string.
        """
        h = self._hasher()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def hash_directory(self, directory_path: Union[str, Path]) -> str:
        """
        Recursively hash all files in a directory.

        Args:
            directory_path (Union[str, Path]): Directory path.

        Returns:
            str: Combined hash of all file contents.
        """
        h = self._hasher()
        directory = Path(directory_path)
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                h.update(path.name.encode('utf-8'))
                h.update(self.hash_file(path).encode('utf-8'))
        return h.hexdigest()

    def hash_object(self, obj: Any) -> str:
        """
        Generate a hash of any serializable Python object.

        Args:
            obj (Any): The object to hash (must be pickle-serializable).

        Returns:
            str: Hash digest string.
        """
        h = self._hasher()
        serialized = pickle.dumps(obj)
        h.update(serialized)
        return h.hexdigest()

    def compare_hashes(self, hash1: str, hash2: str) -> bool:
        """
        Compare two hash digests for equality.

        Args:
            hash1 (str): First hash.
            hash2 (str): Second hash.

        Returns:
            bool: True if hashes match, False otherwise.
        """
        return hash1 == hash2
