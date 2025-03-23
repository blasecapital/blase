from typing import Dict, Optional


class Performance:
    """
    Tracks system and process-level performance metrics during each step of the `blase` workflow.

    This utility class captures key runtime metrics to help users monitor resource utilization,
    identify bottlenecks, and ensure efficient training and data processing workflows. It is
    designed to work alongside the `Track` module, logging relevant stats automatically or on demand.

    Scope:
    ------
    Includes:
    - Wall-clock timing of step-level functions or batches.
    - CPU and RAM usage (average and peak) using `psutil`.
    - Optional GPU usage (if available) using `nvidia-smi` or `pynvml`.
    - Optional baseline comparison of prior runs from `Track` log directories.

    Excludes (for now):
    - Low-level function profiling (line-level)
    - Runtime heatmaps or interactive visualizations
    - Bottleneck detection and trace analysis

    Methods:
    --------
    - start_timer(): Begin timing a process.
    - end_timer(): End timing and store duration.
    - get_duration(): Return duration between start and end.
    - get_cpu_usage(): Return CPU utilization stats.
    - get_memory_usage(): Return RAM usage stats.
    - get_gpu_usage(): Return GPU usage if available.
    - get_summary(): Return all collected performance stats as a dictionary.
    - compare_runs(log_path_1, log_path_2): Compare metrics between two prior tracked runs.

    Example:
    --------
    >>> perf = Performance(step_name="train")
    >>> perf.start_timer()
    >>> train_model(...)
    >>> perf.end_timer()
    >>> summary = perf.get_summary()
    >>> print(summary)

    Integration:
    ------------
    - Used internally by `Track` and optionally within any main `blase` module.
    - Automatically stores output under each run's `/logs/` directory.
    - CLI-compatible via the `--performance` flag in the `blase` command line interface.

    Notes:
    ------
    - GPU stats require NVIDIA-compatible devices and `nvidia-smi` or `pynvml`.
    - This module is lightweight and safe to use in both local and headless workflows.
    """

    def __init__(self, step_name: Optional[str] = None, log_path: Optional[str] = None):
        """
        Initialize the performance tracker.

        Args:
            step_name (str): Optional name of the ML step (e.g., 'train', 'prepare').
            log_path (str): Optional path to save performance metrics as JSON.
        """
        pass

    def start_timer(self) -> None:
        """Start the timer for tracking wall-clock duration."""
        pass

    def end_timer(self) -> None:
        """Stop the timer and store elapsed time internally."""
        pass

    def get_duration(self) -> float:
        """
        Return the duration (in seconds) between start and end.

        Returns:
            float: Elapsed time in seconds.
        """
        pass

    def get_cpu_usage(self) -> Dict[str, float]:
        """
        Get CPU utilization statistics.

        Returns:
            Dict[str, float]: Average and peak CPU usage during the step.
        """
        pass

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get RAM usage statistics.

        Returns:
            Dict[str, float]: Current, peak, and average memory usage (MB).
        """
        pass

    def get_gpu_usage(self) -> Optional[Dict[str, float]]:
        """
        Get GPU usage statistics, if available.

        Returns:
            Optional[Dict[str, float]]: GPU utilization and VRAM stats (MB), or None if unavailable.
        """
        pass

    def get_summary(self) -> Dict[str, any]:
        """
        Aggregate and return all performance metrics collected during the step.

        Returns:
            Dict[str, any]: Summary dictionary of all performance data.
        """
        pass

    def save_summary(self, file_path: Optional[str] = None) -> None:
        """
        Save the summary performance report to a JSON file.

        Args:
            file_path (str): Path to save the summary. Defaults to `log_path/performance.json`.
        """
        pass

    def compare_runs(self, log_path_1: str, log_path_2: str) -> Dict[str, Dict[str, float]]:
        """
        Compare two performance logs side-by-side.

        Args:
            log_path_1 (str): Path to the first performance JSON log.
            log_path_2 (str): Path to the second performance JSON log.

        Returns:
            Dict[str, Dict[str, float]]: Metric deltas between the two runs.
        """
        pass
