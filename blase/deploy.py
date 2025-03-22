from typing import Any, Callable, Dict, Optional


class Deploy:
    """
    Packages trained models, metadata, and dependencies into a structured, reproducible format
    ready for production deployment in a variety of environments.

    The `Deploy` module prepares all necessary components for serving a model—including the
    model file, dependency list, metadata, environment configurations, and optional scripts
    for cloud, container, and local deployments. Users can use built-in exporters or plug in
    custom logic to generate deployment artifacts.

    Core Responsibilities:
    ----------------------
    - Save the trained model in a specified format (TensorFlow, PyTorch, ONNX, etc.)
    - Generate deployment metadata (task type, inputs, outputs, framework, hash, training args)
    - Save a reproducible list of required packages (requirements.txt or pyproject.toml)
    - Hash the model and config to ensure reproducibility
    - Create a unified deployment directory that can be zipped, containerized, or shared
    - Allow user to specify the desired deployment method (e.g. "onnx", "flask", "docker", etc.)
    - Register custom deployment functions if special behavior is needed

    Built-In Deployment Options:
    ----------------------------
    - **Local Folder Deployment**: Save model, logs, and predict script for direct usage
    - **ONNX Export**: Convert supported models to ONNX format for compatibility
    - **Docker Export**: Generate a Dockerfile and docker-compose.yml from deployment config
    - **API Wrapper**: Auto-generate a `predict.py` FastAPI or Flask script for serving
    - **Self-Contained Archive**: Package everything into a `.zip` or `.tar.gz` archive

    Methods:
    --------
    deploy_model(
        model: Any,
        save_path: str,
        method: str = "local",
        framework: str = "tensorflow",
        metadata: Optional[Dict[str, Any]] = None,
        include_api: bool = False,
        include_docker: bool = False,
        include_requirements: bool = True
    )
        Main deployment entry point. Saves model, metadata, dependencies, and optional wrappers.

    export_onnx(model: Any, save_path: str) -> None
        Converts and saves a PyTorch or TensorFlow model in ONNX format.

    generate_api_wrapper(save_path: str, framework: str = "tensorflow") -> None
        Creates a Flask or FastAPI script to serve predictions via REST API.

    generate_docker_assets(save_path: str) -> None
        Generates Dockerfile and docker-compose.yml for container deployment.

    archive_deployment(save_path: str, format: str = "zip") -> None
        Archives the deployment directory into a zip or tarball.

    register_custom_deployer(name: str, deploy_fn: Callable) -> None
        Registers a user-defined function for a custom deployment strategy.

    Example:
    --------
    >>> deployer = Deploy()
    >>> deployer.deploy_model(
    >>>     model=model,
    >>>     save_path="deployments/my_model_v2/",
    >>>     method="docker",
    >>>     framework="tensorflow",
    >>>     include_api=True
    >>> )
    >>> deployer.archive_deployment("deployments/my_model_v2/", format="zip")

    Notes:
    ------
    - All deployment artifacts are saved in a standardized directory structure.
    - Reproducibility is ensured via hashing and stored logs.
    - Users may skip certain components (e.g., API, Docker) or supply their own logic via `register_custom_deployer`.
    - ONNX export is only supported for models with compatible architectures.
    - Metadata should include model inputs/outputs, preprocessing references, and training args.
    """

    def deploy_model(
        self,
        model: Any,
        save_path: str,
        method: str = "local",
        framework: str = "tensorflow",
        metadata: Optional[Dict[str, Any]] = None,
        include_api: bool = False,
        include_docker: bool = False,
        include_requirements: bool = True
    ) -> None:
        """
        Packages the model and supporting artifacts for deployment.
        """
        pass

    def export_onnx(
        self,
        model: Any,
        save_path: str,
        input_sample: Optional[Any] = None
    ) -> None:
        """
        Converts a compatible model to ONNX format.
        """
        pass

    def generate_api_wrapper(
        self,
        save_path: str,
        framework: str = "tensorflow",
        use_fastapi: bool = True
    ) -> None:
        """
        Generates a REST API wrapper script (FastAPI or Flask).
        """
        pass

    def generate_docker_assets(
        self,
        save_path: str,
        base_image: str = "python:3.10-slim"
    ) -> None:
        """
        Generates Dockerfile and docker-compose.yml for deployment.
        """
        pass

    def archive_deployment(
        self,
        save_path: str,
        format: str = "zip"
    ) -> None:
        """
        Archives the deployment directory into a zip or tarball.
        """
        pass

    def register_custom_deployer(
        self,
        name: str,
        deploy_fn: Callable[[Any, str, Optional[Dict[str, Any]]], None]
    ) -> None:
        """
        Registers a user-defined deployment function under a custom method name.
        """
        pass
