# Installation Guide for `blase`

As of now, the installation process is simple and assumes a single unified environment. However, future versions will support multiple configurations for TensorFlow, PyTorch, and non-ML workflows, with prebuilt Docker containers for each.

---

## Current Setup

`blase` currently uses a **single Docker image** based on the following:

- **Base Image**: `nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04`
- **GPU**: CUDA 11.2 compatible (works on native Windows systems with NVIDIA GPUs)
- **Dependencies**: All major libraries bundled into one environment

### Core dependencies included:
- `tensorflow-gpu==2.5.0`
- `torch==1.10.1+cu111`, `torchvision`, `torchaudio`
- `pandas==1.1.5`, `numpy==1.19.5`
- `pyarrow`, `duckdb`, `polars`, `lance`
- `matplotlib`, `scikit-learn`, `lime`, `tqdm`, `pytest`, etc.

You can build this container with:

```
docker build -t blase:latest .
```

Or run it interactively:

```
docker run -it --gpus all blase:latest bash
```

> Note: This setup includes *all dependencies* in a single environment. It is ideal for development but not optimal for production or lightweight use cases.

---

## Forward-Looking Plan

Future versions of `blase` will be tied to **modular, versioned Docker images**, each suited for specific use cases:

### Planned Image Variants

| Image Tag                        | Description                               |
|----------------------------------|-------------------------------------------|
| `blase:<version>-tf-cuda11`      | TensorFlow environment w/ CUDA 11.2       |
| `blase:<version>-torch-cuda12`   | PyTorch environment w/ CUDA 12.x          |
| `blase:<version>-nonml`          | For data pipelines without ML frameworks  |
| `blase:<version>-dev`            | Full development build with all extras    |
| `blase:<version>-slim`           | Minimal build for inference or CI/CD use  |

These images will be published to [Docker Hub](https://hub.docker.com/u/blasecapital) and version-locked to each `blase` release. Users will be able to pull a specific image that matches their setup.

---

## 🔧 Easy Installation from the Command Line

Once versioned Docker images are available, installing and running `blase` will be as simple as:

### Using Docker:

```
# Pull a specific version
docker pull blase/blase:v0.1.0-tf-cuda11

# Run it interactively
docker run -it --gpus all blase/blase:v0.1.0-tf-cuda11 bash
```

### Using Docker Compose:

You can also use `docker-compose` for a more persistent development setup. Example:

```
version: '3.9'
services:
  blase:
    image: blase/blase:v0.1.0-torch-cuda12
    runtime: nvidia
    volumes:
      - .:/workspace/blase
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

Run it with:

```
docker-compose up --build
```

---

## Future Features

- A `docker_matrix.yaml` file to map `blase` versions to supported CUDA, OS, and ML framework combinations.
- CLI tool `blase configure` to help generate custom builds or `docker-compose` files.
- `requirements/` folder with partitioned dependencies for `tf`, `torch`, `nonml`, and `dev` workflows.

---

## Summary

| Installation Mode        | Recommended For                       |
|--------------------------|----------------------------------------|
| `docker build`           | Development and experimentation        |
| `docker run`             | Quick access to a specific setup       |
| `docker-compose`         | Persistent local development workflows |
| Future: Prebuilt images  | Easy onboarding, guaranteed compatibility |

For now, just clone the repo, build the default image, and you’re good to go.

Stay tuned for versioned container support and flexible runtime configurations!

---