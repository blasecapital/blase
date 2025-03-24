# blase
## blase: Local-first machine learning pipelines, made reproducible.

## What is Blase?

`blase` (*'blāz'*) is a modular machine learning framework built for single-machine environments and medium-sized datasets (10GB–250GB). It removes cloud dependency by default and offers a full pipeline for extracting, transforming, training, and deploying models — **all with built-in batching and reproducibility**.

You control the architecture, features, and training flow — Blase handles the rest:  
- Batch loading of datasets larger than RAM  
- Clean modular interfaces for each ML step  
- Persistent tracking of arguments, scripts, and artifacts for full reproducibility  
- Local-first, production-ready output for any ML use case  

**One project directory. One command. One reproducible pipeline.**

## Table of Contents
- [Features](#features)
- [Roadmap](#roadmap)
- [Setup](#setup)
- [Who blase Is For](#who-blase-is-for)
- [Contribute](#contribute)
- [License](#license)
- [About](#about)

## Features

- **Works on your machine** – No need for distributed systems or cloud orchestration
- **End-to-end pipeline modules** – From `Extract()` to `Deploy()`
- **Supports any ML framework** – TensorFlow, PyTorch, XGBoost, RL, etc.
- **Handles data larger than RAM** – Extract and load data in chunks
- **Track every experiment** – Automatically logs scripts, arguments, and hashes with `Track()`
- **Use only what you need** – Every module is decoupled, so you can use just ETL, just Train, or the full pipeline
- **Comes with templates** – Start a full project with one CLI command
- **Addresses the Reproducibility Crisis** - Model training is complex and has many steps. It is easy to overlook best practices and documentation, evidenced by the [reproducibility crisis](https://reproducible.cs.princeton.edu/)

Check out the scope and workflow vision [here](./docs/working_example_dev.md)

## Roadmap
`blase` is under active development. You can view the full roadmap [here](./docs/contributor_guides/roadmap.md)

Key priorities:
- Core pipeline modules with memory-aware design  
- End-to-end reproducibility via `Track()`  
- Local-first tooling for real ML workflows (classification, regression, RL, etc.)  
- Model packaging and monitoring tools for post-deployment workflows  
- Monolithic-first design, outlined [here](./docs/contributor_guides/architecture.md)
- Make function compatible with all datasets stored in [tests](./blase/tests/fixtures/data/)

Explicit workflow compatibility: 

*Each of the following has data stored in /blase/tests/fixtures/data/ and corresponding integration tests.*

- Healthcare:
    - Readmission prediction (classification)
    - Disease detection with images (classification)
- Finance:
    - Sentiment detection with news/filings (classification, language-based analysis)
    - Trade prediction (timeseries, classification)
Home price estimation (regression)
- Environmental Science:
    - Species identification (classification, fine tune a pre-trained model)
    - Weather prediction (regression/multi-class classification, timeseries)
- Industry:
    - Defect detection (computer vision, unsupervised)
    - Machine failure from vibration data (unsupervised, tabular data)
- Generative/Language:
    - Fine-tune/create small language model
    - Image generation
- Reinforcement Learning:
    - Gaming env

## Setup
### Recommended steps using Docker:
**Prerequisites**
1. [Install Docker](https://www.docker.com/products/docker-desktop/)
    - Enable WSL2 Backend if using Windows - Follow [this guide](https://learn.microsoft.com/en-us/windows/wsl/install)
2. [Install VS Code](https://code.visualstudio.com/download) or Docker-compatible IDE
3. Install Docker and Python extensions in VS Code

**Run 'balcones_train'**
1. Clone the repository
   ```sh
   cd <your preferred install directory>
   git clone https://github.com/blasecapital/blase.git
   ```
2. Open VS Code and navigate to the 'blase' directory in 'Explorer'
3. Right-click on 'docker-compose.yml' and select 'Compose Up' to build the container
4. Navigate to the Docker extension, right-click on the 'blase_env' under 'Containers,' and select 'Attach Visual Studio Code'
5. Navigate to the /workspace folder in the 'Explorer' tab to view the project directory

### Pip Install Alternative:
**Prerequisites:**
1. **CUDA & cuDNN:**  
   - Install **CUDA 11.2** and **cuDNN 8.1** (latest compatible version for Windows).  
   - Follow this [installation guide](https://youtu.be/hHWkvEcDBO0?si=3xxz4VfhOVcnri8E). 

**Steps:**
1. Clone the repository
    ```sh
    cd <your preferred install directory>
    git clone https://github.com/blasecapital/blase.git
    ```
2. Create python environment
    - Windows:
        ```sh
        python -m venv C:\path\to\new\virtual\environment
        C:\path\to\new\virtual\environment\Scripts\activate
        ```
    - macOS/Linux:
        ```sh
        python -m venv ~/path/to/new/virtual/environment
        source ~/path/to/new/virtual/environment/bin/activate
        ```
3. Pip install:
    ```sh
    pip install .
    or
    pip install .[full]
    ```
4. Verify installation:
    ```sh
    python -c "import blase; print(blase.__version__)"
    ```
5. If you are using PyTorch with GPU, run the following to install the compatible CUDA version:
    ```sh
    pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
    ```

## Who `blase` Is For

Blase is built for **individuals and small teams** who want serious ML workflows without managing infrastructure.

| Title | Example Use |
|-------|-------------|
| **The Clinical Researcher / Health Data Scientist** | Working with patient-level tabular data or medical images. |
| **The Solo Data Scientist / Consultant** | Running end-to-end pipelines on a workstation, across domains. |
| **The Domain-Heavy Data Scientist** | Quant, energy, or environmental analysts seeking clean, fast pipelines. |
| **The Independent Researcher** | Building reproducible models offline, at home or for publication. |
| **The Applied ML Learner** | Standardizes learning and project documentation for career growth. |
| **The Research Lab (No MLOps Team)** | Collaborating locally with no infra team support. |


## Contribute
This project is **community-driven** and still early stage. We welcome contributors! Check out the discussions and issues pages for open tasks and goals. [Roadmaps](./docs/contributor_guides/roadmap.md) details the path to v1.0.0.

Lets build `blase` to improve modeling in high-impact areas like:
- Healthcare
- Environmental science
- Risk management
- Local-first AI

## License
This project is licensed under the BSD 3-Clause License.

### Third-Party Dependencies
This project relies on open-source dependencies specified in `requirements.txt`.  
Each dependency is subject to its own license, which can be found in their respective repositories.

## About

`blase` was created by Blase Capital Management, LLC as a general-purpose evolution of internal tooling [`balcones_train`](https://github.com/blasecapital/balcones_train) — with the goal of empowering reproducible machine learning for solo practitioners and small teams.
