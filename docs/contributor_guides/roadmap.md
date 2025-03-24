# Blase Roadmap

## The Problem Blase Solves

Machine learning workflows are often optimized for enterprise-scale infrastructure, assuming:
- Datasets too large for memory **must go to the cloud**
- Users have dedicated MLOps teams to handle experiment tracking and deployment
- Reproducibility is optional or handled externally

But for many practitioners — solo researchers, domain experts, and consultants — the reality is:
- Their datasets are large enough to strain memory (10GB–250GB), but too small to justify cloud costs
- They work on laptops or desktops with limited resources but still require **structured, reproducible training**
- They often revisit experiments iteratively and need **flexible but traceable workflows**
- Most tools are either too lightweight (e.g., notebooks) or too heavyweight (e.g., managed MLOps platforms)

**Blase was created for this underserved middle ground.**

It brings memory-aware data handling, reproducible pipelines, and a consistent API to **local-first machine learning development** — no cloud dependency, no vendor lock-in.

---

## Who Blase Is For

Blase is built for **individuals and small teams** who want serious workflows without managing infrastructure.

**Primary user profiles:**

| Title | Description |
|-------|-------------|
| **The Clinical Researcher / Health Data Scientist** | Working with patient-level tabular data or medical images; privacy and local reproducibility are crucial. |
| **The Solo Data Scientist / Consultant** | Running end-to-end ML pipelines on a personal workstation, sometimes across multiple client domains. |
| **The Domain-Heavy Data Scientist** | Quant researchers, environmental scientists, energy modelers — who know their data well and want clean training workflows. |
| **The Independent Researcher / Hobbyist** | Building models at home or for publication, not supported by enterprise-scale MLOps tooling. |
| **The Applied ML Learner / Career Changer** | Learning by doing — needs a way to standardize workflows, document results, and build a reproducible portfolio. |
| **The Research Lab (No MLOps Team)** | Small teams collaborating locally, focused on research goals, not tooling overhead. |

All of these users typically have:
- Local access to **10GB–250GB datasets**
- **1TB of storage**, a **GPU**, and no access to distributed compute clusters
- A need for **training control**, **reproducibility**, and **portability**

## Comparison: How Blase Differs from Other ML Pipeline Tools

Blase is optimized for **local-first, reproducible ML workflows** — filling the gap between notebook-based experimentation and cloud-native orchestration frameworks. Here’s how it compares to several major ML pipeline tools:

| Feature / Aspect | **Blase** (Local ML Pipeline) | **Metaflow** (Netflix) | **Kedro** (QuantumBlack) | **ZenML** (Orchestrator-Agnostic) | **Flyte** (Lyft) | **Kubeflow Pipelines** |
|------------------|-------------------------------|-------------------------|---------------------------|----------------------------------|------------------|------------------------|
| **Local Development** | Native. Runs all steps on a single machine with no setup. Designed for local workflows from day one. | Easy to start locally with Python decorators. | Structured local pipelines using a Data Catalog. | Local-first, can later scale with orchestrator switch. | Sandbox mode exists but dev needs local K8s. | Requires K8s even for local runs (e.g., mini-clusters). |
| **Scalability** | Disk-limited scaling with batching & chunking. Perfect for up to ~250GB on desktop. | Hybrid scaling via AWS/K8s. | Plugin-based scaling (e.g., Airflow/Spark). | Infrastructure-agnostic via orchestrators. | Highly scalable, built for distributed DAGs. | K8s-native; scales with pods and cloud infra. |
| **Data Handling** | Out-of-core batching; ideal for "fits-on-disk, not-in-RAM" use cases. Local-only by default. | Uses step-level artifact storage (e.g., S3). | Uses Data Catalog config. Needs plugins for big data. | Uses artifact store. Chunking requires manual logic. | Type-aware, references big data, user must chunk. | Component data handled via shared volumes/blobs. |
| **Pipeline Orchestration** | Sequential. End-to-end steps are callable and tracked. No DAG overhead. | Python class-based step DAGs. | Node graph, parallelizable via runners. | DAGs built from Python steps, orchestrator-agnostic. | Full DAG orchestration on K8s with retries, UIs, etc. | Containerized DAGs via Argo on K8s. |
| **Ease of Use** | Easy. One-command setup (e.g., via Docker), pure Python APIs. Geared toward solo practitioners. | Simple to start; scales with AWS skills. | Moderate. Requires learning pipeline/node structure. | Simple to start, modular as needed. | Steep learning curve; requires infra experience. | Steep. Requires K8s/Docker fluency. |
| **Best For** | Researchers, consultants, domain scientists with medium-sized datasets and local workflows. | Hybrid teams needing simple local dev + cloud scale. | Structured projects with dev best practices. | Flexibility across teams, infra, or orchestrators. | Engineering-heavy teams with K8s expertise. | Cloud-native orgs with infra budgets and MLOps teams. |

## Milestones

Blase’s development is divided into iterative, functional milestones designed to ensure a stable, reproducible, and memory-efficient machine learning pipeline system from day one. This roadmap prioritizes core pipeline generalization, reliable tracking, and developer ergonomics.

### v0.1.0 – Core Modules + Template Foundations (In Progress)

**Goals:**
- Implement all main modules with empty method bodies and docstrings
- Finalize data batching and memory-safe extraction logic
- Design tracking integration across all steps
- Prepare project template with:
  - Standard `/scripts`, `/data`, `/models`, `/outputs`, `/runs` structure
  - Example training script
  - `Track` instance baked into workflows

**Checklist:**
- [ ] Draft core module outlines (Extract, Transform, Load, etc.)
- [ ] Draft `Track` logging module with public + protected methods
- [ ] CLI-enabled project generator (e.g., `blase create`)
- [ ] Basic project template with script-based usage
- [ ] Implement Extract class with working chunked loading backends
- [ ] Implement Transform with dynamic + standard function support
- [ ] Implement Prepare to create ready-to-train `.npy` or `.tfrecord` files
- [ ] Write unit tests for hashing, tracking, and resource monitors
- [ ] Write integration test that mimics a full structured pipeline

---

### v0.2.0 – Functional ML Training & Evaluation

**Goals:**
- Add TensorFlow and PyTorch support
- Enable metadata logging during training and evaluation
- Track saved models and training configs

**Planned Tasks:**
- [ ] Build `Train.set_model` and `Train.train` workflows (TensorFlow first)
- [ ] Implement `Evaluate.evaluate_live` with built-in metrics
- [ ] Save predictions + logs during `Evaluate`
- [ ] Save reproducible model artifacts with `Deploy`

---

### v0.3.0 – Full Run Reproducibility

**Goals:**
- Tie each step’s data, config, and model to a hash-based snapshot
- Add rollback and reproduction tooling
- Write project-based cache and checkpoint manager

**Planned Tasks:**
- [ ] `Track` step tracing from any saved run
- [ ] Hash user scripts used in `Transform` and `Train`
- [ ] Restore data pipeline up to a previous run
- [ ] Diff tools to compare two runs or outputs

---

### v0.4.0 – Post-Deployment Monitoring & Update Workflows

**Goals:**
- Implement basic monitoring for drift and performance degradation
- Add triggers for retraining workflows via `Update`

**Planned Tasks:**
- [ ] Aggregate live metrics and plot with `Monitor`
- [ ] Implement `Update.run()` for incremental learning
- [ ] Include agent interface (WIP design) for guided workflows

---

### v0.5.0 – Packaging, Distribution, and Plugins

**Goals:**
- Add optional plugins (cloud extractors, DVC-like versioning)
- Publish `blase` to PyPI
- Polish CLI ergonomics and help docs

**Planned Tasks:**
- [ ] Optional support for AWS/GCP extractors
- [ ] CLI improvements (`blase run`, `blase inspect`, `blase deploy`)
- [ ] `README`, badges, and contribution templates

---

### v1.0.0 – General Release: Stability, Wishlist Integration, and Polishing

Blase's `v1.0.0` milestone marks its transition from a structured early-stage pipeline framework to a **mature, production-grade tool** for solo practitioners and applied ML developers. This version prioritizes stability, reproducibility, and ease of use, while addressing user feedback and incorporating the most impactful wishlist features from the community.

### Goals

- Finalize a robust interface that supports **end-to-end ML workflows**.
- Ensure **minimal friction** for both CLI and programmatic usage.
- Bake in **strong reproducibility guarantees**, including hashing, logging, and metadata capture.
- Close out **critical bugs and usability gaps** found in pre-1.0 versions.
- Deliver on **wishlist features** that extend Blase into agentic workflows and no-code interfaces.
- Maintain **blazing-fast local performance**, while enabling cloud extensibility.

**Planned Features & Tasks**

#### Bug Fixing & Stability
- Address any critical issues identified during `v0.5.0` and community testing.
- Finalize the integration tests that simulate realistic ETL → Train → Evaluate → Deploy cycles.
- Harden project generation, hash reproducibility, and CLI experience.

#### Wishlist Feature Integration

*Possibly add-ons post v1.0.0*

- **Agentic Workflows**
  - Add optional agent module powered by an LLM API (e.g., OpenAI, Ollama).
  - Include customizable autonomy levels (e.g., "manual mode", "semi-auto", "fully autonomous").
  - Let the agent explore logs, recommend next steps, and execute pipeline components on command.

- **Plugin System**
  - Define plugin spec for user-defined modules in `extract`, `transform`, `train`, `monitor`.
  - Load plugins dynamically and integrate with project YAML configuration.

- **No-Code Interface (GUI)**
  - Offer a simple web-based or Electron-based GUI to construct pipelines.
  - Drag-and-drop support for connecting steps, adding data, previewing logs and metrics.

- **Multi-Project Dashboard**
  - Create a dashboard to visualize and compare tracked runs across projects.
  - Include filters by model, date, hash, performance, etc.
  - Link directly to artifacts, metrics, and logs for inspection or rollback.

- **Built-in Schema Suggestion**
  - Auto-inspect loaded tabular data (from `Extract`) and suggest:
    - Feature types (categorical, continuous)
    - Imputation or normalization suggestions
    - Label column candidates
    - Outlier detection warnings

### Testing & Finalization

- Full **project-template compatibility tests** for:
  - Structured data (e.g., CSV)
  - Unstructured data (images, audio, JSON)
  - Reinforcement learning simulation use cases

- Ensure **all default templates and CLI tools** support:
  - Tracking with `Track`
  - Reproduction using `Hash`
  - Exporting and packaging models with `Deploy`

> `v1.0.0` will serve as the **baseline for long-term backward compatibility**, with patches and minor versions (`1.x.x`) focused on refinement and expanded community support.
