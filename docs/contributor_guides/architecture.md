# Blase Modularity Principles

This guide defines the modular architecture principles that guide Blase's design, helping contributors understand how modules are built, extended, and composed together.

Blase is **monolithic-first**, meaning that each module works out of the box with built-in logic for batching, logging, and reproducibility. Tracking via the `Track` module is baked in by default to ensure production readiness and iterative transparency — but users can **disable tracking** in any core module by setting `track=False`. This enables Blase to function as a lightweight toolkit for partial workflows like ETL pipelines, standalone training, or feature inspection, without unnecessary overhead.

This design balances **best-practice defaults** for serious projects with the **modularity and flexibility** needed for ad hoc experimentation or integration into other pipelines.

---

## 1. Monolithic-First with Internal Extensibility

> Each module works out of the box with strong defaults, but allows optional injection of user-defined behavior.

**Examples:**
```python
extractor.load_csv("data.csv")

# User customization through simple callables
transformer.apply_function(data, my_custom_cleaning_function)
```

Avoid enforcing subclassing early on:
```python
# Not preferred in v1
extractor = CustomExtractorSubclass()
```

---

## 2. Public Interface, Protected Internals

> Only expose public methods required by users. Helpers and setup logic should be marked with `_underscore` prefix to discourage misuse.

**Example:**
```python
class Extract:
    def load_csv(...):        # Public method
        ...
    def _calculate_memory():  # Internal helper
        ...
```

---

## 3. Callable Functions Over Subclassing

> Modules like `Transform` and `Clean` should take user-defined functions (or simple classes with `.transform()` methods) instead of requiring users to subclass.

**Example:**
```python
# Prefer this
def clean_fn(batch):
    return batch[batch["value"] > 0]

transformer.apply_function(data, clean_fn)
```

```python
# Class with transform method is okay too
class MyTransformer:
    def transform(self, batch): ...
```

---

## 4. Shared Protocols Over Rigid Interfaces

> Functions/classes passed into Blase modules must conform to **behavior expectations**, not inherit from specific classes.

**Example:**
```python
if not hasattr(custom_obj, "transform"):
    raise ValueError("Expected object with `.transform()` method.")
```

This supports unstructured code and lowers the barrier for contribution.

---

## 5. Batching, Logging, Hashing Are Default

> These are **strongly recommended** but optional. Every module:
- Assumes data is loaded/processed in batches (can auto-detect for full in-memory loading).
- Logs metadata using the `Track` class (default but can be disabled).
- Hashes user functions/data with the `Hash` utility for reproducibility (util for `Track`).

Users should never worry about forgetting to track their work — it’s built in.

---

## 6. User-Facing Code Is Explicit; Internals Use Smart Defaults

> User-facing APIs should be easy to read and declarative. Internals handle automatic chunking, hashing, and logging.

**Example:**
```python
# User code should be readable and direct
for batch in extractor.load_csv("file.csv"):
    for processed in transformer.apply_function(batch, clean_fn):
        loader.save_to_db(processed)
```

---

## 7. File Structure Mirrors Responsibility

Each main module (`extract.py`, `transform.py`, etc.) should live in the top-level directory. Supporting logic goes in a matching subdirectory.

**Structure:**
```
blase/
├── extract.py
├── extracting/
│   └── _csv_loader.py

├── transform.py
├── transforming/
│   └── transform_utils.py
```

This supports discoverability and separation of concerns.

---

## 8. Public API is Argument-Driven, Not Chain-Heavy

> Prefer declarative API design. Allow user configuration through clear method parameters, not chaining.

Good:
```python
trainer.set_model(model, framework="tensorflow")
trainer.configure_training(optimizer="adam", loss_fn="mse")
trainer.train(epochs=10)
```

Avoid:
```python
trainer.set().compile().train().run()
```

## 9. Modular Usage Patterns

Each core module in Blase is designed to function independently or as part of the full pipeline. Users can:

- Use `Extract`, `Transform`, and `Load` as a local-first data pipeline
- Start from `Prepare` and move into training workflows
- Skip early steps entirely if they have pre-cleaned, preprocessed data
- Build custom loops around `Evaluate` and `Deploy` for experimentation or production

This design supports various project maturities, from simple experiments to complex, reproducible pipelines.

| Workflow Type     | Modules Involved                             | Example Use Case                                                                 |
|-------------------|-----------------------------------------------|----------------------------------------------------------------------------------|
| **ETL**           | `Extract`, `Transform`, `Load`                | Preprocess tabular data for visualization or upload into a feature store        |
| **Data Inspection** | `Extract`, `Examine`     | Visualize image datasets for artifacts or class imbalance before training       |
| **Training**      | `Prepare`, `Train`, `Evaluate`, `Deploy`      | Train model using pre-engineered features             |
| **Post-Deploy**   | `Monitor`, `Update`                           | Track model drift on new financial time series and trigger fine-tuning          |


---

## Summary

Blase uses a **modular monolith** approach:
- Built-in logic for batching, logging, and tracking
- Extensible via functions and simple interfaces
- Internals are protected and testable
- Consistent user experience from start to finish

Contributors should build new modules with these principles in mind to maintain a clean and powerful developer experience.

---
