# `blase` Scope and Workflow Vision
*Use to review the main blase modules during planning and development.*

## Setup
### Clone Repo
```sh
cd <your preferred install directory>
git clone https://github.com/blasecapital/blase.git
```

### Create venv
```sh
python3 -m venv .env
source .env/bin/activate
pip install .
```

### Create Project From Template
```sh
cd <preferred project directory>
blase create <template name> <project name>
```

### Navigate to Root
```sh
cd <project directory>
```

### Modify Main.py, /data, /scripts, etc. In Project

```sh
# main.py

import argparse

from blase import Extract, Transform, Load, Examine, Clean, Prepare, Train, Evaluate, Deploy, Monitor, Update, Track
from scripts.engineer import example_func, filter_bad_images
from scripts.model import model


RAW_IMAGE_DIR = "data/raw/images"
PROCESSED_DIR = "data/interim/processed_images"


def etl():
    """
    This example shows a minimal ETL workflow using blase.
    
    - Loads images in batches from a raw directory
    - Applies a user-defined transformation (`example_func`)
    - Saves the output to a structured intermediate directory
    - Logs everything to enable reproducibility via `Track`
    """
    extractor = Extract()
    transformer = Transform()
    loader = Load()

    for batch in extractor.load_images(directory=RAW_IMAGE_DIR):
        transformed_data = transformer.apply_function(batch, example_func)
        loader.save_to_filesystem(transformed_data, directory=PROCESSED_DIR)
    
def inspect():
    """
    Aggregates and visualizes statistics across batches of raw input.
    """
    examiner = Examine(aggregate_batches=True)
    extractor = Extract()
    for batch in extractor.load_images(directory=RAW_IMAGE_DIR):
        examiner.process_batch(batch)
    examiner.generate_summary()
    
def clean():
    """
    Applies custom filters to identify and remove unwanted data, storing exclusion keys.
    """
    cleaner = Clean()
    extractor = Extract()

    for batch in extractor.load_images(directory=RAW_IMAGE_DIR):
        cleaner.apply_filter(batch, filter_bad_images)
        cleaner.remove_keys(batch)

    cleaner.clear_keys()
        
def prepare():
    """
    Combines feature and target sources, applies splits and rolling windows,
    and writes .npy training files.
    """
    prepare = Prepare()
    extractor = Extract()

    prepare.register_source("features", extractor.load_images(directory=RAW_IMAGE_DIR))
    prepare.register_source("targets", extractor.load_images(directory=RAW_IMAGE_DIR))
    prepare.load_exclusion_keys("cleaned_v1")
    prepare.set_split(train=0.7, val=0.2, test=0.1, seed=42)
    prepare.convert(output_format="npy", output_dir="./data/prepared/")

def train():
    """
    Configures and runs model training using the specified deep learning framework.
    Saves model and training logs for reproducibility.
    """
    trainer = Train()

    trainer.set_model(model, framework="tensorflow")
    trainer.configure_training(optimizer="adam", loss_fn="categorical_crossentropy")
    trainer.set_mode(mode="fine_tune", checkpoint_path="models/base_model.h5", freeze_layers=["conv1", "conv2"])
    trainer.train(epochs=10, data_dir=RAW_IMAGE_DIR)
    trainer.save_model("models/trained_model.h5")
    trainer.save_logs("logs/train_run_001.json")

def evaluate():
    """
    Evaluates model performance on prepared test data.
    Supports saving predictions and logs for review.
    """
    evaluator = Evaluate()

    evaluator.evaluate_live(
        model_path="models/my_model.h5",
        test_data_dir="data/prepared/test/",
        task_type="classification",
        save_preds=True
    )
    evaluator.save_evaluation_logs("logs/eval_run_001.json")
    
def deploy():
    """
    Packages the trained model for deployment and archives it.
    Supports Docker and other target environments.
    """
    deployer = Deploy()

    deployer.deploy_model(
        model=model,
        save_path="deployments/my_model_v2/",
        method="docker",
        framework="tensorflow",
        include_api=True
    )
    deployer.archive_deployment("deployments/my_model_v2/", format="zip")
    
def monitor():
    """
    Monitors live data, checks for drift, tracks metrics, and triggers updates.
    """
    monitor = Monitor()

    monitor.load_feature_priors("priors/features.json")
    monitor.load_prediction_priors("priors/predictions.json")

    for features, preds, true in monitor.stream_batches():
        monitor.submit_live_batch(features, preds, true)
        monitor.track_performance_metrics()
        monitor.detect_feature_drift(features)

        if monitor.check_triggers():
            print("Retraining condition met!")

def update():
    """
    Automatically incorporates new data, applies transformations, retrains,
    and optionally re-evaluates. Works with versioned base model.
    """
    updater = Update()

    updater.run(
        base_model_path='models/v1/',
        new_data_path='raw_data/',
        update_mode='append_only',
        transform_fn='transforming.clean_features',
        evaluate_after=True
    )

def main():
    """
    Command-line interface for running individual steps of a blase machine learning pipeline.

    This function enables modular execution of key ML pipeline steps—such as data extraction,
    preparation, training, evaluation, and deployment—via command-line arguments. It is designed
    for CLI use only and expects `--step` to be specified to select which module to run.

    Tracking Integration:
    ---------------------
    All core blase modules (e.g., Extract, Train, Evaluate) include built-in tracking. By default,
    each module logs its step metadata using the `Track` system internally.

    This function also supports manual run control via the following optional flags:
    - `--start_run`: Begins a new run and creates a log directory for the session.
    - `--end_run`: Finalizes the run and ensures all logs, scripts, and artifacts are captured.

    These allow you to wrap multiple CLI steps under one logical run session.

    Example usage:
    --------------
    Run a single step with full tracking:
        $ python main.py --step train --start_run --end_run

    Run steps in stages while keeping them under the same tracked run:
        $ python main.py --step etl --start_run
        $ python main.py --step train
        $ python main.py --step evaluate --end_run

    Notes:
    ------
    - This CLI should be executed from the root of the project directory.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=[
        "etl", "inspect", "clean", "prepare", "train", "evaluate", "deploy", "monitor", "update"
    ], required=True)
    
    parser.add_argument("--start_run", action="store_true", help="Start a tracked run (creates log folders, etc.)")
    parser.add_argument("--end_run", action="store_true", help="Finalize a tracked run (writes final logs)")

    args = parser.parse_args()

    step_mapping = {
        "etl": etl,
        "inspect": inspect,
        "clean": clean,
        "prepare": prepare,
        "train": train,
        "evaluate": evaluate,
        "deploy": deploy,
        "monitor": monitor,
        "update": update
    }

    # Track instance (project name could be dynamic later)
    track = Track(project="example")
    
    # Start run if requested
    if args.start_run:
        track.start_run()
    
    try:
        step_mapping[args.step]()
    finally:
        # End run only if it was requested and step completes
        if args.end_run:
            track.end_run()

if __name__ == "__main__":
    main()

```

### CLI Workflow After Project Configuration
```sh
# Creates and closes a unique 'run' directory in /runs
# /runs tracks your project states
python main.py --step train --start_run --end_run
```

```sh
# Add to existing run
python main.py --step prepare
```

```sh
# Explicitly wrap multiple steps in one run
python main.py --step etl --start_run
# then
python main.py --step train
# finally
python main.py --step evaluate --end_run
```
