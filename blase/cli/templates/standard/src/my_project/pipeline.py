import argparse
import blase

from .transforms import func_pandas


def etl(track=None):
    extractor = blase.Extract()
    transformer = blase.Transform()
    loader = blase.Load()

    for batch, is_last, _ in extractor.read_csv(
        file_path="data/House_Rent_Dataset.csv",
        backend="pandas",
        mode="manual",
        batch_size=500,
        track=track,
    ):
        out = transformer.apply_function(
            data=batch, transform_func=func_pandas, last_batch=is_last, track=track
        )
        loader.save_to_csv(
            data=out,
            file_name="house_rent.csv",
            backend="pandas",
            last_batch=is_last,
            track=track,
        )


def train(track=None):
    # example stub – your to_tfrecord + training steps here
    pass


def restore(target: str, kind: str):
    blase.Restore.to(target, node_type=kind, policy=blase.RestorePolicy())


def cli():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("etl")
    sub.add_parser("train")
    r = sub.add_parser("restore")
    r.add_argument("--target", required=True, help="data_hash|step_hash|dataset_id")
    r.add_argument("--kind", choices=["data", "step"], default="data")
    args = p.parse_args()

    track = blase.Track(project="my-project")
    track.start_run()  # auto-creates runs/<run_id>/nodes.db + per-run CAS

    try:
        if args.cmd == "etl":
            etl(track=track)
        elif args.cmd == "train":
            train(track=track)
        elif args.cmd == "restore":
            restore(args.target, args.kind)
    finally:
        track.end_run()
