"""Run with python -m grounded_memory.pipeline --help. All outputs are JSON."""

import argparse
import json
import os
import sqlite3
from pathlib import Path

from . import collection, datasets, deployment, evaluation, experiments, labeling, service
from .contracts import Annotation, Collected, DatasetSpec, Strategy
from .storage import Ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            os.getenv("GM_DATA_DIR", str(Path.home() / "Library/Application Support/Grounded Memory"))
        )
        / "pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    listing = sub.add_parser("list")
    listing.add_argument("kind", choices=sorted(service.KINDS))
    listing.add_argument("--offset", type=int, default=0)
    listing.add_argument("--limit", type=int, default=25)
    get = sub.add_parser("get")
    get.add_argument("id")
    collect = sub.add_parser("collect")
    collect.add_argument("file", type=Path)
    imp = sub.add_parser("import")
    imp.add_argument("--source", type=Path, required=True)
    imp.add_argument("--limit", type=int, default=500)
    suggest = sub.add_parser("suggest")
    suggest.add_argument("id")
    label = sub.add_parser("label")
    label.add_argument("id")
    label.add_argument("file", type=Path)
    policy = sub.add_parser("exclude")
    policy.add_argument("id")
    policy.add_argument("--actor", required=True)
    policy.add_argument("--restore", action="store_true")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("file", type=Path)
    experiment = sub.add_parser("experiment")
    experiment.add_argument("dataset_id")
    experiment.add_argument("--config", type=Path)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("experiment_id")
    evaluate.add_argument("--split", choices=["validation", "test", "regression"], default="validation")
    promote = sub.add_parser("promote")
    promote.add_argument("experiment_id")
    promote.add_argument("evaluation_ids", nargs="+")
    promote.add_argument("--actor", required=True)
    rollback = sub.add_parser("rollback")
    rollback.add_argument("deployment_id")
    rollback.add_argument("--actor", required=True)
    recall = sub.add_parser("recall")
    recall.add_argument("query")
    recall.add_argument("--as-of", type=float)
    args = parser.parse_args()
    ledger = Ledger(args.root)

    def read_file():
        return json.loads(args.file.read_text())

    try:
        match args.command:
            case "status":
                result = service.status(ledger)
            case "list":
                result = service.inventory(ledger, args.kind, args.limit, args.offset)
            case "get":
                result = ledger.get(args.id)
            case "collect":
                result = collection.collect(ledger, Collected.model_validate(read_file()))
            case "import":
                if not 1 <= args.limit <= 2000:
                    raise ValueError("Import limit must be 1–2000")
                result = collection.import_memory(ledger, args.source, args.limit)
            case "suggest":
                result = labeling.suggest(ledger, args.id)
            case "label":
                result = labeling.annotate(ledger, args.id, Annotation.model_validate(read_file()))
            case "exclude":
                labeling.set_policy(ledger, args.id, args.restore, args.actor)
                result = dict(saved=True)
            case "prepare":
                result = datasets.prepare(ledger, DatasetSpec.model_validate(read_file()))
            case "experiment":
                result = experiments.create(
                    ledger,
                    args.dataset_id,
                    Strategy.model_validate(json.loads(args.config.read_text()) if args.config else {}),
                )
            case "evaluate":
                result = evaluation.evaluate(ledger, args.experiment_id, args.split)
            case "promote":
                result = deployment.promote(ledger, args.experiment_id, args.evaluation_ids, args.actor)
            case "rollback":
                result = deployment.rollback(ledger, args.deployment_id, args.actor)
            case "recall":
                result = deployment.recall(ledger, args.query, as_of=args.as_of)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, KeyError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"Pipeline: {exc}\n")


if __name__ == "__main__":
    main()
