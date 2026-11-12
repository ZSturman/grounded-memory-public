"""Disposable complete workflow; no models, network, or personal data. Three PASS lines."""

import json
import tempfile
from pathlib import Path

from grounded_memory.pipeline import collection, datasets, deployment, evaluation, experiments, labeling
from grounded_memory.pipeline.contracts import Annotation, Collected, DatasetSpec, Strategy
from grounded_memory.pipeline.storage import Ledger


def main():
    fixture = json.loads(
        (Path(__file__).resolve().parents[1] / "tests/fixtures/pipeline/regression-v1.json").read_text()
    )
    with tempfile.TemporaryDirectory(prefix="grounded-pipeline-") as root:
        ledger = Ledger(root)
        for record in fixture["records"]:
            collection.collect(ledger, Collected.model_validate(record["raw"]))
            labeling.annotate(ledger, record["raw"]["id"], Annotation.model_validate(record["annotation"]))
        spec = DatasetSpec.model_validate(fixture["spec"])
        dataset = datasets.prepare(ledger, spec)
        assert datasets.prepare(ledger, spec) == dataset
        print("PASS collection → independent review → reproducible dataset")
        candidate = experiments.create(ledger, dataset["id"], Strategy(algorithm="coverage", min_score=0.5))
        results = [evaluation.evaluate(ledger, candidate["id"], split) for split in ("test", "regression")]
        assert all(r["candidate"]["metrics"]["exact"] == 1 for r in results)
        assert (
            evaluation.evaluate(ledger, candidate["id"], "test")["result_digest"]
            == results[0]["result_digest"]
        )
        print("PASS independent final evaluation → repeatable results")
        approved = deployment.promote(ledger, candidate["id"], [r["id"] for r in results], "smoke-reviewer")
        deployment.rollback(ledger, approved["id"], "smoke-reviewer")
        assert deployment.recall(ledger, "Garage key")["hits"]
        print("PASS explicit promotion → grounded recall → rollback (temporary data only)")


if __name__ == "__main__":
    main()
