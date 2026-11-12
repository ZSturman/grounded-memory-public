"""Independent evaluator with per-case evidence and a fixed deterministic baseline."""

import resource
import time

from .contracts import Strategy
from .labeling import usable
from .retrieval import retrieve
from .storage import code_version, digest

CATEGORIES = {"accuracy", "retrieval", "temporal", "contradiction", "consolidation", "grounding", "privacy"}
BASELINE = Strategy()


def measure(dataset, config, split, *, deployment_corpus=False):
    records = [r for r in dataset["records"] if deployment_corpus or r["split"] == split]
    cases = [c for c in dataset["spec"]["cases"] if c["split"] == split]
    if not cases:
        raise ValueError("This split has no evaluation cases")
    results = []
    for case in cases:
        response = retrieve(
            records,
            case["query"],
            config,
            as_of=case["as_of"]
            if case["as_of"] is not None
            else (None if case["include_history"] else dataset["spec"]["evaluation_time"]),
            allowed_sources=case["allowed_sources"],
            include_history=case["include_history"],
        )
        actual = {identity for hit in response["hits"] for identity in hit["evidence_ids"]}
        expected = set(case["expected_ids"])
        correct = len(actual & expected)
        results.append(
            dict(
                case_id=case["id"],
                query=case["query"],
                categories=case["categories"],
                expected_ids=sorted(expected),
                retrieved_ids=sorted(actual),
                exact=(
                    actual == expected
                    and response["uncertain"] == case["expected_uncertain"]
                    and (
                        case["expected_hit_count"] is None
                        or len(response["hits"]) == case["expected_hit_count"]
                    )
                ),
                precision=correct / len(actual) if actual else float(not expected),
                recall=correct / len(expected) if expected else float(not actual),
                response=response,
            )
        )
    metrics = {key: sum(r[key] for r in results) / len(results) for key in ("precision", "recall", "exact")}
    metrics["cases"] = len(results)
    metrics["by_category"] = {
        category: sum(r["exact"] for r in results if category in r["categories"])
        / sum(category in r["categories"] for r in results)
        for category in sorted({c for r in results for c in r["categories"]})
    }
    return dict(metrics=metrics, cases=results)


def evaluate(ledger, experiment_id, split="validation"):
    with ledger.stage("evaluation"):
        if split not in ("validation", "test", "regression"):
            raise ValueError(
                "Evaluation requires validation, test, or regression; training examples are not a test"
            )
        experiment = ledger.get(experiment_id, "experiment")
        if experiment["code_version"] != code_version():
            raise ValueError("Code changed; create a new candidate before evaluation")
        dataset = ledger.get(experiment["dataset_id"], "dataset")
        usable(ledger, dataset)
        start, cpu = time.perf_counter(), time.process_time()
        candidate = measure(dataset, Strategy.model_validate(experiment["config"]), split)
        baseline = measure(dataset, BASELINE, split)
        semantic = dict(candidate=candidate, baseline=baseline)
        if split in ("test", "regression"):
            semantic["deployment_candidate"] = measure(
                dataset, Strategy.model_validate(experiment["config"]), split, deployment_corpus=True
            )
            semantic["deployment_baseline"] = measure(dataset, BASELINE, split, deployment_corpus=True)
        return ledger.append(
            "evaluation",
            dict(
                experiment_id=experiment_id,
                dataset_id=dataset["id"],
                code_version=experiment["code_version"],
                split=split,
                **semantic,
                result_digest=digest(semantic),
                elapsed_seconds=time.perf_counter() - start,
                cpu_seconds=time.process_time() - cpu,
                process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                rss_units="bytes on macOS, KiB on Linux; process lifetime peak, not allocation delta",
            ),
        )
