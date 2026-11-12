"""Explicit evaluation-gated activation and append-only rollback for reviewed recall only."""

import time

from .contracts import Strategy
from .evaluation import CATEGORIES
from .labeling import usable
from .retrieval import retrieve
from .storage import code_version


def active(ledger):
    items = ledger.list("deployment", limit=1)
    return items[0] if items else None


def validate_promotion(ledger, experiment_id, evaluation_ids):
    """Read-only readiness check shared with the actual promotion gate."""
    experiment = ledger.get(experiment_id, "experiment")
    if experiment["code_version"] != code_version():
        raise ValueError("Candidate code differs from the running code; reevaluate")
    dataset = ledger.get(experiment["dataset_id"], "dataset")
    usable(ledger, dataset)
    evaluations = [ledger.get(identity, "evaluation") for identity in evaluation_ids]
    if {e["split"] for e in evaluations} != {"test", "regression"}:
        raise ValueError("Promotion needs independent test AND fixed regression evaluations")
    for evaluation in evaluations:
        if evaluation["experiment_id"] != experiment_id or evaluation["dataset_id"] != dataset["id"]:
            raise ValueError("Evaluation does not belong to this candidate/dataset")
        pooled = evaluation["deployment_candidate"]["metrics"]
        pooled_baseline = evaluation["deployment_baseline"]["metrics"]
        if pooled["exact"] < 1 or any(
            pooled[k] < pooled_baseline[k] for k in ("precision", "recall", "exact")
        ):
            raise ValueError("Candidate must pass all curated cases against the full deployment corpus")
        metrics, baseline = evaluation["candidate"]["metrics"], evaluation["baseline"]["metrics"]
        if set(metrics["by_category"]) != CATEGORIES:
            raise ValueError("Both final suites must cover all seven evaluation categories")
        if metrics["exact"] < 1 or any(metrics[k] < baseline[k] for k in ("precision", "recall", "exact")):
            raise ValueError("Candidate must pass all curated final cases and not regress against baseline")
    return experiment, dataset


def promote(ledger, experiment_id, evaluation_ids, actor):
    with ledger.stage("deployment"):
        experiment, dataset = validate_promotion(ledger, experiment_id, evaluation_ids)
        previous = active(ledger)
        return ledger.append(
            "deployment",
            dict(
                experiment_id=experiment_id,
                dataset_id=dataset["id"],
                config=experiment["config"],
                code_version=experiment["code_version"],
                evaluation_ids=evaluation_ids,
                actor=actor,
                action="promote",
                previous_id=previous["id"] if previous else None,
                scope="reviewed-recall",
                synthetic=dataset["spec"]["purpose"] == "synthetic-regression",
            ),
        )


def rollback(ledger, deployment_id, actor):
    with ledger.stage("deployment"):
        target = ledger.get(deployment_id, "deployment")
        # Recheck source availability and executable compatibility; never silently run stale artifacts.
        usable(ledger, ledger.get(target["dataset_id"], "dataset"))
        if target["code_version"] != code_version():
            raise ValueError("Rollback target used different code; restore that code or reevaluate first")
        previous = active(ledger)
        return ledger.append(
            "deployment",
            dict(
                **{
                    k: v
                    for k, v in target.items()
                    if k not in ("id", "actor", "action", "previous_id", "rollback_to")
                },
                actor=actor,
                action="rollback",
                previous_id=previous["id"] if previous else None,
                rollback_to=target["id"],
            ),
        )


def recall(ledger, query, **options):
    deployed = active(ledger)
    if not deployed:
        raise ValueError("No reviewed-recall configuration is deployed")
    if deployed["code_version"] != code_version():
        raise ValueError("Running code changed; reevaluate the deployed configuration")
    dataset = ledger.get(deployed["dataset_id"], "dataset")
    usable(ledger, dataset)
    if options.get("as_of") is None and not options.get("include_history"):
        options["as_of"] = time.time()
    return dict(
        deployment_id=deployed["id"],
        dataset_id=dataset["id"],
        **retrieve(dataset["records"], query, Strategy.model_validate(deployed["config"]), **options),
    )
