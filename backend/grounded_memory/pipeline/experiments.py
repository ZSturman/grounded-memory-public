"""Candidate configurations are frozen; no online memories or weights are modified."""

import platform
import sqlite3
import sys

from .contracts import Strategy
from .labeling import usable
from .storage import code_version


def create(ledger, dataset_id, config: Strategy):
    with ledger.stage("experiment"):
        dataset = ledger.get(dataset_id, "dataset")
        usable(ledger, dataset)
        return ledger.append(
            "experiment",
            dict(
                dataset_id=dataset_id,
                config=config.model_dump(),
                code_version=code_version(),
                strategy_version="lexical-v1",
                environment=dict(
                    python=sys.version, platform=platform.platform(), sqlite=sqlite3.sqlite_version
                ),
                training="none; frozen deterministic retrieval configuration",
                optimization_splits=["train", "validation"],
            ),
        )
