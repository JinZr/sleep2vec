from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hypnodata.annotations import AnnotationResult
from hypnodata.config import CandidateSpec, HypnodataConfig, SignalSpec
from hypnodata.edf import EdfInventory, EdfSignalInfo
from hypnodata.records import RecordTask
from hypnodata.registry import resolve_callable


@dataclass(frozen=True)
class DefaultAdapter:
    def collect_records(self, config: HypnodataConfig) -> list[RecordTask]:
        raise NotImplementedError("record_discovery.type=custom requires adapter.collect_records(config).")

    def resolve_metadata(self, record: RecordTask, config: HypnodataConfig) -> dict[str, Any]:
        return {}

    def fix_header(
        self,
        record: RecordTask,
        inventories: dict[str, EdfInventory],
        config: HypnodataConfig,
    ) -> dict[str, EdfInventory]:
        return inventories

    def score_channel_candidate(
        self,
        record: RecordTask,
        canonical: str,
        spec: SignalSpec,
        candidate: CandidateSpec,
        signal: EdfSignalInfo,
        config: HypnodataConfig,
    ) -> float | None:
        return None

    def read_annotations(
        self,
        record: RecordTask,
        config: HypnodataConfig,
        duration_sec: float,
    ) -> AnnotationResult:
        return AnnotationResult()


def load_adapter(config: HypnodataConfig) -> Any:
    reference = config.record_discovery.adapter
    if reference is None:
        return DefaultAdapter()
    factory = resolve_callable(reference, {})
    adapter = factory(config)
    return DefaultAdapter() if adapter is None else adapter


def call_collect_records(adapter: Any, config: HypnodataConfig) -> list[RecordTask]:
    collector = getattr(adapter, "collect_records", None)
    if not callable(collector):
        raise ValueError("record_discovery.adapter must provide collect_records(config) for type=custom.")
    return list(collector(config))


def call_resolve_metadata(adapter: Any, record: RecordTask, config: HypnodataConfig) -> dict[str, Any]:
    resolver = getattr(adapter, "resolve_metadata", None)
    if not callable(resolver):
        return {}
    metadata = resolver(record, config)
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError("adapter.resolve_metadata must return a mapping.")
    return dict(metadata)


def call_fix_header(
    adapter: Any,
    record: RecordTask,
    inventories: dict[str, EdfInventory],
    config: HypnodataConfig,
) -> dict[str, EdfInventory]:
    fixer = getattr(adapter, "fix_header", None)
    if not callable(fixer):
        return inventories
    fixed = fixer(record, inventories, config)
    if fixed is None:
        return inventories
    if not isinstance(fixed, dict):
        raise ValueError("adapter.fix_header must return an inventory mapping.")
    return fixed


def call_score_channel_candidate(
    adapter: Any,
    record: RecordTask,
    canonical: str,
    spec: SignalSpec,
    candidate: CandidateSpec,
    signal: EdfSignalInfo,
    config: HypnodataConfig,
) -> float | None:
    scorer = getattr(adapter, "score_channel_candidate", None)
    if not callable(scorer):
        return None
    score = scorer(record, canonical, spec, candidate, signal, config)
    return None if score is None else float(score)


def call_read_annotations(
    adapter: Any,
    record: RecordTask,
    config: HypnodataConfig,
    duration_sec: float,
) -> AnnotationResult:
    reader = getattr(adapter, "read_annotations", None)
    if not callable(reader):
        return AnnotationResult()
    raw = reader(record, config, duration_sec)
    if raw is None:
        return AnnotationResult()
    if isinstance(raw, AnnotationResult):
        return raw
    raise ValueError("adapter.read_annotations must return AnnotationResult.")
