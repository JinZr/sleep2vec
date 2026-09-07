from pathlib import Path

import pytest

from agent_tools import configs, plan_context
from agent_tools.domain import sidecar_summaries


@pytest.mark.parametrize("kind,task", [("survival", "survival"), ("multilabel", "multilabel_classification")])
def test_sidecar_diagnostics_preserve_invalid_raw_values(kind, task):
    summarize = getattr(sidecar_summaries, f"{kind}_summary")
    invalid_key = {"invalid": "key"}
    invalid_dimension = ["not", "an integer"]
    summary = summarize({kind: {"key_column": invalid_key}}, {"type": task, "output_dim": invalid_dimension})
    assert summary["key_column"] is invalid_key
    assert summary["output_dim"] is invalid_dimension
    assert summary["valid"] is False
    assert summary["disease_count"] is None
    assert summary["sidecar_key_count"] is None
    assert summary["issues"][0] == f"finetune.{kind}.key_column must be a non-empty string."
    assert summarize({}, {"type": "classification"}) is None


@pytest.mark.parametrize("kind,task", [("survival", "survival"), ("multilabel", "multilabel_classification")])
def test_deferred_sidecars_do_not_claim_validation(kind, task, monkeypatch):
    summarize = getattr(sidecar_summaries, f"{kind}_summary")
    fields = (
        ("disease_columns_index", "event_time_index", "is_event_index", "has_label_index")
        if kind == "survival"
        else ("disease_columns_index", "label_index", "has_label_index")
    )
    raw = {"key_column": "eid", **{field: f"/remote/{field}.csv" for field in fields}}
    validated_keys = {"existing": {"001"}}

    def unexpected_resolution(*args, **kwargs):
        raise AssertionError("Deferred paths must not be resolved locally")

    monkeypatch.setattr(sidecar_summaries, "resolve_repo_path", unexpected_resolution)
    summary = summarize(
        {kind: raw},
        {"type": task, "output_dim": 2},
        validate_local_paths=False,
        validated_sidecar_keys=validated_keys,
    )
    assert summary["issues"] == []
    assert summary["valid"] is False
    assert summary["sidecar_key_count"] is None
    assert summary["disease_count"] is None
    assert validated_keys == {"existing": {"001"}}
    assert all(summary[field] == raw[field] for field in fields)


def test_config_dispatch_preserves_summary_identity_and_original_path(tmp_path: Path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("authored: original\n")
    payload = {"config_path": "temporary", "warnings": [], "blocking_issues": [], "extra": {"raw": True}}
    seen_paths = []

    class Adapter:
        def matches_config_data(self, data):
            return data == {"authored": "snapshot"}

        def config_summary(self, summary_path):
            seen_paths.append(summary_path)
            assert summary_path.read_bytes() == b"authored: snapshot\n"
            return payload

    monkeypatch.setattr(configs, "all_adapters", lambda: [Adapter()])
    result = configs.config_summary(path, config_bytes=b"authored: snapshot\n")
    assert result is payload
    assert result["extra"] == {"raw": True}
    assert result["config_path"] == str(path)
    assert path.read_bytes() == b"authored: original\n"
    assert len(seen_paths) == 1
    assert not seen_paths[0].exists()


def test_context_skill_keeps_raw_discovery_values(monkeypatch):
    owners = {"raw": "owners"}
    relevant = {"raw": "references"}
    monkeypatch.setattr(
        plan_context,
        "list_skills",
        lambda: [
            {"name": 7, "path": None, "task_types": ["finetune"], "owners": owners, "relevant_index": relevant},
        ],
    )
    skill, documents = plan_context.skill_context("finetune")
    assert skill == {"name": 7, "path": None, "owners": owners}
    assert skill["owners"] is owners
    assert documents is relevant
    assert plan_context.skill_context("unmatched") == ({"name": None, "path": None, "owners": []}, [])
