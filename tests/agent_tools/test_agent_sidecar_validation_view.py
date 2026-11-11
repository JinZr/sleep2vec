"""One consultation keeps one sidecar validation view: index checks reuse a complete, successful local sidecar
validation, and nothing failed, deferred or from another call is reused."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from agent_tool_test_helpers import survival_config_payload, write_finetune_recipe, write_survival_sidecars, write_yaml
import pandas as pd
import pytest
import yaml

from agent_tools import plan_context, plans
from agent_tools.configs import config_summary
from agent_tools.domain import index_csv
from agent_tools.models import REPO_ROOT


def _sidecar_recipe(tmp_path: Path, kind: str, *, variant: str = "sleep2vec") -> Path:
    recipe = write_finetune_recipe(tmp_path, variant=variant)
    index = tmp_path / "index.csv"
    index.write_text(
        "path,split,duration,eid,ppg_mask,age,sex\n"
        "x.npz,train,60,doctor-private-001,1,50,0\n"
        "y.npz,val,60,doctor-private-002,1,60,1\n"
    )
    sidecars = write_survival_sidecars(tmp_path)
    for field in ("event_time_index", "is_event_index", "has_label_index"):
        path = Path(sidecars[field])
        path.write_text(path.read_text().replace("001", "doctor-private-001").replace("002", "doctor-private-002"))
    if variant == "sex_age_baseline":
        payload = yaml.safe_load((REPO_ROOT / "configs/sex_age_baseline/cox.yaml").read_text())
        payload["data"]["finetune_data_index"] = str(index)
        payload["finetune"]["survival"].update(sidecars)
        payload["finetune"]["task"].update(output_dim=2, monitor="val_loss", monitor_mod="min")
    else:
        payload = survival_config_payload(index, sidecars)
    if kind == "multilabel":
        survival = payload["finetune"].pop("survival")
        if variant == "sex_age_baseline":
            payload["finetune"]["loss"] = {"pos_weight": None}
        payload["finetune"]["task"]["type"] = "multilabel_classification"
        payload["finetune"]["multilabel"] = {
            "key_column": "eid",
            "disease_columns_index": sidecars["disease_columns_index"],
            "label_index": sidecars["is_event_index"],
            "has_label_index": sidecars["has_label_index"],
            # The baseline's required covariates move with the task block; the signal source declares none.
            **{key: value for key, value in survival.items() if key.startswith("covariate")},
        }
    write_yaml(tmp_path / "config.yaml", payload)
    payload = yaml.safe_load(recipe.read_text())
    payload["inputs"]["label_name"] = "incident_cox" if kind == "survival" else "disease_label"
    payload["evaluation_policy"].update(selection_metric="val_loss", selection_mode="min")
    if variant == "sex_age_baseline":
        payload["decisions"]["required_channels"] = {"value": [], "source": "explicit_recipe"}
    write_yaml(recipe, payload)
    return recipe


@pytest.fixture
def csv_reads(monkeypatch):
    calls = Counter()
    read_csv = pd.read_csv

    def counted(path, *args, **kwargs):
        calls[Path(path).name] += 1
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", counted)
    return calls


@pytest.mark.parametrize("kind", ["survival", "multilabel"])
def test_evaluate_recipe_retains_one_validation_view_and_refreshes_next_call(tmp_path, monkeypatch, csv_reads, kind):
    recipe_path = _sidecar_recipe(tmp_path, kind)
    if kind == "survival":
        from data import survival as loader

        name = "load_survival_label_table"
    else:
        from data import multilabel as loader

        name = "load_multilabel_label_table"
    load = getattr(loader, name)
    mutations = []

    def mutate_after_validation(*args, **kwargs):
        labels = load(*args, **kwargs)
        if not mutations:
            mutations.append(True)
            for filename in ("event_time.csv", "is_event.csv", "has_label.csv"):
                path = tmp_path / filename
                path.write_text(path.read_text().replace("doctor-private-001", "doctor-private-003"))
        return labels

    monkeypatch.setattr(loader, name, mutate_after_validation)
    recipe, cfg, report = plans.evaluate_recipe(recipe_path)
    assert report.exit_code == 0
    assert csv_reads == (
        {"event_time.csv": 1, "is_event.csv": 1, "has_label.csv": 1, "index.csv": 1}
        if kind == "survival"
        else {"is_event.csv": 1, "has_label.csv": 1, "index.csv": 1}
    )
    public_cfg = {key: value for key, value in cfg.items() if key != "_source_config_bytes"}
    public_payload = json.dumps([recipe, public_cfg, asdict(report)])
    assert "validated_sidecar_keys" not in public_payload
    assert "doctor-private-" not in public_payload

    csv_reads.clear()
    _recipe, _cfg, report = plans.evaluate_recipe(recipe_path)
    assert report.exit_code == 1
    assert any(
        "missing from sidecars" in issue.message and "doctor-private-001" in issue.message for issue in report.issues
    )
    assert all(count == 1 for count in csv_reads.values())


@pytest.mark.parametrize("kind", ["survival", "multilabel"])
def test_valid_empty_sidecar_keys_are_not_treated_as_missing_evidence(tmp_path, csv_reads, kind):
    recipe_path = _sidecar_recipe(tmp_path, kind)
    for filename in ("event_time.csv", "is_event.csv", "has_label.csv"):
        (tmp_path / filename).write_text("eid,d1,d2\n")

    _recipe, cfg, report = plans.evaluate_recipe(recipe_path)

    assert cfg["finetune"][kind]["valid"] is True
    assert cfg["finetune"][kind]["sidecar_key_count"] == 0
    assert report.exit_code == 1
    assert any("2 missing" in issue.message for issue in report.issues)
    assert all(count == 1 for count in csv_reads.values())


@pytest.mark.parametrize("ineligible", ["missing_snapshot", "blocking_config", "invalid_sidecars", "missing_collector"])
def test_index_summary_reuse_requires_complete_local_evidence(tmp_path, monkeypatch, csv_reads, ineligible):
    recipe_path = _sidecar_recipe(tmp_path, "survival")
    recipe, cfg, _report = plans.evaluate_recipe(recipe_path)
    collector = {"survival": {"doctor-private-001", "doctor-private-002"}}
    if ineligible == "missing_snapshot":
        cfg.pop("_source_config_bytes")
    elif ineligible == "blocking_config":
        cfg["blocking_issues"] = ["structural config failure"]
    elif ineligible == "invalid_sidecars":
        cfg["finetune"]["survival"]["valid"] = False
    else:
        collector = None
    csv_reads.clear()
    calls = []

    def summarize(*args, **kwargs):
        calls.append(kwargs.get("validated_summary"))
        return index_csv.index_summary(*args, **kwargs)

    monkeypatch.setattr(plan_context, "index_summary", summarize)
    payload = plan_context.context_index_summary(recipe, cfg, validated_sidecar_keys=collector)
    assert calls == [None]
    assert payload["blocking_issues"] == []
    assert csv_reads == {"event_time.csv": 2, "is_event.csv": 1, "has_label.csv": 1, "index.csv": 1}


def test_sleep2stat_does_not_pass_its_config_to_index_summary(tmp_path, monkeypatch):
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration\nx.npz,train,60\n")
    cfg = {"is_sleep2stat": True, "sleep2stat": {"data": {"index": [str(index)]}}, "_source_config_bytes": b""}
    calls = []

    def summarize(*args, **kwargs):
        calls.append(kwargs)
        return index_csv.index_summary(*args, **kwargs)

    monkeypatch.setattr(plan_context, "index_summary", summarize)
    payload = plan_context.context_index_summary(
        {"task": "sleep2stat", "inputs": {"config": "unused.yaml"}}, cfg, validated_sidecar_keys={}
    )
    assert payload["blocking_issues"] == []
    assert calls[0]["config"] is None
    assert calls[0].get("validated_summary") is None


def test_forced_provider_failure_keeps_unbound_index_summary_behavior(tmp_path, monkeypatch):
    recipe_path = _sidecar_recipe(tmp_path, "survival")
    payload = yaml.safe_load(recipe_path.read_text())
    payload["variant"] = "sex_age_baseline"
    write_yaml(recipe_path, payload)
    calls = []

    def summarize(*args, **kwargs):
        calls.append(kwargs.get("validated_summary"))
        return index_csv.index_summary(*args, **kwargs)

    monkeypatch.setattr(plan_context, "index_summary", summarize)
    _recipe, cfg, report = plans.evaluate_recipe(recipe_path)
    assert report.exit_code != 0
    assert cfg["blocking_issues"]
    assert calls == []  # The rejected provider has no index path; do not recover one from another provider.


def test_standalone_index_summary_still_performs_its_own_validation(tmp_path, csv_reads):
    _sidecar_recipe(tmp_path, "survival")

    payload = index_csv.index_summary([tmp_path / "index.csv"], config=tmp_path / "config.yaml")

    assert payload["blocking_issues"] == []
    assert csv_reads == {"event_time.csv": 2, "is_event.csv": 1, "has_label.csv": 1, "index.csv": 1}


def test_summary_collects_keys_only_after_complete_loader_success(tmp_path):
    _sidecar_recipe(tmp_path, "survival")
    (tmp_path / "has_label.csv").write_text("eid,d1,d2\ndoctor-private-001,1,1\n")
    collector = {}

    cfg = config_summary(tmp_path / "config.yaml", validated_sidecar_keys=collector)

    assert cfg["finetune"]["survival"]["valid"] is False
    assert collector == {}


@pytest.mark.parametrize("kind", ["survival", "multilabel"])
def test_failed_full_sidecar_validation_is_not_reused(tmp_path, monkeypatch, csv_reads, kind):
    recipe_path = _sidecar_recipe(tmp_path, kind)
    (tmp_path / "has_label.csv").write_text("eid,d1,d2\ndoctor-private-001,1,1\n")
    calls = []

    def summarize(*args, **kwargs):
        calls.append(kwargs.get("validated_summary"))
        return index_csv.index_summary(*args, **kwargs)

    monkeypatch.setattr(plan_context, "index_summary", summarize)
    _recipe, cfg, report = plans.evaluate_recipe(recipe_path)

    assert report.exit_code == 2
    assert cfg["finetune"][kind]["valid"] is False
    assert calls == [None]
    assert csv_reads == (
        {"event_time.csv": 2, "is_event.csv": 2, "has_label.csv": 2, "index.csv": 1}
        if kind == "survival"
        else {"is_event.csv": 2, "has_label.csv": 2, "index.csv": 1}
    )


@pytest.mark.parametrize("kind", ["survival", "multilabel"])
def test_reuse_keeps_runtime_relative_paths_and_requested_splits(tmp_path, csv_reads, kind):
    recipe_path = _sidecar_recipe(tmp_path, kind)
    config_path = tmp_path / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["data"]["finetune_data_index"] = "index.csv"
    for field, value in config["finetune"][kind].items():
        if field.endswith("_index"):
            config["finetune"][kind][field] = Path(value).name
    write_yaml(config_path, config)
    recipe = yaml.safe_load(recipe_path.read_text())
    recipe["execution"] = {"workdir": str(tmp_path)}
    write_yaml(recipe_path, recipe)
    index = tmp_path / "index.csv"
    index.write_text(index.read_text() + "z.npz,test,60,unloaded-test-key,1,55,0\n")

    _recipe, _cfg, report = plans.evaluate_recipe(recipe_path)

    assert report.exit_code == 0
    assert csv_reads
    assert all(count == 1 for count in csv_reads.values())


@pytest.mark.parametrize("kind", ["survival", "multilabel"])
def test_consultation_still_validates_sidecar_rows_absent_from_main_index(tmp_path, kind):
    recipe_path = _sidecar_recipe(tmp_path, kind)
    for filename in ("event_time.csv", "is_event.csv", "has_label.csv"):
        path = tmp_path / filename
        value = "bad" if filename == ("event_time.csv" if kind == "survival" else "is_event.csv") else "1"
        path.write_text(path.read_text() + f"not-in-main-index,{value},1\n")

    _recipe, cfg, report = plans.evaluate_recipe(recipe_path)

    assert report.exit_code == 2
    assert cfg["finetune"][kind]["valid"] is False
    assert cfg["finetune"][kind]["issues"]
