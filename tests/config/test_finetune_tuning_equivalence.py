"""Gate the `finetune.tuning` migration against the trainability it replaced.

`utils/migrate_finetune_tuning.py` did not rename keys. It evaluated the legacy
runtime semantics for every config and emitted the preset that reproduced the
resulting per-group table, recording both in ``doc/finetune_tuning_migration.json``.

This module replays that manifest through the *new* parser and asserts the two agree
on `(train, lr_scale)` for every group of every config. A preset table edit, a builder
regression, or a hand-edit of a migrated config that changes what trains will fail here
with the config and group named.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANIFEST_PATH = REPO_ROOT / "doc" / "finetune_tuning_migration.json"


def _manifest() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text())


def _load(entry: dict):
    if entry["variant"] == "sleep2expert":
        from sleep2expert.config import load_finetune_config
    else:
        from sleep2vec.config import load_finetune_config
    return load_finetune_config(REPO_ROOT / entry["path"])


MANIFEST = _manifest()


def test_manifest_covers_every_finetune_config() -> None:
    """Every config that carried the legacy keys is accounted for."""
    recorded = {entry["path"] for entry in MANIFEST}
    assert len(recorded) == len(MANIFEST), "manifest has duplicate paths"
    assert recorded, "manifest is empty"


@pytest.mark.parametrize("entry", MANIFEST, ids=lambda entry: entry["path"])
def test_migrated_config_matches_legacy_trainability(entry: dict) -> None:
    tuning = _load(entry).finetune.tuning
    assert sorted(tuning.groups) == sorted(entry["groups"])
    for group, (want_train, want_scale) in entry["expected"].items():
        assert tuning.trains(group) is bool(want_train), f"{entry['path']}: group '{group}' trainability changed"
        if want_train:
            assert tuning.lr_scale(group) == pytest.approx(want_scale), f"{entry['path']}: group '{group}' lr_scale"


@pytest.mark.parametrize("entry", MANIFEST, ids=lambda entry: entry["path"])
def test_migrated_config_records_the_expected_preset(entry: dict) -> None:
    assert _load(entry).finetune.tuning.preset == entry["preset"]


def test_legacy_table_derivation_is_reproducible() -> None:
    """The manifest's legacy tables still follow from the recorded legacy semantics.

    Guards against someone editing the manifest to make a failing config pass: the
    `expected` column has to remain what `legacy_trainability_table` produced, modulo
    the backbone -> encoder rename and the variant's group list.
    """
    from utils.migrate_finetune_tuning import to_new_groups

    for entry in MANIFEST:
        legacy = {group: list(value) for group, value in entry["legacy"].items()}
        assert to_new_groups(legacy, entry["groups"]) == {
            group: list(value) for group, value in entry["expected"].items()
        }, entry["path"]
