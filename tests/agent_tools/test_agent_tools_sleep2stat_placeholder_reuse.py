from __future__ import annotations

import pytest

from agent_tools.adapters import sleep2stat
from agent_tools.domain import sidecar_summaries


def test_sleep2stat_uses_canonical_placeholder_path_predicate():
    assert sleep2stat.looks_like_placeholder_path is sidecar_summaries.looks_like_placeholder_path


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        ("", True),
        ("ASK_USER", True),
        ("/path/to/model.ckpt", True),
        ("<checkpoint>", True),
        ("checkpoints/model.ckpt", False),
    ],
)
def test_sleep2stat_placeholder_path_behavior_is_unchanged(value, expected):
    assert sleep2stat.looks_like_placeholder_path(value) is expected
