from pathlib import Path
import pickle
from types import SimpleNamespace

import pytest

from agent_tools import plan_context
from agent_tools.domain.presets import preset_summary


def test_preset_failure_keeps_only_core_fields(tmp_path: Path):
    path = tmp_path / "missing.pkl"
    result = preset_summary(path)
    assert result == {
        "preset_path": str(path),
        "samples": 0,
        "warnings": [],
        "blocking_issues": [f"Preset not found: {path}"],
    }
    path.write_bytes(b"not a pickle")
    result = preset_summary(path)
    assert set(result) == {"preset_path", "samples", "warnings", "blocking_issues"}
    assert result["samples"] == 0
    assert result["blocking_issues"][0].startswith("Failed to load preset:")


def test_preset_keeps_dynamic_values_and_manifest(tmp_path: Path):
    path = tmp_path / "samples.pkl"
    sample = SimpleNamespace(
        id=17,
        path=Path("relative.npz"),
        start="001",
        end="009",
        metadata={"source": 3},
        payload={"available_channels": ["ppg"]},
    )
    path.write_bytes(pickle.dumps([sample]))
    path.with_name(path.name + ".manifest.json").write_text('["raw", 1, null]')
    result = preset_summary(path)
    assert result["id_examples"] == [17]
    assert result["path_examples"] == [Path("relative.npz")]
    assert result["start_end"] == {"min_start": "001", "max_end": "009"}
    assert result["sidecar_manifest"] == ["raw", 1, None]
    assert result["source_counts"] == {"3": 1}
    assert result["available_channels_counts"] == {"ppg": 1}
    assert result["blocking_issues"] == []


@pytest.mark.parametrize("kind", ["index", "preset"])
def test_context_summary_distinguishes_absence_from_error(kind, monkeypatch):
    if kind == "index":
        summarize = plan_context.context_index_summary
        recipe = {"inputs": {"index": ["index.csv"]}}
        assert summarize({}, None) is None
    else:
        summarize = plan_context.context_preset_summary
        recipe = {}
        monkeypatch.setattr(plan_context, "effective_preset_path", lambda *args: None)
        assert summarize(recipe, None) is None
        monkeypatch.setattr(plan_context, "effective_preset_path", lambda *args: "preset.pkl")

    def fail(*args, **kwargs):
        raise ValueError("unreadable input")

    monkeypatch.setattr(plan_context, f"{kind}_summary", fail)
    assert summarize(recipe, None) == {"blocking_issues": [f"Failed to summarize {kind}: unreadable input"]}
