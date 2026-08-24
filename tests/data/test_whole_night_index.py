from pathlib import Path

import pytest

from data.whole_night_index import validate_whole_night_index


def _write_index(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path.write_text(
        "path,split,duration\n" + "".join(f"{sample},{split},{duration}\n" for sample, split, duration in rows)
    )
    return path


def test_validate_whole_night_index_returns_selected_token_counts(tmp_path: Path):
    sample = tmp_path / "night.npz"
    sample.touch()
    index = _write_index(tmp_path / "index.csv", [(str(sample), "test", "60"), ("unused.npz", "train", "30")])

    assert validate_whole_night_index([index], eval_split="test", max_source_tokens=2) == {str(sample): 2}


def test_validate_whole_night_index_resolves_sample_paths_from_explicit_base(tmp_path: Path):
    (tmp_path / "night.npz").touch()
    index = _write_index(tmp_path / "index.csv", [("night.npz", "test", "30")])

    assert validate_whole_night_index([index], eval_split="test", max_source_tokens=2, path_base=tmp_path) == {
        "night.npz": 1
    }


def test_validate_whole_night_index_resolves_relative_index_from_explicit_base(tmp_path: Path):
    (tmp_path / "night.npz").touch()
    _write_index(tmp_path / "shhs-index.csv", [("night.npz", "test", "30")])

    assert validate_whole_night_index(
        ["shhs-index.csv"],
        eval_split="test",
        max_source_tokens=2,
        path_base=tmp_path,
        sources=["shhs"],
    ) == {"night.npz": 1}


def test_validate_whole_night_index_rejects_empty_split(tmp_path: Path):
    index = _write_index(tmp_path / "index.csv", [("unused.npz", "train", "30")])

    with pytest.raises(ValueError, match="No whole-night rows found"):
        validate_whole_night_index([index], eval_split="test", max_source_tokens=2)


def test_validate_whole_night_index_rejects_duplicate_paths(tmp_path: Path):
    sample = tmp_path / "night.npz"
    sample.touch()
    index = _write_index(tmp_path / "index.csv", [(str(sample), "test", "30"), (str(sample), "test", "30")])

    with pytest.raises(ValueError, match="Duplicate whole-night path"):
        validate_whole_night_index([index], eval_split="test", max_source_tokens=2)


def test_validate_whole_night_index_rejects_duplicate_sample_keys(tmp_path: Path):
    first = tmp_path / "first" / "set" / "night.npz"
    second = tmp_path / "second" / "set" / "night.npz"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()
    index = tmp_path / "index.csv"
    index.write_text("path,split,duration,source\n" f"{first},test,30,mesa\n" f"{second},test,30,mesa\n")

    with pytest.raises(ValueError, match="Duplicate embedding sample_key"):
        validate_whole_night_index([index], eval_split="test", max_source_tokens=2)


def test_validate_whole_night_index_rejects_duplicate_headers(tmp_path: Path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first.touch()
    second.touch()
    index = tmp_path / "index.csv"
    index.write_text(f"path,split,duration,path\n{first},test,30,{second}\n")

    with pytest.raises(ValueError, match="duplicate columns.*path"):
        validate_whole_night_index([index], eval_split="test", max_source_tokens=2)


def test_validate_whole_night_index_rejects_surplus_row_values(tmp_path: Path):
    sample = tmp_path / "night.npz"
    sample.touch()
    index = tmp_path / "index.csv"
    index.write_text(f"path,split,duration\n{sample},test,30,unexpected\n")

    with pytest.raises(ValueError, match="unexpected extra values"):
        validate_whole_night_index([index], eval_split="test", max_source_tokens=2)


@pytest.mark.parametrize("duration", ["45", "nan"])
def test_validate_whole_night_index_rejects_invalid_duration(tmp_path: Path, duration: str):
    sample = tmp_path / "night.npz"
    sample.touch()
    index = _write_index(tmp_path / "index.csv", [(str(sample), "test", duration)])

    with pytest.raises(ValueError):
        validate_whole_night_index([index], eval_split="test", max_source_tokens=2)


def test_validate_whole_night_index_rejects_token_cap(tmp_path: Path):
    sample = tmp_path / "night.npz"
    sample.touch()
    index = _write_index(tmp_path / "index.csv", [(str(sample), "test", "90")])

    with pytest.raises(ValueError, match=r"expected \[1, 2\]"):
        validate_whole_night_index([index], eval_split="test", max_source_tokens=2)


def test_validate_whole_night_index_rejects_missing_npz(tmp_path: Path):
    index = _write_index(tmp_path / "index.csv", [(str(tmp_path / "missing.npz"), "test", "30")])

    with pytest.raises(FileNotFoundError, match="Whole-night NPZ path not found"):
        validate_whole_night_index([index], eval_split="test", max_source_tokens=2)


def test_validate_whole_night_index_accepts_matching_configured_source(tmp_path: Path):
    sample = tmp_path / "night.npz"
    sample.touch()
    index = tmp_path / "index.csv"
    index.write_text(f"path,split,duration,source\n{sample},test,30,shhs-v1\n")

    assert validate_whole_night_index([index], eval_split="test", max_source_tokens=2, sources=["shhs"]) == {
        str(sample): 1
    }


def test_validate_whole_night_index_uses_blank_source_fallback(tmp_path: Path):
    sample = tmp_path / "night.npz"
    sample.touch()
    index = tmp_path / "shhs-index.csv"
    index.write_text(f"path,split,duration,source\n{sample},test,30,\n")

    assert validate_whole_night_index([index], eval_split="test", max_source_tokens=2, sources=["shhs"]) == {
        str(sample): 1
    }


def test_validate_whole_night_index_rejects_configured_source_mismatch(tmp_path: Path):
    sample = tmp_path / "night.npz"
    sample.touch()
    index = tmp_path / "index.csv"
    index.write_text(f"path,split,duration,source\n{sample},test,30,mesa\n")

    with pytest.raises(ValueError, match="does not match configured sources"):
        validate_whole_night_index([index], eval_split="test", max_source_tokens=2, sources=["shhs"])
