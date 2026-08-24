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
