from __future__ import annotations

from collections import Counter
import csv
import math
from pathlib import Path
import typing as t


def validate_whole_night_index(
    index_paths: t.Iterable[str | Path],
    *,
    eval_split: str,
    max_source_tokens: int,
    path_base: Path | None = None,
    sources: t.Iterable[str] = (),
) -> dict[str, int]:
    expected_tokens_by_path: dict[str, int] = {}
    selected_sources = tuple(sources)
    for index_path in index_paths:
        index_file = Path(index_path)
        if not index_file.is_absolute() and path_base is not None:
            index_file = path_base / index_file
        with index_file.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames or []
            duplicates = sorted(name for name, count in Counter(fieldnames).items() if count > 1)
            if duplicates:
                raise ValueError(f"Whole-night index {index_path} has duplicate columns: {duplicates}")
            required = {"path", "split", "duration"}
            missing = sorted(required - set(fieldnames))
            if missing:
                raise ValueError(f"Whole-night index {index_path} is missing required columns: {missing}")
            for row in reader:
                if None in row:
                    raise ValueError(f"Whole-night index {index_path} contains unexpected extra values.")
                if row["split"] != eval_split:
                    continue
                path = str(row["path"])
                source = row.get("source") or str(index_path)
                if selected_sources and not any(name in str(source) for name in selected_sources):
                    raise ValueError(
                        f"Whole-night path {path} source {source!r} does not match configured sources: "
                        f"{list(selected_sources)}"
                    )
                if path in expected_tokens_by_path:
                    raise ValueError(f"Duplicate whole-night path in selected split: {path}")
                try:
                    duration = float(row["duration"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid duration for whole-night path {path}: {row['duration']!r}") from exc
                if not math.isfinite(duration):
                    raise ValueError(f"Non-finite duration for whole-night path {path}: {duration}")
                duration_tokens = duration / 30
                if not duration_tokens.is_integer():
                    raise ValueError(
                        f"Whole-night path {path} duration must be aligned to 30-second tokens: {duration}."
                    )
                num_tokens = int(duration_tokens)
                if num_tokens < 1 or num_tokens > max_source_tokens:
                    raise ValueError(
                        f"Whole-night path {path} has {num_tokens} source tokens; "
                        f"expected [1, {max_source_tokens}]."
                    )
                sample_path = Path(path)
                if not sample_path.is_absolute() and path_base is not None:
                    sample_path = path_base / sample_path
                if not sample_path.is_file():
                    raise FileNotFoundError(f"Whole-night NPZ path not found: {path}")
                expected_tokens_by_path[path] = num_tokens

    if not expected_tokens_by_path:
        raise ValueError(f"No whole-night rows found for split {eval_split!r}.")
    return expected_tokens_by_path
