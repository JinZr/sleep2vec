from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import typing as t

import pandas as pd


@dataclass(frozen=True)
class DerivationJob:
    row_id: int | str
    path: str
    split: str
    subject_id: int | str
    night_id: int | str
    output_path: str


def validate_subject_split_boundaries(
    df: pd.DataFrame,
    *,
    subject_id_col: str = "subject_id",
    split_col: str = "split",
) -> None:
    missing = [col for col in (subject_id_col, split_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Derivation index missing required columns: {missing}")
    if df[subject_id_col].isna().any():
        offenders = df.index[df[subject_id_col].isna()].astype(str).tolist()
        raise ValueError(f"Derivation index contains missing {subject_id_col} values: {offenders}")
    if df[split_col].isna().any():
        offenders = df.index[df[split_col].isna()].astype(str).tolist()
        raise ValueError(f"Derivation index contains missing {split_col} values: {offenders}")

    split_counts = df.groupby(subject_id_col)[split_col].nunique(dropna=False)
    offenders = sorted(str(subject_id) for subject_id, count in split_counts.items() if count > 1)
    if offenders:
        raise ValueError(f"Subjects appear in multiple splits: {offenders}")


def plan_derivation_jobs(
    df: pd.DataFrame,
    *,
    output_dir: str | Path,
    path_col: str = "path",
    split_col: str = "split",
    subject_id_col: str = "subject_id",
    night_id_col: str = "night_id",
) -> list[DerivationJob]:
    required = [path_col, split_col, subject_id_col, night_id_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Derivation index missing required columns: {missing}")
    validate_subject_split_boundaries(df, subject_id_col=subject_id_col, split_col=split_col)

    output_root = Path(output_dir)
    jobs: list[DerivationJob] = []
    for row_number, row in df.reset_index(drop=True).iterrows():
        row_id = row["id"] if "id" in row and pd.notna(row["id"]) else row_number
        subject_id = row[subject_id_col]
        night_id = row[night_id_col]
        jobs.append(
            DerivationJob(
                row_id=row_id,
                path=str(row[path_col]),
                split=str(row[split_col]),
                subject_id=subject_id,
                night_id=night_id,
                output_path=str(output_root / f"subject-{subject_id}_night-{night_id}_derived.npz"),
            )
        )
    return jobs


def require_derivation_backend(enabled_derivations: t.Sequence[str]) -> None:
    if enabled_derivations:
        raise NotImplementedError(
            "Clinical-grade sleep2wave IBI/RESP derivation is not implemented in PR 2. "
            "This stage only validates split-safe per-record derivation planning."
        )


__all__ = ["DerivationJob", "plan_derivation_jobs", "require_derivation_backend", "validate_subject_split_boundaries"]
