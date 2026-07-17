#!/usr/bin/env python3
"""
Collect sex and age from all UK Biobank-style JSON files under a root folder.

Expected input layout can be like:

460/
├── 4600005/
│   └── some_file.json
├── 4600018/
│   └── ukb676623.json
└── nested/
    └── 4600183/
        └── another_name.json

The script recursively scans every *.json file by default.
It writes one row per JSON file.

Usage:
    python utils/collect_ukb_demographics.py /path/to/460 -o sex_age.csv

Optional:
    python utils/collect_ukb_demographics.py /path/to/460 -o sex_age.csv --dedupe-by-eid
"""

import argparse
import csv
from datetime import date, datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_tools.progress import write_progress

SEX_LABELS = {
    "0": "Female",
    "1": "Male",
}

PRIMARY_SEX_KEYS = [
    "sex__f31_i0_a0",
    "genetic_sex__f22001_i0_a0",
]

PRIMARY_AGE_KEYS = [
    "age_when_attended_assessment_centre__f21003_i0_a0",
    "age_at_recruitment__f21022_i0_a0",
]

DOB_YEAR_KEYS = [
    "year_of_birth__f34_i0_a0",
]

DOB_MONTH_KEYS = [
    "month_of_birth__f52_i0_a0",
]

ASSESSMENT_DATE_KEYS = [
    "date_of_attending_assessment_centre__f53_i0_a0",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("JSON top level is not an object")
    return obj


def clean_value(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def get_values(obj: Dict[str, Any]) -> Dict[str, Any]:
    values = obj.get("values", {})
    if isinstance(values, dict):
        return values
    return {}


def first_present(values: Dict[str, Any], keys) -> Tuple[str, str]:
    for key in keys:
        value = clean_value(values.get(key))
        if value != "":
            return value, key
    return "", ""


def find_fallback_key(values: Dict[str, Any], include_terms, exclude_terms=()) -> Tuple[str, str]:
    include_terms = [t.lower() for t in include_terms]
    exclude_terms = [t.lower() for t in exclude_terms]

    for key in sorted(values):
        key_lower = key.lower()
        if all(t in key_lower for t in include_terms) and not any(t in key_lower for t in exclude_terms):
            value = clean_value(values.get(key))
            if value != "":
                return value, key

    return "", ""


def parse_date(s: str) -> Optional[date]:
    if not s:
        return None

    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(s).date()
    except ValueError:
        return None


def compute_age_from_birth_and_assessment(values: Dict[str, Any]) -> Tuple[str, str]:
    year_str, year_key = first_present(values, DOB_YEAR_KEYS)
    month_str, month_key = first_present(values, DOB_MONTH_KEYS)
    assessment_str, assessment_key = first_present(values, ASSESSMENT_DATE_KEYS)

    if not year_str or not assessment_str:
        return "", ""

    try:
        birth_year = int(float(year_str))
    except ValueError:
        return "", ""

    try:
        birth_month = int(float(month_str)) if month_str else 7
    except ValueError:
        birth_month = 7

    assessment_date = parse_date(assessment_str)
    if assessment_date is None:
        return "", ""

    # UK Biobank usually provides year and month of birth, not day.
    # Use day=15 as a stable mid-month approximation when calculating age.
    birth_date = date(birth_year, max(1, min(12, birth_month)), 15)
    age = (
        assessment_date.year
        - birth_date.year
        - ((assessment_date.month, assessment_date.day) < (birth_date.month, birth_date.day))
    )

    source = f"computed_from:{year_key}"
    if month_key:
        source += f"+{month_key}"
    source += f"+{assessment_key}"

    return str(age), source


def extract_record(path: Path, root: Path) -> Dict[str, str]:
    obj = load_json(path)
    values = get_values(obj)

    eid = clean_value(obj.get("eid"))
    if not eid:
        # Typical layout is root/eid/file.json. If top-level eid is missing,
        # use the parent directory as a fallback.
        eid = path.parent.name

    dataset_id = clean_value(obj.get("dataset_id"))
    if not dataset_id:
        dataset_id = path.stem

    sex_code, sex_source = first_present(values, PRIMARY_SEX_KEYS)
    if not sex_code:
        sex_code, sex_source = find_fallback_key(
            values,
            include_terms=["sex"],
            exclude_terms=["duration", "sexual", "oophorectomy", "testosterone"],
        )

    age, age_source = first_present(values, PRIMARY_AGE_KEYS)
    if not age:
        age, age_source = find_fallback_key(
            values,
            include_terms=["age", "assessment"],
            exclude_terms=["duration", "screen", "time"],
        )

    if not age:
        age, age_source = compute_age_from_birth_and_assessment(values)

    rel_path = path.relative_to(root).as_posix()

    return {
        "eid": eid,
        "dataset_id": dataset_id,
        "sex_code": sex_code,
        "sex_label": SEX_LABELS.get(sex_code, ""),
        "age": age,
        "json_path": str(path),
        "relative_json_path": rel_path,
        "sex_source": sex_source,
        "age_source": age_source,
    }


def iter_json_files(root: Path, pattern: str):
    yield from sorted(root.rglob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively collect eid, sex, and age from all JSON files under a folder."
    )
    parser.add_argument("root", type=Path, help="Root folder to scan, e.g. /path/to/460")
    parser.add_argument("-o", "--output", type=Path, default=Path("sex_age.csv"), help="Output CSV path")
    parser.add_argument("--pattern", default="*.json", help='JSON filename pattern. Default: "*.json"')
    parser.add_argument(
        "--dedupe-by-eid",
        action="store_true",
        help="Keep only the first JSON record for each eid after sorting paths.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Create an empty CSV with headers if no JSON files are found.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    if not root.is_dir():
        raise SystemExit(f"Root is not a directory: {root}")

    rows = []
    errors = []

    json_files = list(iter_json_files(root, args.pattern))
    started_at = time.time()
    write_progress(
        output.parent,
        status="running",
        task="collect_ukb_demographics",
        processed=0,
        total=len(json_files),
        success=0,
        failed=0,
        start_time=started_at,
    )
    for processed, json_path in enumerate(tqdm(json_files, desc="Collecting JSON metadata", unit="file"), start=1):
        try:
            rows.append(extract_record(json_path, root))
        except Exception as exc:
            errors.append((str(json_path), str(exc)))
        write_progress(
            output.parent,
            status="running",
            task="collect_ukb_demographics",
            processed=processed,
            total=len(json_files),
            success=len(rows),
            failed=len(errors),
            start_time=started_at,
            current_item=str(json_path),
        )

    if args.dedupe_by_eid:
        seen = set()
        deduped = []
        for row in rows:
            eid = row["eid"]
            if eid in seen:
                continue
            seen.add(eid)
            deduped.append(row)
        rows = deduped

    fieldnames = [
        "eid",
        "dataset_id",
        "sex_code",
        "sex_label",
        "age",
        "json_path",
        "relative_json_path",
        "sex_source",
        "age_source",
    ]

    if not rows and not args.allow_empty:
        write_progress(
            output.parent,
            status="failed",
            task="collect_ukb_demographics",
            processed=len(json_files),
            total=len(json_files),
            success=len(rows),
            failed=len(errors),
            start_time=started_at,
            message="no JSON files found",
        )
        raise SystemExit(
            f"No JSON files found under {root} with pattern {args.pattern}. "
            "Use --allow-empty if you still want an empty CSV."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")
    write_progress(
        output.parent,
        status="completed",
        task="collect_ukb_demographics",
        processed=len(json_files),
        total=len(json_files),
        success=len(rows),
        failed=len(errors),
        start_time=started_at,
        message=f"Wrote {len(rows)} rows to {output}",
    )

    if errors:
        error_log = output.with_suffix(output.suffix + ".errors.txt")
        with error_log.open("w", encoding="utf-8") as f:
            for path, message in errors:
                f.write(f"{path}\t{message}\n")
        print(f"Skipped {len(errors)} unreadable JSON files. Error log: {error_log}")


if __name__ == "__main__":
    main()
