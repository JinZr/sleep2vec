#!/usr/bin/env python3
"""Parse UK Biobank annotation exports into dataset metadata and per-person JSON files."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re

from tqdm import tqdm

MISSING_VALUES = {"", "NA"}


def clean_html_fragment(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    fragment = html.unescape(fragment)
    lines = []
    for raw_line in fragment.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def parse_vector_items(inner: str) -> list[str]:
    inner = re.sub(r"\s*\n\s*", " ", inner.strip())
    reader = csv.reader([inner], skipinitialspace=True)
    items = next(reader, [])
    return [item.strip() for item in items]


def sanitize_slug(text: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if not slug:
        slug = fallback
    return slug[:80].strip("_") or fallback


def build_ml_feature_name(description: str, field_id, instance, array_index, raw_column: str) -> str:
    if raw_column == "f.eid":
        return "eid"
    fallback = f"field_{field_id}" if field_id is not None else raw_column.replace(".", "_")
    slug = sanitize_slug(description or "", fallback)
    return f"{slug}__f{field_id}_i{instance}_a{array_index}"


def parse_udi(udi: str) -> dict:
    if udi == "eid":
        return {
            "field_id": None,
            "instance": None,
            "array_index": None,
            "raw_column": "f.eid",
        }
    match = re.fullmatch(r"(\d+)-(\d+)\.(\d+)", udi)
    if not match:
        return {
            "field_id": None,
            "instance": None,
            "array_index": None,
            "raw_column": f"f.{udi.replace('-', '.')}",
        }
    field_id, instance, array_index = (int(part) for part in match.groups())
    return {
        "field_id": field_id,
        "instance": instance,
        "array_index": array_index,
        "raw_column": f"f.{field_id}.{instance}.{array_index}",
    }


def parse_html_dictionary(path: Path) -> tuple[dict, list[dict]]:
    text = path.read_text(encoding="utf-8", errors="replace")

    extracted_at_match = re.search(r"<tr><td>Date Extracted:</td><td>([^<]+)</td></tr>", text)
    data_columns_match = re.search(r"<tr><td>Data columns:</td><td>(\d+)</td></tr>", text)
    application_match = re.search(r"<h1>UK Biobank : Data Dictionary for Application (\d+)</h1>", text)
    basket_match = re.search(r"<!-- Basket ID: (\d+) -->", text)
    run_match = re.search(r"<!-- Run ID: (\d+) -->", text)

    dictionary_rows = None
    for table_html in re.findall(r"<table\b[^>]*>(.*?)</table>", text, flags=re.DOTALL | re.IGNORECASE):
        rows = re.findall(r"<tr>(.*?)</tr>", table_html, flags=re.DOTALL | re.IGNORECASE)
        if not rows:
            continue
        header_cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", rows[0], flags=re.DOTALL | re.IGNORECASE)
        header_text = {clean_html_fragment(cell) for cell in header_cells}
        if {"Column", "UDI", "Count"}.issubset(header_text):
            dictionary_rows = rows
            break
    if dictionary_rows is None:
        raise ValueError(f"Could not locate dictionary table in {path.name}")

    columns = []
    current_type = ""
    current_description = ""

    for row_html in dictionary_rows[1:]:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.DOTALL)
        cell_text = [clean_html_fragment(cell) for cell in cells]
        if len(cell_text) == 5:
            column_index, udi, count, current_type, current_description = cell_text
            value_type = current_type
            description_text = current_description
        elif len(cell_text) == 3:
            column_index, udi, count = cell_text
            value_type = current_type
            description_text = current_description
        else:
            continue

        description_lines = [line for line in description_text.splitlines() if line]
        coding_id = None
        clean_description_lines = []
        for line in description_lines:
            coding_match = re.search(r"Uses data-coding (\d+)", line)
            if coding_match:
                coding_id = int(coding_match.group(1))
            else:
                clean_description_lines.append(line)
        description = " ".join(clean_description_lines).strip()

        parsed_udi = parse_udi(udi)
        raw_column = parsed_udi["raw_column"]
        columns.append(
            {
                "column_index": int(column_index),
                "udi": udi,
                "raw_column": raw_column,
                "field_id": parsed_udi["field_id"],
                "instance": parsed_udi["instance"],
                "array_index": parsed_udi["array_index"],
                "non_null_count": int(count),
                "type": value_type,
                "description": description,
                "coding_id": coding_id,
            }
        )

    metadata = {
        "application_id": int(application_match.group(1)) if application_match else None,
        "basket_id": int(basket_match.group(1)) if basket_match else None,
        "run_id": int(run_match.group(1)) if run_match else None,
        "extracted_at": extracted_at_match.group(1) if extracted_at_match else None,
        "declared_data_columns": int(data_columns_match.group(1)) if data_columns_match else None,
    }
    return metadata, columns


def parse_r_codings(path: Path) -> dict[int, list[dict]]:
    coding_levels = {}
    coding_labels = {}

    assignment_pattern = re.compile(r"(?ms)^\s*(lvl|lbl)\.(\d+)\s*<-\s*c\((.*?)\)\s*$")
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in assignment_pattern.finditer(text):
        kind, coding_id_text, items_text = match.groups()
        coding_id = int(coding_id_text)
        if kind == "lvl":
            coding_levels[coding_id] = parse_vector_items(items_text)
        else:
            coding_labels[coding_id] = parse_vector_items(items_text)

    codings = {}
    for coding_id in sorted(set(coding_levels) | set(coding_labels)):
        levels = coding_levels.get(coding_id, [])
        labels = coding_labels.get(coding_id, [])
        size = max(len(levels), len(labels))
        items = []
        for idx in range(size):
            items.append(
                {
                    "coding_id": coding_id,
                    "coded_value": levels[idx] if idx < len(levels) else "",
                    "label": labels[idx] if idx < len(labels) else "",
                }
            )
        codings[coding_id] = items
    return codings


def parse_header_only(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        header = handle.readline().rstrip("\n").rstrip("\r")
    raw_columns = header.split("\t")
    columns = []
    for index, raw_column in enumerate(raw_columns):
        if raw_column == "f.eid":
            field_id = instance = array_index = None
            udi = "eid"
            description = "Encoded anonymised participant ID"
        else:
            match = re.fullmatch(r"f\.(\d+)\.(\d+)\.(\d+)", raw_column)
            if match:
                field_id, instance, array_index = (int(part) for part in match.groups())
                udi = f"{field_id}-{instance}.{array_index}"
            else:
                field_id = instance = array_index = None
                udi = raw_column.removeprefix("f.")
            description = ""

        columns.append(
            {
                "column_index": index,
                "udi": udi,
                "raw_column": raw_column,
                "field_id": field_id,
                "instance": instance,
                "array_index": array_index,
                "non_null_count": None,
                "type": "",
                "description": description,
                "coding_id": None,
            }
        )
    return columns


def build_field_summary(columns: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for column in columns:
        grouped[column["field_id"]].append(column)

    summary_rows = []
    for field_id, items in grouped.items():
        items = sorted(items, key=lambda row: row["column_index"])
        instances = sorted({row["instance"] for row in items if row["instance"] != ""})
        array_indices = sorted({row["array_index"] for row in items if row["array_index"] != ""})
        summary_rows.append(
            {
                "field_id": field_id,
                "description": items[0]["description"],
                "type": items[0]["type"],
                "coding_id": items[0]["coding_id"],
                "column_count": len(items),
                "instances": ",".join(str(value) for value in instances),
                "array_indices": ",".join(str(value) for value in array_indices),
                "first_raw_column": items[0]["raw_column"],
                "first_udi": items[0]["udi"],
            }
        )

    summary_rows.sort(key=lambda row: (row["field_id"] == "", row["field_id"]))
    return summary_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_annotation_root(input_root: Path) -> Path:
    if any(input_root.glob("ukb*.tab")):
        return input_root
    annotation_root = input_root / "annotations"
    if any(annotation_root.glob("ukb*.tab")):
        return annotation_root
    raise SystemExit(f"No ukb*.tab files found in {input_root} or {annotation_root}")


def participant_path(output_root: Path, eid: str, dataset_id: str) -> Path:
    return output_root / "participants" / eid[:3] / eid / f"{dataset_id}.json"


def row_count_from_columns(columns: list[dict]) -> int | None:
    for column in columns:
        if column["raw_column"] == "f.eid" and column["non_null_count"] != "":
            return int(column["non_null_count"])
    return None


def normalize_columns(dataset_id: str, columns: list[dict]) -> list[dict]:
    normalized = []
    for column in columns:
        row = dict(column)
        row["dataset_id"] = dataset_id
        row["ml_feature_name"] = build_ml_feature_name(
            row["description"],
            row["field_id"],
            row["instance"],
            row["array_index"],
            row["raw_column"],
        )
        row["coding_id"] = "" if row["coding_id"] is None else row["coding_id"]
        row["field_id"] = "" if row["field_id"] is None else row["field_id"]
        row["instance"] = "" if row["instance"] is None else row["instance"]
        row["array_index"] = "" if row["array_index"] is None else row["array_index"]
        row["non_null_count"] = "" if row["non_null_count"] is None else row["non_null_count"]
        normalized.append(row)
    return normalized


def build_dataset_outputs(annotation_root: Path, output_root: Path, tab_path: Path) -> tuple[dict, list[dict]]:
    dataset_id = tab_path.stem
    base = annotation_root / dataset_id
    html_path = base.with_suffix(".html")
    r_path = base.with_suffix(".r")
    log_path = base.with_suffix(".log")

    metadata = {
        "dataset_id": dataset_id,
        "tab_file": tab_path.name,
        "tab_size_bytes": tab_path.stat().st_size,
        "has_html_dictionary": html_path.exists(),
        "has_r_codings": r_path.exists(),
        "has_log": log_path.exists(),
    }

    if html_path.exists():
        html_metadata, columns = parse_html_dictionary(html_path)
        metadata.update(html_metadata)
        schema_mode = "full_dictionary"
    else:
        columns = parse_header_only(tab_path)
        schema_mode = "header_only"

    codings = parse_r_codings(r_path) if r_path.exists() else {}
    columns = normalize_columns(dataset_id, columns)

    dataset_dir = output_root / "datasets" / dataset_id
    columns_csv = dataset_dir / "columns.csv"
    columns_jsonl = dataset_dir / "columns.jsonl"
    fields_csv = dataset_dir / "fields.csv"
    codings_csv = dataset_dir / "codings.csv"
    missing_codings_csv = dataset_dir / "missing_codings.csv"

    column_fieldnames = [
        "dataset_id",
        "column_index",
        "raw_column",
        "udi",
        "field_id",
        "instance",
        "array_index",
        "ml_feature_name",
        "type",
        "description",
        "coding_id",
        "non_null_count",
    ]
    write_csv(columns_csv, columns, column_fieldnames)
    write_jsonl(columns_jsonl, columns)

    field_summary = build_field_summary(columns)
    write_csv(
        fields_csv,
        field_summary,
        [
            "field_id",
            "description",
            "type",
            "coding_id",
            "column_count",
            "instances",
            "array_indices",
            "first_raw_column",
            "first_udi",
        ],
    )

    coding_rows = []
    for coding_id, items in sorted(codings.items()):
        for item in items:
            coding_rows.append(
                {
                    "dataset_id": dataset_id,
                    "coding_id": coding_id,
                    "coded_value": item["coded_value"],
                    "label": item["label"],
                }
            )
    write_csv(codings_csv, coding_rows, ["dataset_id", "coding_id", "coded_value", "label"])

    available_coding_ids = {str(coding_id) for coding_id in codings}
    missing_by_id = defaultdict(list)
    for column in columns:
        coding_id = column["coding_id"]
        if coding_id and str(coding_id) not in available_coding_ids:
            missing_by_id[str(coding_id)].append(column)

    missing_coding_rows = []
    for coding_id, items in sorted(missing_by_id.items(), key=lambda pair: int(pair[0])):
        first = min(items, key=lambda row: row["column_index"])
        missing_coding_rows.append(
            {
                "dataset_id": dataset_id,
                "coding_id": coding_id,
                "referenced_column_count": len(items),
                "first_raw_column": first["raw_column"],
                "first_udi": first["udi"],
                "description": first["description"],
            }
        )
    write_csv(
        missing_codings_csv,
        missing_coding_rows,
        [
            "dataset_id",
            "coding_id",
            "referenced_column_count",
            "first_raw_column",
            "first_udi",
            "description",
        ],
    )

    metadata.update(
        {
            "schema_mode": schema_mode,
            "column_count": len(columns),
            "coding_count": len(codings),
            "missing_coding_count": len(missing_coding_rows),
            "output_files": {
                "columns_csv": str(columns_csv.relative_to(output_root)),
                "columns_jsonl": str(columns_jsonl.relative_to(output_root)),
                "fields_csv": str(fields_csv.relative_to(output_root)),
                "codings_csv": str(codings_csv.relative_to(output_root)),
                "missing_codings_csv": str(missing_codings_csv.relative_to(output_root)),
            },
        }
    )
    return metadata, columns


def parse_withdrawals(annotation_root: Path, output_root: Path) -> tuple[dict, set[str]]:
    withdrawal_files = sorted(annotation_root.glob("withdraw*.txt"))
    entries = []
    withdrawn_eids = set()
    for file_path in withdrawal_files:
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                eid = raw_line.strip()
                if eid:
                    entries.append({"source_file": file_path.name, "eid": eid})
                    withdrawn_eids.add(eid)

    output_path = output_root / "withdrawals" / "withdrawn_eids.csv"
    write_csv(output_path, entries, ["source_file", "eid"])
    return (
        {
            "count": len(entries),
            "source_files": [path.name for path in withdrawal_files],
            "output_file": str(output_path.relative_to(output_root)),
        },
        withdrawn_eids,
    )


def validate_tab_header(tab_path: Path, columns: list[dict]) -> tuple[list[str], list[str], int]:
    with tab_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)

    metadata_columns = [column["raw_column"] for column in columns]
    if header != metadata_columns:
        raise ValueError(f"{tab_path.name} header does not match parsed column metadata")
    eid_index = header.index("f.eid")
    feature_names = [column["ml_feature_name"] for column in columns]
    return header, feature_names, eid_index


def write_participant_json(path: Path, eid: str, dataset_id: str, source_file: str, values: dict[str, str]) -> None:
    payload = {
        "eid": eid,
        "dataset_id": dataset_id,
        "source_file": source_file,
        "values": values,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_participants(
    output_root: Path,
    tab_path: Path,
    dataset_id: str,
    columns: list[dict],
    withdrawn_eids: set[str],
    exclude_withdrawn: bool,
    limit_rows: int | None,
) -> dict:
    _, feature_names, eid_index = validate_tab_header(tab_path, columns)
    total_rows = row_count_from_columns(columns)
    if limit_rows is not None:
        total_rows = min(total_rows, limit_rows) if total_rows is not None else limit_rows

    for old_path in (output_root / "participants").glob(f"*/*/{dataset_id}.json"):
        old_path.unlink()

    written_count = 0
    skipped_withdrawn_count = 0
    with tab_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        progress = tqdm(reader, total=total_rows, desc=f"{dataset_id} participants", unit="row")
        for row_index, row in enumerate(progress):
            if limit_rows is not None and row_index >= limit_rows:
                break

            eid = row[eid_index]
            if exclude_withdrawn and eid in withdrawn_eids:
                skipped_withdrawn_count += 1
                continue

            values = {}
            for value_index, value in enumerate(row):
                if value_index == eid_index or value in MISSING_VALUES:
                    continue
                values[feature_names[value_index]] = value

            path = participant_path(output_root, eid, dataset_id)
            write_participant_json(path, eid, dataset_id, tab_path.name, values)
            written_count += 1
        progress.close()

    return {
        "participant_json_count": written_count,
        "skipped_withdrawn_count": skipped_withdrawn_count,
        "participant_output": f"participants/<eid_prefix>/<eid>/{dataset_id}.json",
    }


def write_manifest(
    output_root: Path,
    annotation_root: Path,
    dataset_metadata: list[dict],
    withdrawal_metadata: dict,
    exclude_withdrawn: bool,
    limit_rows: int | None,
) -> Path:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(annotation_root),
        "datasets": dataset_metadata,
        "withdrawals": withdrawal_metadata,
        "participants": {
            "layout": "participants/<eid_prefix>/<eid>/<dataset_id>.json",
            "eid_prefix": "first three characters of eid",
            "missing_values_omitted": sorted(MISSING_VALUES),
            "values_are_raw_strings": True,
            "exclude_withdrawn": exclude_withdrawn,
            "limit_rows": limit_rows,
        },
    }
    path = output_root / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def write_readme(output_root: Path, dataset_metadata: list[dict], withdrawal_metadata: dict) -> Path:
    lines = [
        "# Parsed UK Biobank Annotation Bundle",
        "",
        "This directory is derived from raw UK Biobank export files.",
        "All originals remain unchanged. The files here are intended to make the bundle easier to inspect "
        "by hand and easier to consume from downstream ML/data pipelines.",
        "",
        "## Contents",
        "",
        "- `manifest.json`: machine-readable inventory of the derived bundle.",
        "- `datasets/<dataset_id>/columns.csv`: one row per raw column with parsed UDI parts and a stable "
        "ML-friendly feature name.",
        "- `datasets/<dataset_id>/columns.jsonl`: same schema as `columns.csv`, better for line-oriented readers.",
        "- `datasets/<dataset_id>/fields.csv`: one row per UKB field, grouped across instances and array positions.",
        "- `datasets/<dataset_id>/codings.csv`: decoded categorical value mappings parsed from the "
        "companion `.r` file when available.",
        "- `datasets/<dataset_id>/missing_codings.csv`: coding IDs referenced by the HTML dictionary but "
        "not defined in the companion `.r` file.",
        "- `participants/<eid_prefix>/<eid>/<dataset_id>.json`: one participant's non-missing raw values "
        "for one dataset.",
        "- `withdrawals/withdrawn_eids.csv`: normalized withdrawal list with a header.",
        "",
        "## Naming",
        "",
        "- `raw_column`: exact name from the `.tab` header, such as `f.31.0.0`.",
        "- `udi`: UKB UDI form, such as `31-0.0`.",
        "- `ml_feature_name`: a deterministic alias built from the field description plus the field, "
        "instance, and array identifiers, such as `sex__f31_i0_a0`.",
        "- `eid_prefix`: the first three characters of `eid`, used to avoid putting all participant "
        "directories in one parent directory.",
        "",
        "## Participant JSON",
        "",
        "Participant files keep only non-missing values. Empty strings and `NA` are omitted, and all "
        "retained values remain raw strings from the `.tab` file.",
        "",
        "## Dataset Summary",
        "",
        "| Dataset | Columns | Participant JSON files | Schema mode | HTML dictionary | R codings | "
        "Missing codings | Notes |",
        "| --- | ---: | ---: | --- | --- | --- | ---: | --- |",
    ]

    for item in dataset_metadata:
        if item["schema_mode"] == "full_dictionary":
            notes = "Full field descriptions and codings parsed."
        else:
            notes = "Header-only schema; no companion dictionary found."
        lines.append(
            f"| {item['dataset_id']} | {item['column_count']} | "
            f"{item.get('participant_json_count', 0)} | {item['schema_mode']} | "
            f"{'yes' if item['has_html_dictionary'] else 'no'} | "
            f"{'yes' if item['has_r_codings'] else 'no'} | "
            f"{item.get('missing_coding_count', 0)} | {notes} |"
        )

    lines.extend(
        [
            "",
            "## Withdrawals",
            "",
            f"- Parsed {withdrawal_metadata['count']} withdrawn participant IDs into "
            f"`{withdrawal_metadata['output_file']}`.",
            "- By default participant JSON files include withdrawn IDs. Use `--exclude-withdrawn` to skip "
            "them during generation.",
            "",
            "## Recommended downstream usage",
            "",
            "1. Read `manifest.json` to discover available datasets and output paths.",
            "2. Use `participants/<eid_prefix>/<eid>/<dataset_id>.json` for one person's values.",
            "3. Use `datasets/<dataset_id>/columns.csv` or `columns.jsonl` to map participant JSON keys "
            "back to raw UKB fields.",
            "4. Join `coding_id` against `datasets/<dataset_id>/codings.csv` for categorical decoding when needed.",
            "5. Filter out any `eid` that appears in `withdrawals/withdrawn_eids.csv` before task "
            "construction, unless the parser was run with `--exclude-withdrawn`.",
        ]
    )

    path = output_root / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse UK Biobank annotation exports into dataset metadata and "
            "participants/<eid_prefix>/<eid>/<dataset_id>.json files."
        )
    )
    parser.add_argument("input_root", type=Path, help="Raw UKB root or its annotations subdirectory")
    parser.add_argument("output_root", type=Path, help="Destination for the parsed annotation bundle")
    parser.add_argument(
        "--exclude-withdrawn",
        action="store_true",
        help="Skip participant JSON files for IDs listed in withdraw*.txt",
    )
    parser.add_argument("--limit-rows", type=int, default=None, help="Process at most this many rows per .tab file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_root = resolve_annotation_root(args.input_root)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    tab_paths = sorted(annotation_root.glob("ukb*.tab"))
    withdrawal_metadata, withdrawn_eids = parse_withdrawals(annotation_root, output_root)

    dataset_metadata = []
    for tab_path in tab_paths:
        metadata, columns = build_dataset_outputs(annotation_root, output_root, tab_path)
        participant_metadata = parse_participants(
            output_root=output_root,
            tab_path=tab_path,
            dataset_id=metadata["dataset_id"],
            columns=columns,
            withdrawn_eids=withdrawn_eids,
            exclude_withdrawn=args.exclude_withdrawn,
            limit_rows=args.limit_rows,
        )
        metadata.update(participant_metadata)
        dataset_metadata.append(metadata)

    dataset_metadata.sort(key=lambda item: item["dataset_id"])
    write_manifest(
        output_root,
        annotation_root,
        dataset_metadata,
        withdrawal_metadata,
        args.exclude_withdrawn,
        args.limit_rows,
    )
    write_readme(output_root, dataset_metadata, withdrawal_metadata)


if __name__ == "__main__":
    main()
