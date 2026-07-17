#!/usr/bin/env python3
import argparse
from pathlib import Path
import pickle
import sys
import time
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_tools.progress import write_progress


def _load_preset(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _validate_items(path: Path, data) -> List:
    if not isinstance(data, list):
        raise TypeError(f"Expected list from {path}, got {type(data).__name__}")
    return data


def _flatten(lists: Iterable[List]) -> List:
    merged: List = []
    for lst in lists:
        merged.extend(lst)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple dataset preset pickle files into one.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input preset pickle paths (space-separated).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output preset pickle path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in args.inputs]
    output_path = Path(args.output)
    started_at = time.time()
    write_progress(
        output_path.parent,
        status="running",
        task="merge_dataset_presets",
        processed=0,
        total=len(input_paths),
        success=0,
        failed=0,
        start_time=started_at,
    )

    loaded_lists = []
    for processed, p in enumerate(input_paths, start=1):
        data = _validate_items(p, _load_preset(p))
        loaded_lists.append(data)
        write_progress(
            output_path.parent,
            status="running",
            task="merge_dataset_presets",
            processed=processed,
            total=len(input_paths),
            success=processed,
            failed=0,
            start_time=started_at,
            current_item=str(p),
        )

    merged = _flatten(loaded_lists)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Merged {len(input_paths)} presets into {output_path}")
    print(f"Total samples: {len(merged)}")
    write_progress(
        output_path.parent,
        status="completed",
        task="merge_dataset_presets",
        processed=len(input_paths),
        total=len(input_paths),
        success=len(input_paths),
        failed=0,
        start_time=started_at,
        message=f"Total samples: {len(merged)}",
    )


if __name__ == "__main__":
    main()
