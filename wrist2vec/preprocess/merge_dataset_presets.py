#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import pickle
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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

    loaded_lists = []
    for p in input_paths:
        data = _validate_items(p, _load_preset(p))
        loaded_lists.append(data)

    merged = _flatten(loaded_lists)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Merged {len(input_paths)} presets into {output_path}")
    print(f"Total samples: {len(merged)}")


if __name__ == "__main__":
    main()
