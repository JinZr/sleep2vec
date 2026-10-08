from pathlib import Path
import sys
import zlib

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
for path in (TESTS_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("shard")
    group.addoption(
        "--shard-count",
        type=int,
        default=1,
        help="Number of shards the collected tests are split into; 1 means no sharding.",
    )
    group.addoption(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard this process runs, from 0 to --shard-count minus 1.",
    )


def pytest_configure(config: pytest.Config) -> None:
    count, index = config.getoption("shard_count"), config.getoption("shard_index")
    if count < 1 or not 0 <= index < count:
        raise pytest.UsageError(
            f"invalid shard selection --shard-count {count} --shard-index {index}: "
            "need count >= 1 and 0 <= index < count"
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Membership depends only on the node id, so every test runs in exactly one shard
    # whatever the collection order (pytest-randomly) or the number of xdist workers.
    count, index = config.getoption("shard_count"), config.getoption("shard_index")
    if count == 1:
        return
    kept, dropped = [], []
    for item in items:
        if zlib.crc32(item.nodeid.encode()) % count == index:
            kept.append(item)
        else:
            dropped.append(item)
    items[:] = kept
    config.hook.pytest_deselected(items=dropped)
