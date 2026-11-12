from agent_tools import run_artifacts as artifacts


def _rows(scores):
    return [{"run_id": f"run-{index:03d}", "score": score} for index, score in enumerate(scores)]


def test_assign_ranks_max_mode_orders_descending_from_one():
    rows = _rows(["0.2", "0.9", "0.5"])

    ranked = artifacts.assign_ranks(rows, key="score", reverse=True)

    assert [(row["run_id"], row["rank"]) for row in ranked] == [
        ("run-001", 1),
        ("run-002", 2),
        ("run-000", 3),
    ]


def test_assign_ranks_min_mode_orders_ascending_from_one():
    rows = _rows(["0.2", "0.9", "0.5"])

    ranked = artifacts.assign_ranks(rows, key="score", reverse=False)

    assert [(row["run_id"], row["rank"]) for row in ranked] == [
        ("run-000", 1),
        ("run-002", 2),
        ("run-001", 3),
    ]


def test_assign_ranks_is_stable_for_tied_scores():
    rows = _rows(["0.5", "0.5", "0.5"])

    ranked = artifacts.assign_ranks(rows, key="score", reverse=True)

    assert [row["run_id"] for row in ranked] == ["run-000", "run-001", "run-002"]


def test_assign_ranks_writes_ranks_in_place_without_reordering_input():
    rows = _rows(["0.2", "0.9"])

    ranked = artifacts.assign_ranks(rows, key="score", reverse=True)

    assert ranked[0] is rows[1]
    assert ranked[1] is rows[0]
    assert [row["run_id"] for row in rows] == ["run-000", "run-001"]
    assert rows[1]["rank"] == 1
