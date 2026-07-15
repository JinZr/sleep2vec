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


def test_assign_ranks_truncates_before_numbering():
    rows = _rows(["0.2", "0.9", "0.5", "0.7"])

    ranked = artifacts.assign_ranks(rows, key="score", reverse=True, top_k=2)

    assert [(row["run_id"], row["rank"]) for row in ranked] == [("run-001", 1), ("run-003", 2)]
    # Rows beyond top_k are neither returned nor rank-annotated.
    assert "rank" not in rows[0]
    assert "rank" not in rows[2]


def test_assign_ranks_writes_ranks_in_place_without_reordering_input():
    rows = _rows(["0.2", "0.9"])

    ranked = artifacts.assign_ranks(rows, key="score", reverse=True)

    assert ranked[0] is rows[1]
    assert ranked[1] is rows[0]
    assert [row["run_id"] for row in rows] == ["run-000", "run-001"]
    assert rows[1]["rank"] == 1


def test_assign_ranks_writes_rank_metric_when_requested():
    rows = [{"run_id": "run-000", "val_auroc": "0.8"}, {"run_id": "run-001", "val_auroc": "0.9"}]

    ranked = artifacts.assign_ranks(rows, key="val_auroc", reverse=True, rank_metric="val_auroc")

    assert [(row["rank"], row["rank_metric"]) for row in ranked] == [(1, "val_auroc"), (2, "val_auroc")]


def test_assign_ranks_sinks_unparseable_scores():
    rows = _rows(["0.5", "not-a-number", "0.9"])

    ranked = artifacts.assign_ranks(rows, key="score", reverse=True)

    assert [row["run_id"] for row in ranked] == ["run-002", "run-000", "run-001"]
