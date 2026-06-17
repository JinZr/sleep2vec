from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hypnodata.annotations import (
    AnnotationSignal,
    filter_events_to_sleep_stages,
    materialize_ahi_from_events,
    materialize_anchor_events,
    materialize_dense_events,
    materialize_event_table,
    read_event_csv,
    read_stage_csv,
)


def test_annotation_signal_keeps_legacy_positional_steps():
    signal = AnnotationSignal("stage5", np.asarray([0]), 1 / 30, "stage.csv", "Type", None, ["legacy_step"])

    assert signal.steps == ["legacy_step"]
    assert signal.materialization == "stage"


def test_read_stage_csv_uses_default_stage5_mapping(tmp_path: Path):
    path = tmp_path / "stage.csv"
    pd.DataFrame(
        [
            {"Type": "Wake", "Start": 0, "Duration": 30},
            {"Type": "N2", "Start": 30, "Duration": 30},
        ]
    ).to_csv(path, index=False)

    signal = read_stage_csv(
        path,
        duration_sec=90,
        epoch_sec=30,
        label_column="Type",
        start_column="Start",
        duration_column="Duration",
    )

    np.testing.assert_array_equal(signal.data, np.asarray([0, 2, -1], dtype=np.int64))
    assert signal.materialization == "stage"


def test_read_stage_csv_rejects_unaligned_stage_start(tmp_path: Path):
    path = tmp_path / "stage.csv"
    pd.DataFrame([{"Type": "N2", "Start": 5, "Duration": 30}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="not aligned"):
        read_stage_csv(
            path,
            duration_sec=90,
            epoch_sec=30,
            label_column="Type",
            start_column="Start",
            duration_column="Duration",
        )


def test_read_stage_csv_rejects_partial_stage_duration(tmp_path: Path):
    path = tmp_path / "stage.csv"
    pd.DataFrame([{"Type": "N2", "Start": 0, "Duration": 45}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="stop=45 is not aligned"):
        read_stage_csv(
            path,
            duration_sec=90,
            epoch_sec=30,
            label_column="Type",
            start_column="Start",
            duration_column="Duration",
        )


def test_read_stage_csv_rejects_overlapping_stage_epochs(tmp_path: Path):
    path = tmp_path / "stage.csv"
    pd.DataFrame(
        [
            {"Type": "N2", "Start": 0, "Duration": 60},
            {"Type": "REM", "Start": 30, "Duration": 30},
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="overlaps an existing epoch"):
        read_stage_csv(
            path,
            duration_sec=90,
            epoch_sec=30,
            label_column="Type",
            start_column="Start",
            duration_column="Duration",
        )


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ({"Type": "N2", "Start": -30, "Duration": 30}, "invalid extent"),
        ({"Type": "N2", "Start": 30, "Duration": 0}, "invalid extent"),
        ({"Type": "N2", "Start": 60, "Duration": 60}, "exceeds duration_sec"),
    ],
)
def test_read_stage_csv_rejects_invalid_or_overlong_extents(tmp_path: Path, row: dict[str, object], match: str):
    path = tmp_path / "stage.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match=match):
        read_stage_csv(
            path,
            duration_sec=90,
            epoch_sec=30,
            label_column="Type",
            start_column="Start",
            duration_column="Duration",
        )


def test_read_event_csv_maps_standard_event_rows(tmp_path: Path):
    path = tmp_path / "events.csv"
    pd.DataFrame(
        [
            {"Type": "Apnea", "Start": 1.0, "Duration": 10.0},
            {"Type": "Hypopnea", "Start": 20.0, "Duration": 5.0},
        ]
    ).to_csv(path, index=False)

    rows = read_event_csv(path, mapping={"Apnea": 0, "Hypopnea": 1})

    np.testing.assert_allclose(rows, np.asarray([[0, 1, 10], [1, 20, 5]], dtype=np.float32))


def test_event_table_materializer_preserves_empty_shape():
    signal = materialize_event_table(np.empty((0, 3)), canonical_channel="ah_event_table")

    assert signal.materialization == "event_table"
    assert signal.sfreq is None
    assert signal.data.shape == (0, 3)


def test_dense_event_materializer_marks_overlapping_bins():
    rows = np.asarray([[0, 1.2, 2.1], [1, 5.0, 1.0]], dtype=np.float32)

    signal = materialize_dense_events(rows, duration_sec=8, interval_sec=1, canonical_channel="ah_event")

    np.testing.assert_array_equal(signal.data, np.asarray([0, 1, 1, 1, 0, 1, 0, 0], dtype=np.float32))
    assert signal.sfreq == 1.0
    assert signal.steps == ["event_csv", "dense:1s"]


def test_dense_event_materializer_can_use_event_type_values():
    rows = np.asarray([[4, 1.0, 2.0]], dtype=np.float32)

    signal = materialize_dense_events(rows, duration_sec=4, interval_sec=1, canonical_channel="ah_event", value=None)

    np.testing.assert_array_equal(signal.data, np.asarray([0, 4, 4, 0], dtype=np.float32))


def test_dense_event_materializer_rejects_overlong_event_rows():
    rows = np.asarray([[0, 7.0, 2.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="exceed record duration"):
        materialize_dense_events(rows, duration_sec=8, interval_sec=1, canonical_channel="ah_event")


def test_anchor_event_materializer_splits_events_across_windows():
    rows = np.asarray([[0, 50.0, 20.0]], dtype=np.float32)

    signal = materialize_anchor_events(
        rows,
        duration_sec=120,
        window_sec=60,
        anchor_num=2,
        canonical_channel="desaturation_anchor",
    )

    assert signal.materialization == "event_anchor"
    assert signal.data.shape == (2, 6)
    np.testing.assert_allclose(signal.data[0], np.asarray([1, 50 / 60, 1, 0, 0, 0], dtype=np.float32))
    np.testing.assert_allclose(signal.data[1], np.asarray([1, 0, 10 / 60, 0, 0, 0], dtype=np.float32))
    assert signal.steps == ["event_csv", "anchor:60s:2"]


def test_anchor_event_materializer_rejects_overlong_event_rows():
    rows = np.asarray([[0, 110.0, 20.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="exceed record duration"):
        materialize_anchor_events(
            rows,
            duration_sec=120,
            window_sec=60,
            anchor_num=2,
            canonical_channel="desaturation_anchor",
        )


def test_anchor_event_materializer_rejects_anchor_overflow():
    rows = np.asarray([[0, 10.0, 5.0], [0, 20.0, 5.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="exceeds anchor_num"):
        materialize_anchor_events(
            rows,
            duration_sec=60,
            window_sec=60,
            anchor_num=1,
            canonical_channel="desaturation_anchor",
        )


def test_filter_events_to_sleep_stages_keeps_only_sleep_overlap():
    rows = np.asarray(
        [
            [0, 5.0, 5.0],
            [0, 25.0, 10.0],
            [0, 35.0, 5.0],
        ],
        dtype=np.float32,
    )
    stage = np.asarray([0, 2, 0], dtype=np.int64)

    filtered = filter_events_to_sleep_stages(rows, stage, epoch_sec=30)

    np.testing.assert_allclose(filtered, rows[1:])


def test_materialize_ahi_from_events_filters_events_and_writes_scalars():
    stage = np.asarray([0, 1, -1, 2, 0, 4], dtype=np.int64)
    rows = np.asarray(
        [
            [0, 35, 9],
            [0, 35, 10],
            [0, 0, 12],
            [0, 55, 20],
            [0, 125, 10],
            [0, 145, 10],
        ],
        dtype=np.float32,
    )

    signal = materialize_ahi_from_events(rows, stage, duration_sec=180, epoch_sec=30, interval_sec=1)

    expected = np.zeros(180, dtype=np.float32)
    expected[35:45] = 1
    expected[55:75] = 1
    expected[145:155] = 1
    np.testing.assert_array_equal(signal.data, expected)
    assert signal.materialization == "ahi"
    assert signal.output_key == "ah_event"
    assert set(signal.extra_outputs) == {"ahi", "tst"}
    np.testing.assert_allclose(signal.extra_outputs["tst"], np.asarray(90 / 3600, dtype=np.float32))
    np.testing.assert_allclose(signal.extra_outputs["ahi"], np.asarray(120.0, dtype=np.float32))
    assert "aasm_min_duration:10s" in signal.steps
    assert "stage_sleep_filter" in signal.steps


def test_materialize_ahi_from_events_rejects_overlong_rows_before_stage_filter():
    stage = np.asarray([0, 2, 0], dtype=np.int64)
    rows = np.asarray([[0, 80, 20]], dtype=np.float32)

    with pytest.raises(ValueError, match="exceed record duration"):
        materialize_ahi_from_events(rows, stage, duration_sec=90, epoch_sec=30, interval_sec=1)


def test_materialize_ahi_from_events_rejects_zero_tst():
    stage = np.asarray([0, -1], dtype=np.int64)
    rows = np.asarray([[0, 0, 10]], dtype=np.float32)

    with pytest.raises(ValueError, match="positive TST"):
        materialize_ahi_from_events(rows, stage, duration_sec=60, epoch_sec=30, interval_sec=1)
