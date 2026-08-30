from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SIDECAR_MODULES = [
    f"{package}.{task}"
    for package in ("data", "sleep2vec2.data", "sleep2expert.data")
    for task in ("survival", "multilabel")
]


@pytest.fixture(params=SIDECAR_MODULES)
def sidecar_module(request):
    return importlib.import_module(request.param)


@pytest.fixture
def table_loader(sidecar_module):
    task = sidecar_module.__name__.rsplit(".", 1)[-1]
    return getattr(sidecar_module, f"load_{task}_label_table")


@pytest.fixture
def label_config(tmp_path):
    columns_path = tmp_path / "diseases.txt"
    columns_path.write_text("d1\nd2\n")
    config = SimpleNamespace(key_column="eid", disease_columns_index=columns_path)
    for field in ("event_time_index", "is_event_index", "label_index", "has_label_index"):
        path = tmp_path / f"{field}.csv"
        path.write_text("eid,d1,d2\n001,1,0\nNA,0,1\n")
        setattr(config, field, path)
    return config


def test_sidecar_preserves_string_keys_and_independent_float32_rows(sidecar_module, tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("eid,d1,d2\n 001 ,1,2\n1,3,4\nNA,5,6\nNULL,7,8\n")

    values = sidecar_module._load_sidecar(path, "eid", ["d1", "d2"], "labels")

    assert list(values) == ["001", "1", "NA", "NULL"]
    for key, expected in zip(values, ([1, 2], [3, 4], [5, 6], [7, 8])):
        assert values[key].dtype == np.float32
        np.testing.assert_array_equal(values[key], expected)
    assert not np.shares_memory(values["001"], values["1"])
    values["001"][0] = 99
    np.testing.assert_array_equal(values["1"], [3, 4])


def test_sidecar_converts_mixed_scalars_and_float32_boundaries(sidecar_module, tmp_path):
    path = tmp_path / "labels.csv"
    columns = [f"d{index}" for index in range(8)]
    path.write_text(
        "eid," + ",".join(columns) + "\n"
        "001,True,word, 1.5 ,NaN,inf,-inf,1e39,16777217\n"
        "NA,False,2,3,NULL,0,4,-1e39,1.00000001\n"
    )

    with np.errstate(over="ignore"):
        values = sidecar_module._load_sidecar(path, "eid", columns, "labels")

    np.testing.assert_array_equal(values["001"], [1, np.nan, 1.5, np.nan, np.inf, -np.inf, np.inf, 16777216])
    np.testing.assert_array_equal(values["NA"], [0, 2, 3, np.nan, 0, 4, -np.inf, 1])
    assert all(value.dtype == np.float32 for value in values.values())


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ("001,9223372036854775807,18446744073709551615\n", [2**63, 2**64]),
        ("001,-9223372036854775808,1e-45\n", [-(2**63), 1e-45]),
    ],
)
def test_sidecar_preserves_large_integer_and_small_float_conversion(sidecar_module, tmp_path, rows, expected):
    path = tmp_path / "labels.csv"
    path.write_text("eid,d1,d2\n" + rows)

    values = sidecar_module._load_sidecar(path, "eid", ["d1", "d2"], "labels")

    np.testing.assert_array_equal(values["001"], np.asarray(expected, dtype=np.float32))


@pytest.mark.parametrize("rows", ["001,1,0\n 001 ,0,1\n,1,0\n", ",1,0\n001,1,0\n001,0,1\n"])
def test_sidecar_reports_first_key_error_in_row_order(sidecar_module, tmp_path, rows):
    path = tmp_path / "labels.csv"
    path.write_text("eid,d1,d2\n" + rows)
    task = sidecar_module.__name__.rsplit(".", 1)[-1].capitalize()
    expected = (
        "labels contains duplicate key '001'."
        if rows.startswith("001")
        else f"{task} key column 'eid' contains an empty value."
    )

    with pytest.raises(ValueError) as error:
        sidecar_module._load_sidecar(path, "eid", ["d1", "d2"], "labels")

    assert str(error.value) == expected


@pytest.mark.parametrize("columns", ["eid,d2,d1", "d1,eid,d2", "eid,d1", "eid,d1,d2,d3", "eid,d1,d1"])
def test_sidecar_rejects_noncanonical_headers_before_key_errors(sidecar_module, tmp_path, columns):
    path = tmp_path / "labels.csv"
    path.write_text(columns + "\n" + ",".join("" for _ in columns.split(",")) + "\n")

    with pytest.raises(ValueError) as error:
        sidecar_module._load_sidecar(path, "eid", ["d1", "d2"], "labels")

    assert str(error.value) == "labels columns must exactly match [key_column] + disease_columns_index."


def test_sidecar_distinguishes_header_only_and_empty_files(sidecar_module, tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("eid,d1,d2\n")
    assert sidecar_module._load_sidecar(path, "eid", ["d1", "d2"], "labels") == {}
    path.write_text("")
    with pytest.raises(pd.errors.EmptyDataError):
        sidecar_module._load_sidecar(path, "eid", ["d1", "d2"], "labels")


def test_label_table_accepts_empty_matching_sidecars(table_loader, label_config):
    for field in ("event_time_index", "is_event_index", "label_index", "has_label_index"):
        getattr(label_config, field).write_text("eid,d1,d2\n")

    table = table_loader(label_config, expected_output_dim=2)

    assert table.label_names == ["d1", "d2"]
    assert table.has_label == {}


def test_label_table_matches_reordered_sidecars_by_key(table_loader, label_config):
    label_config.is_event_index.write_text("eid,d1,d2\nNA,0,1\n001,1,0\n")
    label_config.has_label_index.write_text("eid,d1,d2\nNA,0,1\n001,1,0\n")

    table = table_loader(label_config, expected_output_dim=2)

    assert list(table.has_label) == ["NA", "001"]
    np.testing.assert_array_equal(table.has_label["001"], [1, 0])
    np.testing.assert_array_equal(table.has_label["NA"], [0, 1])


def test_label_table_dimension_and_keyset_errors_precede_label_errors(sidecar_module, table_loader, label_config):
    label_config.event_time_index.write_text("eid,d1,d2\n001,bad,bad\nNA,bad,bad\n")
    label_config.label_index.write_text("eid,d1,d2\n001,bad,bad\nNA,bad,bad\n")
    label_config.has_label_index.write_text("eid,d1,d2\nother,1,1\n")
    task = sidecar_module.__name__.rsplit(".", 1)[-1].capitalize()

    with pytest.raises(ValueError) as error:
        table_loader(label_config, expected_output_dim=3)
    assert str(error.value) == f"{task} output_dim (3) must match disease column count (2)."

    with pytest.raises(ValueError) as error:
        table_loader(label_config, expected_output_dim=2)
    fields = "event_time, is_event, and has_label" if task == "Survival" else "label_index and has_label_index"
    assert str(error.value) == f"{task} sidecar key sets must match across {fields}."


def test_label_table_loads_files_before_validating_output_dim(sidecar_module, table_loader, label_config):
    task = sidecar_module.__name__.rsplit(".", 1)[-1]
    first_field = "event_time_index" if task == "survival" else "label_index"
    getattr(label_config, first_field).write_text("eid,d1,d2\n001,1,0\n001,0,1\n")
    label_config.has_label_index.write_text("invalid_header\n")
    label_config.disease_columns_index.write_text("d1\nd1\n")

    with pytest.raises(ValueError) as error:
        table_loader(label_config, expected_output_dim=3)
    assert str(error.value) == "disease_columns_index contains duplicate disease column 'd1'."

    label_config.disease_columns_index.write_text("d1\nd2\n")
    with pytest.raises(ValueError) as error:
        table_loader(label_config, expected_output_dim=3)
    assert str(error.value) == f"{first_field} contains duplicate key '001'."

    getattr(label_config, first_field).write_text("eid,d1,d2\n001,1,0\nNA,0,1\n")
    if task == "survival":
        label_config.is_event_index.write_text("eid,d1,d2\n001,1,0\n001,0,1\n")
        with pytest.raises(ValueError) as error:
            table_loader(label_config, expected_output_dim=3)
        assert str(error.value) == "is_event_index contains duplicate key '001'."
        label_config.is_event_index.write_text("eid,d1,d2\n001,1,0\nNA,0,1\n")

    with pytest.raises(ValueError) as error:
        table_loader(label_config, expected_output_dim=3)
    assert str(error.value) == "has_label_index columns must exactly match [key_column] + disease_columns_index."


@pytest.mark.parametrize("module_name", [module for module in SIDECAR_MODULES if module.endswith("survival")])
def test_survival_preserves_masked_missing_and_nonbinary_values(module_name, label_config):
    module = importlib.import_module(module_name)
    label_config.event_time_index.write_text("eid,d1,d2\n001,bad,-3\nNA,inf,NaN\n")
    label_config.is_event_index.write_text("eid,d1,d2\nNA,2,bad\n001,bad,-1\n")
    label_config.has_label_index.write_text("eid,d1,d2\nNA,2,NaN\n001,0,1\n")

    table = module.load_survival_label_table(label_config, expected_output_dim=2)

    np.testing.assert_array_equal(table.event_time["001"], [np.nan, -3])
    np.testing.assert_array_equal(table.event_time["NA"], [np.inf, np.nan])
    np.testing.assert_array_equal(table.is_event["NA"], [2, np.nan])


@pytest.mark.parametrize("module_name", [module for module in SIDECAR_MODULES if module.endswith("survival")])
def test_survival_rounds_mask_before_threshold_and_reports_first_subject(module_name, label_config):
    module = importlib.import_module(module_name)
    label_config.event_time_index.write_text("eid,d1,d2\n001,bad,0\nNA,bad,0\n")
    label_config.has_label_index.write_text("eid,d1,d2\nNA,0,1\n001,0.50000001,1\n")
    assert module.load_survival_label_table(label_config).has_label["001"][0] == np.float32(0.5)

    label_config.has_label_index.write_text("eid,d1,d2\nNA,1,1\n001,0.5000001,1\n")
    with pytest.raises(ValueError) as error:
        module.load_survival_label_table(label_config)
    assert str(error.value) == "Survival labels for key '001' contain missing event_time or is_event values."


@pytest.mark.parametrize("module_name", [module for module in SIDECAR_MODULES if module.endswith("multilabel")])
def test_multilabel_rounds_before_binary_check_and_ignores_masked_values(module_name, label_config):
    module = importlib.import_module(module_name)
    label_config.label_index.write_text("eid,d1,d2\n001,bad,1.00000001\nNA,-inf,1e39\n")
    label_config.has_label_index.write_text("eid,d1,d2\nNA,0,0\n001,0,1.00000001\n")

    with np.errstate(over="ignore"):
        table = module.load_multilabel_label_table(label_config)

    np.testing.assert_array_equal(table.disease_label["001"], [np.nan, 1])
    np.testing.assert_array_equal(table.disease_label["NA"], [-np.inf, np.inf])
    np.testing.assert_array_equal(table.has_label["001"], [0, 1])


@pytest.mark.parametrize("module_name", [module for module in SIDECAR_MODULES if module.endswith("multilabel")])
@pytest.mark.parametrize(
    ("labels", "mask", "expected"),
    [
        ("bad,2", "NaN,2", "Multilabel has_label for key '001' contains missing values."),
        ("bad,2", "1,2", "Multilabel has_label for key '001' must be 0 or 1."),
        ("bad,2", "1,1", "Multilabel labels for key '001' contain missing values where has_label is true."),
        ("2,0", "1,1", "Multilabel labels for key '001' must be 0 or 1 where has_label is true."),
    ],
)
def test_multilabel_error_order_is_per_subject(module_name, label_config, labels, mask, expected):
    module = importlib.import_module(module_name)
    label_config.label_index.write_text(f"eid,d1,d2\n001,{labels}\nNA,bad,bad\n")
    label_config.has_label_index.write_text(f"eid,d1,d2\nNA,NaN,2\n001,{mask}\n")

    with pytest.raises(ValueError) as error:
        module.load_multilabel_label_table(label_config)

    assert str(error.value) == expected
