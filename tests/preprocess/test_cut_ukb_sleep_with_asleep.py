import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils import cut_ukb_sleep_with_asleep as cutter


class FakeSleepWindows:
    @staticmethod
    def find_sleep_block_duration(sleep_df):
        rows = []
        labels = list(sleep_df["label"])
        idx = 0
        while idx < len(labels):
            end = idx + 1
            while end < len(labels) and labels[end] == labels[idx]:
                end += 1
            rows.append([labels[idx], end - idx, idx])
            idx = end
        return np.asarray(rows)

    @staticmethod
    def find_valid_sleep_blocks(counter, epoch_length):
        return [idx for idx, row in enumerate(counter) if row[0] == cutter.IS_SLEEP_LABEL]

    @staticmethod
    def find_gaps2fill(valid_sleep_block_idxes, epoch_length, counter):
        return []

    @staticmethod
    def fill_gaps(my_df, counter, gap2fill):
        return my_df


class CutUkbSleepWithAsleepTest(unittest.TestCase):
    def test_parse_args_defaults_num_workers_to_one(self):
        args = cutter.parse_args(["input.cwa", "out"])

        self.assertEqual(args.num_workers, 1)

    def test_parse_args_rejects_non_positive_num_workers(self):
        with patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            cutter.parse_args(["input.cwa", "out", "--num-workers", "0"])

    def test_device_to_local_times_uses_uk_summer_and_winter_offsets(self):
        local_times = cutter.device_to_local_times(
            [
                pd.Timestamp("2015-09-18 10:00:06"),
                pd.Timestamp("2015-11-13 10:00:08"),
            ]
        )

        self.assertEqual(local_times[0].isoformat(), "2015-09-18T11:00:06+01:00")
        self.assertEqual(local_times[1].isoformat(), "2015-11-13T10:00:08+00:00")

    def test_fall_back_repeated_hour_keeps_distinct_offsets_and_filenames(self):
        local_times = cutter.device_to_local_times(
            [
                pd.Timestamp("2015-10-25 00:30:00"),
                pd.Timestamp("2015-10-25 01:30:00"),
            ]
        )

        self.assertEqual(local_times[0].isoformat(), "2015-10-25T01:30:00+01:00")
        self.assertEqual(local_times[1].isoformat(), "2015-10-25T01:30:00+00:00")
        self.assertEqual(cutter.timestamp_for_filename(local_times[0]), "20151025T013000+0100")
        self.assertEqual(cutter.timestamp_for_filename(local_times[1]), "20151025T013000+0000")

    def test_spring_forward_skips_nonexistent_local_hour(self):
        local_times = cutter.device_to_local_times(
            [
                pd.Timestamp("2015-03-29 00:30:00"),
                pd.Timestamp("2015-03-29 01:30:00"),
            ]
        )

        self.assertEqual(local_times[0].isoformat(), "2015-03-29T00:30:00+00:00")
        self.assertEqual(local_times[1].isoformat(), "2015-03-29T02:30:00+01:00")

    def test_process_file_reads_raw_cwa_in_device_time_when_fixed_shift_is_requested(self):
        with TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            source_dir.mkdir()
            cwa_path = source_dir / "sample.cwa"
            cwa_path.touch()
            output_dir = Path(tmp) / "out"

            read_calls = []
            write_calls = []

            def fake_get_parsed_data(raw_data_path, info_data_path, resample_hz, args):
                return pd.DataFrame(), None

            def fake_transform_data2model_input(data2model_path, times_path, non_wear_path, data, args):
                times = np.array(
                    [
                        np.datetime64("2024-01-02T00:00:00"),
                        np.datetime64("2024-01-02T00:00:30"),
                    ]
                )
                return np.zeros((2, 3, 900), dtype=np.float32), times, np.zeros(2, dtype=bool)

            def fake_get_sleep_windows(data2model, times, non_wear, args):
                blocks = pd.DataFrame(
                    {
                        "start": [pd.Timestamp("2024-01-02 00:00:00")],
                        "end": [pd.Timestamp("2024-01-02 00:00:30")],
                    }
                )
                return np.asarray([1, 1]), blocks, blocks.copy(), None, None

            def fake_read_cwa_signal_segment(path, start, end_exclusive):
                read_calls.append((path, start, end_exclusive))
                return (
                    np.ones((2, 3), dtype=np.float32),
                    pd.DatetimeIndex(["2024-01-01 23:00:00", "2024-01-01 23:00:30"]),
                )

            def fake_write_night_npz(output_path, segment, times, device_times):
                write_calls.append((output_path, segment, times, device_times))

            original_read = cutter.read_cwa_signal_segment
            original_write = cutter.write_night_npz
            cutter.read_cwa_signal_segment = fake_read_cwa_signal_segment
            cutter.write_night_npz = fake_write_night_npz
            try:
                rows = cutter.process_file(
                    cwa_path,
                    source_dir,
                    output_dir,
                    SimpleNamespace(
                        force_run=False,
                        force_download=False,
                        pytorch_device="cpu",
                        time_shift="+1",
                        overwrite=True,
                        remove_cache=False,
                    ),
                    (fake_get_parsed_data, fake_transform_data2model_input, fake_get_sleep_windows, FakeSleepWindows),
                )
            finally:
                cutter.read_cwa_signal_segment = original_read
                cutter.write_night_npz = original_write

            self.assertEqual(
                read_calls,
                [
                    (
                        cwa_path,
                        pd.Timestamp("2024-01-01 23:00:00"),
                        pd.Timestamp("2024-01-01 23:01:00"),
                    )
                ],
            )
            self.assertEqual(
                list(write_calls[0][2]),
                [
                    pd.Timestamp("2024-01-02 00:00:00"),
                    pd.Timestamp("2024-01-02 00:00:30"),
                ],
            )
            self.assertEqual(
                list(write_calls[0][3]),
                [
                    pd.Timestamp("2024-01-01 23:00:00"),
                    pd.Timestamp("2024-01-01 23:00:30"),
                ],
            )
            self.assertEqual(rows[0]["start_time"], "2024-01-02T00:00:00")
            self.assertEqual(rows[0]["device_start_time"], "2024-01-01T23:00:00")
            self.assertEqual(rows[0]["timezone_mode"], "fixed_shift_+1")

    def test_process_file_auto_mode_uses_dynamic_offsets_across_fall_back(self):
        with TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            source_dir.mkdir()
            cwa_path = source_dir / "sample.cwa"
            cwa_path.touch()
            output_dir = Path(tmp) / "out"

            asleep_time_shifts = []
            read_calls = []
            write_calls = []

            def fake_get_parsed_data(raw_data_path, info_data_path, resample_hz, args):
                asleep_time_shifts.append(args.time_shift)
                return pd.DataFrame(), None

            def fake_transform_data2model_input(data2model_path, times_path, non_wear_path, data, args):
                times = np.array(
                    [
                        np.datetime64("2015-10-25T00:59:30"),
                        np.datetime64("2015-10-25T01:00:00"),
                    ]
                )
                return np.zeros((2, 3, 900), dtype=np.float32), times, np.zeros(2, dtype=bool)

            def fake_get_sleep_windows(data2model, times, non_wear, args):
                return np.asarray([1, 1]), pd.DataFrame(), pd.DataFrame(), None, None

            def fake_read_cwa_signal_segment(path, start, end_exclusive):
                read_calls.append((path, start, end_exclusive))
                return (
                    np.ones((2, 3), dtype=np.float32),
                    pd.DatetimeIndex(["2015-10-25 00:59:30", "2015-10-25 01:00:00"]),
                )

            def fake_write_night_npz(output_path, segment, times, device_times):
                write_calls.append((output_path, segment, times, device_times))

            original_read = cutter.read_cwa_signal_segment
            original_write = cutter.write_night_npz
            cutter.read_cwa_signal_segment = fake_read_cwa_signal_segment
            cutter.write_night_npz = fake_write_night_npz
            try:
                rows = cutter.process_file(
                    cwa_path,
                    source_dir,
                    output_dir,
                    SimpleNamespace(
                        force_run=False,
                        force_download=False,
                        pytorch_device="cpu",
                        time_shift="auto",
                        overwrite=True,
                        remove_cache=False,
                    ),
                    (fake_get_parsed_data, fake_transform_data2model_input, fake_get_sleep_windows, FakeSleepWindows),
                )
            finally:
                cutter.read_cwa_signal_segment = original_read
                cutter.write_night_npz = original_write

            self.assertEqual(asleep_time_shifts, ["0"])
            self.assertEqual(
                read_calls,
                [
                    (
                        cwa_path,
                        pd.Timestamp("2015-10-25 00:59:30"),
                        pd.Timestamp("2015-10-25 01:00:30"),
                    )
                ],
            )
            self.assertEqual(
                [t.isoformat() for t in write_calls[0][2]],
                [
                    "2015-10-25T01:59:30+01:00",
                    "2015-10-25T01:00:00+00:00",
                ],
            )
            self.assertEqual(rows[0]["start_time"], "2015-10-25T01:59:30+01:00")
            self.assertEqual(rows[0]["end_time_exclusive"], "2015-10-25T01:00:30+00:00")
            self.assertEqual(rows[0]["device_start_time"], "2015-10-25T00:59:30")
            self.assertEqual(rows[0]["device_end_time_exclusive"], "2015-10-25T01:00:30")
            self.assertEqual(rows[0]["start_utc_offset_hours"], 1)
            self.assertEqual(rows[0]["end_utc_offset_hours"], 0)
            self.assertEqual(rows[0]["timezone_mode"], "auto")
            self.assertTrue(rows[0]["asleep_cache_dir"].endswith("timezone_auto_device"))
            self.assertIn("+0100", Path(rows[0]["output_path"]).name)

    def test_configure_asleep_runtime_uses_output_cache_and_single_loader_worker(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_model = tmp_path / "source" / "ssl.joblib.lzma"
            source_model.parent.mkdir()
            source_model.write_bytes(b"model")
            model_cache_dir = tmp_path / "_asleep_models"

            asleep_package = ModuleType("asleep")
            asleep_package.__path__ = []
            get_sleep_module = ModuleType("asleep.get_sleep")
            models_module = ModuleType("asleep.models")
            sslmodel_module = ModuleType("asleep.sslmodel")
            load_calls = []

            def load_model(model_path, force_download=False):
                load_calls.append((Path(model_path), force_download))
                return "loaded"

            class FakeDataLoader:
                def __init__(self, *args, **kwargs):
                    self.args = args
                    self.kwargs = kwargs

            get_sleep_module.load_model = load_model
            models_module.DataLoader = FakeDataLoader

            with patch.dict(
                sys.modules,
                {
                    "asleep": asleep_package,
                    "asleep.get_sleep": get_sleep_module,
                    "asleep.models": models_module,
                    "asleep.sslmodel": sslmodel_module,
                },
            ):
                cutter.configure_asleep_runtime(model_cache_dir)
                loaded = get_sleep_module.load_model(str(source_model))
                loader = models_module.DataLoader("dataset", num_workers=8)

            self.assertEqual(loaded, "loaded")
            self.assertEqual(load_calls, [(model_cache_dir / "ssl.joblib.lzma", False)])
            self.assertEqual((model_cache_dir / "ssl.joblib.lzma").read_bytes(), b"model")
            self.assertEqual(sslmodel_module.torch_cache_path, model_cache_dir / "torch_hub_cache")
            self.assertEqual(loader.args, ("dataset",))
            self.assertEqual(loader.kwargs["num_workers"], 0)

    def test_warm_asleep_model_cache_loads_ssl_model_and_torch_hub_repo(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            package_dir = tmp_path / "package"
            package_dir.mkdir()
            (package_dir / "ssl.joblib.lzma").write_bytes(b"model")
            model_cache_dir = tmp_path / "_asleep_models"

            asleep_package = ModuleType("asleep")
            asleep_package.__path__ = []
            get_sleep_module = ModuleType("asleep.get_sleep")
            models_module = ModuleType("asleep.models")
            sslmodel_module = ModuleType("asleep.sslmodel")
            load_calls = []
            sslnet_calls = []

            def load_model(model_path, force_download=False):
                load_calls.append((Path(model_path), force_download))
                return SimpleNamespace(repo_tag="v1.2.3")

            class FakeDataLoader:
                pass

            get_sleep_module.__file__ = str(package_dir / "get_sleep.py")
            get_sleep_module.load_model = load_model
            models_module.DataLoader = FakeDataLoader
            sslmodel_module.get_sslnet = lambda **kwargs: sslnet_calls.append(kwargs)

            with patch.dict(
                sys.modules,
                {
                    "asleep": asleep_package,
                    "asleep.get_sleep": get_sleep_module,
                    "asleep.models": models_module,
                    "asleep.sslmodel": sslmodel_module,
                },
            ):
                cutter.warm_asleep_model_cache(model_cache_dir, force_download=False)

            self.assertEqual(load_calls, [(model_cache_dir / "ssl.joblib.lzma", False)])
            self.assertEqual(sslnet_calls, [{"tag": "v1.2.3", "pretrained": False}])

    def test_process_files_parallel_preserves_manifest_order(self):
        with TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            source_dir.mkdir()
            first = source_dir / "first.cwa"
            second = source_dir / "second.cwa"
            first.touch()
            second.touch()
            args = SimpleNamespace(
                num_workers=2,
                input_root=source_dir,
                output_dir=Path(tmp) / "out",
                pytorch_device=None,
                force_download=True,
            )
            warm_calls = []

            class FakeFuture:
                def __init__(self, result):
                    self._result = result

                def result(self):
                    return self._result

            class FakeExecutor:
                def __init__(self, max_workers):
                    self.max_workers = max_workers

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return False

                def submit(self, fn, *args):
                    index = args[0]
                    path = args[2]
                    return FakeFuture((index, [{"relative_source": path.name}]))

            original_executor = cutter.ProcessPoolExecutor
            original_as_completed = cutter.as_completed
            original_warm = cutter.warm_asleep_model_cache
            try:
                cutter.ProcessPoolExecutor = FakeExecutor
                cutter.as_completed = lambda futures: list(reversed(futures))
                cutter.warm_asleep_model_cache = lambda *args: warm_calls.append(args)
                with patch("sys.stdout", io.StringIO()):
                    rows = cutter.process_files([first, second], args)
            finally:
                cutter.ProcessPoolExecutor = original_executor
                cutter.as_completed = original_as_completed
                cutter.warm_asleep_model_cache = original_warm

            self.assertEqual([row["relative_source"] for row in rows], ["first.cwa", "second.cwa"])
            self.assertEqual(args.pytorch_device, "cpu")
            self.assertFalse(args.force_download)
            self.assertEqual(warm_calls, [(Path(tmp) / "out" / "_asleep_models", True)])
