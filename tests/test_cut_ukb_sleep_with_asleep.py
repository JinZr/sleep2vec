from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

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
