from datetime import datetime
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from utils import cut_ukb_sleep_with_asleep as cutter


def write_fake_cwa(path, timestamp):
    encoded = (
        ((timestamp.year - 2000) << 26)
        | (timestamp.month << 22)
        | (timestamp.day << 17)
        | (timestamp.hour << 12)
        | (timestamp.minute << 6)
        | timestamp.second
    )
    block = bytearray(cutter.CWA_BLOCK_BYTES)
    block[:2] = b"AX"
    struct.pack_into("<I", block, 14, encoded)
    path.write_bytes(block)


class CutUkbSleepWithAsleepTest(unittest.TestCase):
    def test_auto_time_shift_uses_uk_timezone_offset_from_cwa_timestamp(self):
        with TemporaryDirectory() as tmp:
            summer_path = Path(tmp) / "summer.cwa"
            winter_path = Path(tmp) / "winter.cwa"
            write_fake_cwa(summer_path, datetime(2015, 9, 18, 10, 0, 6))
            write_fake_cwa(winter_path, datetime(2015, 11, 13, 10, 0, 8))

            self.assertEqual(cutter.resolve_time_shift_hours(summer_path, "auto"), 1)
            self.assertEqual(cutter.resolve_time_shift_hours(winter_path, "auto"), 0)

    def test_process_file_reads_raw_cwa_in_device_time_when_asleep_times_are_shifted(self):
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
                return None, blocks, blocks.copy(), None, None

            def fake_read_cwa_signal_segment(path, start, end_exclusive):
                read_calls.append((path, start, end_exclusive))
                return (
                    np.ones((2, 3), dtype=np.float32),
                    pd.DatetimeIndex(["2024-01-01 23:00:00", "2024-01-01 23:00:30"]),
                )

            def fake_write_night_npz(output_path, segment, times):
                write_calls.append((output_path, segment, times))

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
                    (fake_get_parsed_data, fake_transform_data2model_input, fake_get_sleep_windows),
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
            self.assertEqual(rows[0]["start_time"], "2024-01-02 00:00:00")
            self.assertEqual(rows[0]["end_time_exclusive"], "2024-01-02 00:01:00")

    def test_process_file_resolves_auto_time_shift_before_calling_asleep(self):
        with TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            source_dir.mkdir()
            cwa_path = source_dir / "sample.cwa"
            write_fake_cwa(cwa_path, datetime(2015, 9, 18, 10, 0, 6))
            output_dir = Path(tmp) / "out"

            asleep_time_shifts = []
            read_calls = []

            def fake_get_parsed_data(raw_data_path, info_data_path, resample_hz, args):
                asleep_time_shifts.append(args.time_shift)
                return pd.DataFrame(), None

            def fake_transform_data2model_input(data2model_path, times_path, non_wear_path, data, args):
                times = np.array(
                    [
                        np.datetime64("2015-09-19T00:25:05"),
                        np.datetime64("2015-09-19T00:25:35"),
                    ]
                )
                return np.zeros((2, 3, 900), dtype=np.float32), times, np.zeros(2, dtype=bool)

            def fake_get_sleep_windows(data2model, times, non_wear, args):
                blocks = pd.DataFrame(
                    {
                        "start": [pd.Timestamp("2015-09-19 00:25:05")],
                        "end": [pd.Timestamp("2015-09-19 00:25:35")],
                    }
                )
                return None, blocks, blocks.copy(), None, None

            def fake_read_cwa_signal_segment(path, start, end_exclusive):
                read_calls.append((path, start, end_exclusive))
                return (
                    np.ones((2, 3), dtype=np.float32),
                    pd.DatetimeIndex(["2015-09-18 23:25:05", "2015-09-18 23:25:35"]),
                )

            original_read = cutter.read_cwa_signal_segment
            cutter.read_cwa_signal_segment = fake_read_cwa_signal_segment
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
                    (fake_get_parsed_data, fake_transform_data2model_input, fake_get_sleep_windows),
                )
            finally:
                cutter.read_cwa_signal_segment = original_read

            self.assertEqual(asleep_time_shifts, ["+1"])
            self.assertEqual(
                read_calls,
                [
                    (
                        cwa_path,
                        pd.Timestamp("2015-09-18 23:25:05"),
                        pd.Timestamp("2015-09-18 23:26:05"),
                    )
                ],
            )
            self.assertEqual(rows[0]["time_shift_hours"], 1)
            self.assertTrue(rows[0]["asleep_cache_dir"].endswith("time_shift_+1"))
