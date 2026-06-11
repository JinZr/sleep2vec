#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple
import warnings
import zipfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_tools.progress import write_progress

try:
    import pyedflib  # type: ignore
except Exception:
    pyedflib = None

try:
    from scipy.signal import butter, medfilt, sosfiltfilt  # type: ignore
except Exception:
    butter = None
    medfilt = None
    sosfiltfilt = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

FRAME_ORDER: Tuple[int, ...] = (2, 3, 1, 4, 5)
VALID_FRAME_HIGHS = set(FRAME_ORDER)
HEADER_BYTES = 1024


@dataclass
class StudyMetadata:
    source_path: str
    patient_code: str = ""
    recording_start: Optional[datetime] = None
    export_start: Optional[datetime] = None
    export_offset_est_s: float = 0.0
    manufacturer: str = ""
    device_model: str = ""
    device_serial: str = ""
    probe_id: str = ""
    software_version: str = ""
    creation_date_text: str = ""
    creation_time_text: str = ""


@dataclass
class SignalSpec:
    label: str
    samples: np.ndarray
    sample_frequency: int
    dimension: str
    transducer: str = ""
    prefilter: str = ""
    description: str = ""


@dataclass
class StreamLayout:
    high_rate_order: Tuple[int, ...]
    high_rate_hz: int
    low_rate_hz: Dict[int, int]
    description: str


@dataclass
class ChannelMapping:
    ppg_red: int
    ppg_ir: int
    pat: int
    actigraphy: Optional[int]
    probe_pressure: Optional[int] = None
    probe_pressure_aux_high: Optional[int] = None
    confidence: Dict[str, str] = None  # type: ignore[assignment]
    diagnostics: Dict[str, float] = None  # type: ignore[assignment]


class ZzpDecodeError(RuntimeError):
    pass


class EdfWriteError(RuntimeError):
    pass


def _ascii_runs(data: bytes, min_len: int = 4) -> List[str]:
    return [
        m.decode("ascii", errors="ignore").strip()
        for m in re.findall(rb"[\x20-\x7e]{%d,}" % min_len, data)
        if m.strip()
    ]


def parse_patient_dat(data: bytes) -> str:
    runs = _ascii_runs(data, min_len=3)
    # Prefer the first all-digit field with at least 3 digits.
    for run in runs:
        if re.fullmatch(r"\d{3,}", run):
            return run
    return ""


def parse_log_dat(data: bytes, recording_start: Optional[datetime]) -> Dict[str, str]:
    info: Dict[str, str] = {}
    if not data:
        return info

    records: List[Tuple[datetime, str]] = []
    for off in range(512, len(data) - 63, 64):
        chunk = data[off : off + 64]
        if not chunk.strip(b"\x00"):
            continue
        text = chunk.decode("ascii", errors="ignore").strip("\x00\r\n ")
        m = re.match(r"^([A-Z][a-z]{2} [A-Z][a-z]{2}\s+\d{1,2} \d\d:\d\d:\d\d \d{4})-(.*)$", text)
        if not m:
            continue
        ts_text = " ".join(m.group(1).split())
        try:
            ts = datetime.strptime(ts_text, "%a %b %d %H:%M:%S %Y")
        except ValueError:
            continue
        msg = m.group(2).strip()
        records.append((ts, msg))

    if not records:
        return info

    if recording_start is None:
        relevant = records[-100:]
    else:
        lower = recording_start - timedelta(hours=12)
        upper = recording_start + timedelta(hours=1)
        relevant = [(ts, msg) for ts, msg in records if lower <= ts <= upper]
        if not relevant:
            relevant = records[-100:]

    for _, msg in relevant:
        m = re.search(r"Probe=(\d+)", msg)
        if m:
            info["probe_id"] = m.group(1)
        m = re.search(r"Device S/N=(\d+)", msg)
        if m:
            info["device_serial"] = m.group(1)
        m = re.search(r"Version=([0-9.]+)", msg)
        if m:
            info["software_version"] = m.group(1)

    return info


def parse_sleep_header(header: bytes) -> Dict[str, object]:
    meta: Dict[str, object] = {}
    runs = _ascii_runs(header[:128], min_len=4)
    if len(runs) >= 5:
        meta["creation_date_text"] = runs[0]
        meta["creation_time_text"] = runs[1]
        meta["manufacturer"] = runs[2]
        meta["device_model"] = runs[3]
        meta["device_serial"] = runs[4]

    # Recording start is stored in config tags F0 22 (date) and F0 21 (time).
    p_date = header.find(b"\xf0\x22")
    p_time = header.find(b"\xf0\x21")
    if p_date >= 0 and p_date + 5 < len(header) and p_time >= 0 and p_time + 4 < len(header):
        y = 2000 + header[p_date + 2]
        mo = header[p_date + 3]
        day = header[p_date + 4]
        hh = header[p_time + 2]
        mm = header[p_time + 3]
        ss = header[p_time + 4]
        try:
            meta["recording_start"] = datetime(int(y), int(mo), int(day), int(hh), int(mm), int(ss))
        except ValueError:
            pass

    return meta


def _find_valid_markers(payload: bytes) -> Tuple[List[int], List[int]]:
    positions: List[int] = []
    counters_ms: List[int] = []
    expected = 1000
    search_from = 0
    while True:
        pos = payload.find(b"\xf0\x20", search_from)
        if pos < 0:
            break
        if (
            pos + 9 < len(payload)
            and payload[pos + 6] == 0xE0
            and payload[pos + 8] == 0x7F
            and payload[pos + 9] == 0x7F
        ):
            counter = int.from_bytes(payload[pos + 2 : pos + 6], "big")
            if counter == expected:
                positions.append(pos)
                counters_ms.append(counter)
                expected += 1000
        search_from = pos + 1
    if len(positions) < 2:
        raise ZzpDecodeError("Could not find enough valid F0 20 timing markers in Sleep.dat payload.")
    return positions, counters_ms


def _estimate_pre_marker_frames(prefix: bytes) -> int:
    # Only needed for sub-second start-offset estimation. We only need a rough estimate.
    n_tokens = 0
    for i in range(0, len(prefix) - 1, 2):
        if (prefix[i] >> 4) in VALID_FRAME_HIGHS:
            n_tokens += 1
    return n_tokens // 5


def infer_stream_layout(payload: bytes, marker_positions: Sequence[int], probe_seconds: int = 20) -> StreamLayout:
    probe_n = min(max(1, probe_seconds), len(marker_positions) - 1)
    counts_by_high: Dict[int, List[int]] = {h: [] for h in range(1, 6)}

    for sec in range(probe_n):
        seg = payload[marker_positions[sec] + 10 : marker_positions[sec + 1]]
        highs = [seg[i] >> 4 for i in range(0, len(seg) - 1, 2)]
        c = Counter(highs)
        for h in range(1, 6):
            counts_by_high[h].append(int(c.get(h, 0)))

    med = {h: int(round(float(np.median(v)))) if v else 0 for h, v in counts_by_high.items()}

    if med[1] >= 80 and med[2] >= 80 and med[3] >= 80 and med[4] >= 80 and med[5] >= 80:
        return StreamLayout(high_rate_order=(2, 3, 1, 4, 5), high_rate_hz=100, low_rate_hz={}, description="5x100Hz")

    if med[1] >= 80 and med[2] >= 80 and med[3] >= 80 and med[4] >= 80 and 1 <= med[5] < 80:
        return StreamLayout(
            high_rate_order=(2, 3, 1, 4),
            high_rate_hz=100,
            low_rate_hz={5: med[5]},
            description=f"4x100Hz + ch5@{med[5]}Hz",
        )

    if med[1] >= 80 and med[2] >= 80 and med[3] >= 80 and med[4] >= 80:
        return StreamLayout(high_rate_order=(2, 3, 1, 4), high_rate_hz=100, low_rate_hz={}, description="4x100Hz")

    raise ZzpDecodeError(f"Unsupported Sleep.dat frame layout; median counts per second were {med}.")


def _resample_small_vector(values: np.ndarray, target_n: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == target_n:
        return values.astype(np.int16)
    if values.size == 0:
        raise ZzpDecodeError("Cannot resample an empty auxiliary channel block.")
    x_old = np.linspace(0.0, 1.0, num=values.size, endpoint=False) + 0.5 / values.size
    x_new = np.linspace(0.0, 1.0, num=target_n, endpoint=False) + 0.5 / target_n
    return np.rint(np.interp(x_new, x_old, values)).astype(np.int16)


def _parse_full_second_segment(
    seg: bytes,
    layout: StreamLayout,
    frames_out: np.ndarray,
) -> Tuple[Dict[int, np.ndarray], int, int, int, int, int, int]:
    fi = 0
    slot = 0
    metric8 = -1
    metric9 = -1
    metricA = -1
    b0 = -1
    c0 = -1
    d0 = -1
    mv = memoryview(seg)
    i = 0
    n = len(mv)
    aux_lists: Dict[int, List[int]] = {h: [] for h in layout.low_rate_hz}

    while i + 1 < n:
        idb = mv[i]
        val = mv[i + 1]
        high = idb >> 4

        if fi < 100 and high == layout.high_rate_order[slot]:
            frames_out[fi, slot] = ((idb & 0x0F) << 8) | val
            slot += 1
            if slot == len(layout.high_rate_order):
                slot = 0
                fi += 1
            i += 2
            continue

        if high in aux_lists:
            aux_lists[high].append(((idb & 0x0F) << 8) | val)
        elif high == 8 and metric8 < 0:
            metric8 = ((idb & 0x0F) << 8) | val
        elif high == 9 and metric9 < 0:
            metric9 = ((idb & 0x0F) << 8) | val
        elif high == 10 and metricA < 0:
            metricA = ((idb & 0x0F) << 8) | val
        elif idb == 0xB0 and b0 < 0:
            b0 = val
        elif idb == 0xC0 and c0 < 0:
            c0 = val
        elif idb == 0xD0 and d0 < 0:
            d0 = val
        i += 2

    if fi != 100:
        raw_counts = Counter((mv[j] >> 4) for j in range(0, n - 1, 2))
        raise ZzpDecodeError(
            f"Expected 100 complete {'/'.join(hex(h) for h in layout.high_rate_order)} frames in a marker interval, "
            f"found {fi}; high-nibble counts were {dict(sorted(raw_counts.items()))}."
        )
    if min(metric8, metric9, metricA, b0, c0, d0) < 0:
        raise ZzpDecodeError("Missing one or more expected 1 Hz side metrics in a marker interval.")

    aux_out: Dict[int, np.ndarray] = {}
    for high, target_hz in layout.low_rate_hz.items():
        aux_out[high] = _resample_small_vector(np.asarray(aux_lists[high], dtype=np.int16), target_hz)

    return aux_out, metric8, metric9, metricA, b0, c0, d0


def decode_sleep_dat(sleep_dat: bytes, metadata: StudyMetadata, verbose: bool = False) -> Dict[str, object]:
    if len(sleep_dat) <= HEADER_BYTES:
        raise ZzpDecodeError("Sleep.dat is too small.")

    header = sleep_dat[:HEADER_BYTES]
    payload = sleep_dat[HEADER_BYTES:]

    header_meta = parse_sleep_header(header)
    for key, value in header_meta.items():
        if hasattr(metadata, key):
            setattr(metadata, key, value)  # type: ignore[arg-type]

    marker_positions, counters_ms = _find_valid_markers(payload)
    layout = infer_stream_layout(payload, marker_positions)
    n_full_seconds = len(marker_positions) - 1
    pre_frames = _estimate_pre_marker_frames(payload[: marker_positions[0]])
    metadata.export_offset_est_s = pre_frames / 100.0
    if metadata.recording_start is not None:
        metadata.export_start = metadata.recording_start + timedelta(seconds=int(round(metadata.export_offset_est_s)))

    frames = np.empty((n_full_seconds, 100, len(layout.high_rate_order)), dtype=np.int16)
    low_rate_channels: Dict[int, np.ndarray] = {
        high: np.empty((n_full_seconds, hz), dtype=np.int16) for high, hz in layout.low_rate_hz.items()
    }
    metric8 = np.empty(n_full_seconds, dtype=np.int16)
    metric9 = np.empty(n_full_seconds, dtype=np.int16)
    metricA = np.empty(n_full_seconds, dtype=np.int16)
    b0 = np.empty(n_full_seconds, dtype=np.int16)
    c0 = np.empty(n_full_seconds, dtype=np.int16)
    d0 = np.empty(n_full_seconds, dtype=np.int16)
    scratch = np.empty((100, len(layout.high_rate_order)), dtype=np.int16)

    for sec in range(n_full_seconds):
        seg_start = marker_positions[sec] + 10
        seg_end = marker_positions[sec + 1]
        aux_out, m8, m9, mA, bb, cc, dd = _parse_full_second_segment(payload[seg_start:seg_end], layout, scratch)
        frames[sec] = scratch
        for high, arr in aux_out.items():
            low_rate_channels[high][sec] = arr
        metric8[sec] = m8
        metric9[sec] = m9
        metricA[sec] = mA
        b0[sec] = bb
        c0[sec] = cc
        d0[sec] = dd
        if verbose and sec and sec % 5000 == 0:
            print(f"decoded {sec}/{n_full_seconds} full seconds", file=sys.stderr)

    return {
        "frames": frames,
        "layout": {
            "description": layout.description,
            "high_rate_order": list(layout.high_rate_order),
            "high_rate_hz": layout.high_rate_hz,
            "low_rate_hz": dict(layout.low_rate_hz),
        },
        "low_rate_channels": low_rate_channels,
        "counter_ms": np.asarray(counters_ms[:-1], dtype=np.int32),
        "metric8": metric8,
        "metric9": metric9,
        "metricA": metricA,
        "b0": b0,
        "c0": c0,
        "d0": d0,
        "pre_frames": pre_frames,
    }


def infer_channel_mapping(
    frames: np.ndarray,
    spo2: np.ndarray,
    low_rate_channels: Optional[Dict[int, np.ndarray]] = None,
) -> ChannelMapping:
    if frames.ndim != 3 or frames.shape[1] != 100:
        raise ValueError("frames must have shape (seconds, 100, channels)")

    n_seconds = frames.shape[0]
    n_channels = frames.shape[2]
    warmup = min(600, max(120, n_seconds // 20))
    stable_len = min(3600, max(300, n_seconds - warmup))
    stable_start = min(warmup, max(0, n_seconds - stable_len))
    stable_end = min(n_seconds, stable_start + stable_len)

    stable_flat = frames[stable_start:stable_end].reshape(-1, n_channels).astype(np.float64)
    corr = np.corrcoef(stable_flat.T)
    best_pair = (0, 1)
    best_corr = -np.inf
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            if corr[i, j] > best_corr:
                best_corr = float(corr[i, j])
                best_pair = (i, j)

    sec_means = frames.astype(np.float64).mean(axis=1)
    sec_sds = frames.astype(np.float64).std(axis=1)
    a, b = best_pair
    post = slice(warmup, n_seconds)
    red = a
    ir = b

    if np.unique(spo2[post]).size > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            r_ab = (sec_sds[post, a] / np.maximum(sec_means[post, a], 1e-9)) / (
                sec_sds[post, b] / np.maximum(sec_means[post, b], 1e-9)
            )
            r_ba = (sec_sds[post, b] / np.maximum(sec_means[post, b], 1e-9)) / (
                sec_sds[post, a] / np.maximum(sec_means[post, a], 1e-9)
            )
        mask_ab = np.isfinite(r_ab)
        mask_ba = np.isfinite(r_ba)
        corr_ab = float(np.corrcoef(r_ab[mask_ab], spo2[post][mask_ab])[0, 1]) if mask_ab.sum() > 10 else np.nan
        corr_ba = float(np.corrcoef(r_ba[mask_ba], spo2[post][mask_ba])[0, 1]) if mask_ba.sum() > 10 else np.nan
        if np.isfinite(corr_ab) and np.isfinite(corr_ba):
            if corr_ba < corr_ab:
                red, ir = b, a
        elif sec_means[post, b].mean() > sec_means[post, a].mean():
            red, ir = b, a
    else:
        corr_ab = float("nan")
        corr_ba = float("nan")
        if sec_means[post, b].mean() > sec_means[post, a].mean():
            red, ir = b, a

    remaining = [c for c in range(n_channels) if c not in (red, ir)]
    stable = slice(stable_start, stable_end)
    freqs = np.fft.rfftfreq(100, d=1.0 / 100.0)
    hf_band = freqs >= 5.0
    lf_band = (freqs >= 0.5) & (freqs < 5.0)

    hf_scores: Dict[int, float] = {}
    for c in remaining:
        ratios: List[float] = []
        for second in frames[stable, :, c].astype(np.float64)[::10]:
            x = second - second.mean()
            spec = np.abs(np.fft.rfft(x)) ** 2
            ratios.append(float(spec[hf_band].sum() / (spec[lf_band].sum() + 1e-9)))
        hf_scores[c] = float(np.median(ratios)) if ratios else 0.0

    probe_pressure = None
    probe_pressure_aux_high = None

    if low_rate_channels and 5 in low_rate_channels and len(remaining) >= 2:
        actigraphy = max(remaining, key=hf_scores.get)
        pat = [c for c in remaining if c != actigraphy][0]
        probe_pressure_aux_high = 5
        pressure_score = float(
            (np.max(low_rate_channels[5][: min(120, n_seconds)]) + 1.0)
            / max(float(np.median(low_rate_channels[5][min(600, n_seconds // 2) :])), 1.0)
        )
    else:
        if len(remaining) < 3:
            raise ZzpDecodeError("Need at least three non-PPG channels to infer PAT/actigraphy/probe pressure.")
        start = slice(0, min(120, n_seconds))
        pressure_scores: Dict[int, float] = {}
        for c in remaining:
            start_max = float(frames[start, :, c].max())
            stable_median = float(np.median(frames[stable, :, c]))
            pressure_scores[c] = (start_max + 1.0) / max(stable_median, 1.0)
        probe_pressure = max(pressure_scores, key=pressure_scores.get)
        pressure_score = float(pressure_scores[probe_pressure])
        remaining2 = [c for c in remaining if c != probe_pressure]
        actigraphy = max(remaining2, key=hf_scores.get)
        pat = [c for c in remaining2 if c != actigraphy][0]

    confidence = {
        "ppg_pair": "high" if best_corr > 0.9 else "moderate",
        "red_ir_orientation": (
            "high" if np.unique(spo2[post]).size > 1 and np.isfinite(corr_ab) and np.isfinite(corr_ba) else "low"
        ),
        "pat": "moderate-high",
        "actigraphy": "high",
        "probe_pressure": "moderate-high",
    }
    diagnostics = {
        "ppg_pair_correlation": best_corr,
        "ratio_vs_spo2_corr_ab": float(corr_ab),
        "ratio_vs_spo2_corr_ba": float(corr_ba),
        "pressure_score": pressure_score,
        "actigraphy_hf_score": float(hf_scores[actigraphy]),
    }

    return ChannelMapping(
        ppg_red=red,
        ppg_ir=ir,
        pat=pat,
        actigraphy=actigraphy,
        probe_pressure=probe_pressure,
        probe_pressure_aux_high=probe_pressure_aux_high,
        confidence=confidence,
        diagnostics=diagnostics,
    )


def _bandpass_for_pulse(x: np.ndarray, fs: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if butter is not None and sosfiltfilt is not None:
        sos = butter(2, [0.5, 3.0], btype="bandpass", fs=fs, output="sos")
        return sosfiltfilt(sos, x)
    # Fallback: high-pass-ish first difference + short smoothing.
    dx = np.diff(x, prepend=x[0])
    kernel = np.ones(5, dtype=np.float64) / 5.0
    return np.convolve(dx, kernel, mode="same")


def derive_pulse_rate(
    ppg_signal: np.ndarray, fs: int = 100, window_sec: int = 10, min_bpm: int = 35, max_bpm: int = 140
) -> np.ndarray:
    x = _bandpass_for_pulse(np.asarray(ppg_signal, dtype=np.float64), fs)
    n_seconds = len(x) // fs
    window_len = window_sec * fs
    half = window_len // 2
    nfft = 4096 if window_len <= 4096 else 1 << math.ceil(math.log2(window_len))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    band = (freqs >= min_bpm / 60.0) & (freqs <= max_bpm / 60.0)
    band_freqs = freqs[band]
    taper = np.hanning(window_len)
    padded = np.pad(x, (half, half), mode="edge")
    out = np.empty(n_seconds, dtype=np.float64)

    for sec in range(n_seconds):
        center = sec * fs + fs // 2
        win = padded[center : center + window_len] * taper
        spec = np.abs(np.fft.rfft(win, n=nfft)) ** 2
        out[sec] = float(band_freqs[np.argmax(spec[band])]) * 60.0

    if medfilt is not None:
        out = medfilt(out, kernel_size=5)
    out = np.clip(np.nan_to_num(out, nan=0.0), min_bpm, max_bpm)
    return np.rint(out).astype(np.int16)


def build_signals(
    decoded: Dict[str, object],
    mapping: ChannelMapping,
    include_internal_1hz: bool,
    include_pulse_rate: bool,
) -> List[SignalSpec]:
    frames = np.asarray(decoded["frames"], dtype=np.int16)
    flat = frames.reshape(-1, frames.shape[2])
    layout = decoded.get("layout", {})
    low_rate_channels = decoded.get("low_rate_channels", {})

    signals: List[SignalSpec] = [
        SignalSpec(
            "PPG_RED",
            flat[:, mapping.ppg_red].copy(),
            100,
            "count",
            transducer="WatchPAT oximeter red",
            prefilter="raw 12-bit",
        ),
        SignalSpec(
            "PPG_IR",
            flat[:, mapping.ppg_ir].copy(),
            100,
            "count",
            transducer="WatchPAT oximeter IR",
            prefilter="raw 12-bit",
        ),
        SignalSpec(
            "PAT", flat[:, mapping.pat].copy(), 100, "count", transducer="WatchPAT PAT probe", prefilter="raw 12-bit"
        ),
    ]

    if mapping.actigraphy is not None:
        signals.append(
            SignalSpec(
                "Actigraphy",
                flat[:, mapping.actigraphy].copy(),
                100,
                "count",
                transducer="WatchPAT wrist actigraph",
                prefilter="raw 12-bit",
            )
        )

    if mapping.probe_pressure is not None:
        signals.append(
            SignalSpec(
                "ProbePress",
                flat[:, mapping.probe_pressure].copy(),
                100,
                "count",
                transducer="WatchPAT probe pressure/servo",
                prefilter="raw 12-bit",
            )
        )
    elif mapping.probe_pressure_aux_high is not None and mapping.probe_pressure_aux_high in low_rate_channels:
        aux = np.asarray(low_rate_channels[mapping.probe_pressure_aux_high], dtype=np.int16)
        aux_hz = int(layout.get("low_rate_hz", {}).get(mapping.probe_pressure_aux_high, aux.shape[1]))
        signals.append(
            SignalSpec(
                "ProbePress",
                aux.reshape(-1).copy(),
                aux_hz,
                "count",
                transducer="WatchPAT probe pressure/servo",
                prefilter=f"raw 12-bit high-nibble {mapping.probe_pressure_aux_high} at {aux_hz} Hz",
            )
        )

    signals.append(
        SignalSpec(
            "SpO2",
            np.asarray(decoded["b0"], dtype=np.int16).copy(),
            1,
            "%",
            transducer="WatchPAT oximeter summary",
            prefilter="1 Hz side packet B0",
        )
    )

    if include_pulse_rate:
        pr = derive_pulse_rate(flat[:, mapping.ppg_ir], fs=100)
        signals.append(
            SignalSpec("PulseRate", pr, 1, "bpm", transducer="derived from PPG_IR", prefilter="10 s FFT + bandpass"),
        )

    if include_internal_1hz:
        signals.extend(
            [
                SignalSpec(
                    "WP8_RAW",
                    np.asarray(decoded["metric8"], dtype=np.int16).copy(),
                    1,
                    "count",
                    transducer="WatchPAT internal 1 Hz metric",
                    prefilter="side packet 0x8*",
                ),
                SignalSpec(
                    "WP9_RAW",
                    np.asarray(decoded["metric9"], dtype=np.int16).copy(),
                    1,
                    "count",
                    transducer="WatchPAT internal 1 Hz metric",
                    prefilter="side packet 0x9*",
                ),
                SignalSpec(
                    "WPA_RAW",
                    np.asarray(decoded["metricA"], dtype=np.int16).copy(),
                    1,
                    "count",
                    transducer="WatchPAT internal 1 Hz metric",
                    prefilter="side packet 0xA*",
                ),
                SignalSpec(
                    "WPC0_RAW",
                    np.asarray(decoded["c0"], dtype=np.int16).copy(),
                    1,
                    "count",
                    transducer="WatchPAT internal 1 Hz metric",
                    prefilter="side packet C0",
                ),
                SignalSpec(
                    "WPD0_RAW",
                    np.asarray(decoded["d0"], dtype=np.int16).copy(),
                    1,
                    "count",
                    transducer="WatchPAT internal 1 Hz metric",
                    prefilter="side packet D0",
                ),
            ]
        )

    return signals


def _edf_ascii(text: object, width: int) -> bytes:
    s = str(text) if text is not None else ""
    s = re.sub(r"[\r\n\t]+", " ", s)
    return s.encode("ascii", errors="replace")[:width].ljust(width, b" ")


def _identity_header_dict(label: str, sf: int, dimension: str, transducer: str, prefilter: str) -> Dict[str, object]:
    return {
        "label": label[:16],
        "dimension": dimension[:8],
        "sample_frequency": int(sf),
        "physical_min": -32768.0,
        "physical_max": 32767.0,
        "digital_min": -32768,
        "digital_max": 32767,
        "transducer": transducer[:80],
        "prefilter": prefilter[:80],
    }


def write_edf_manual(path: str, signals: Sequence[SignalSpec], metadata: StudyMetadata) -> None:
    if not signals:
        raise EdfWriteError("No signals to write.")

    durations = [len(sig.samples) / sig.sample_frequency for sig in signals]
    if max(durations) - min(durations) > 1e-9:
        raise EdfWriteError(f"Signals do not share a common duration: {durations}")
    n_records = int(round(durations[0]))
    if n_records <= 0:
        raise EdfWriteError("Empty EDF output.")

    n_signals = len(signals)
    start_dt = metadata.export_start or metadata.recording_start or datetime.utcnow().replace(microsecond=0)
    patient_id = f"WatchPAT {metadata.patient_code}".strip()
    rec_note = (
        f"{metadata.device_model or 'WatchPAT'} SN={metadata.device_serial or '?'} "
        f"src={os.path.basename(metadata.source_path)}"
    )

    header_bytes = 256 + 256 * n_signals
    with open(path, "wb") as f:
        f.write(_edf_ascii("0", 8))
        f.write(_edf_ascii(patient_id, 80))
        f.write(_edf_ascii(rec_note, 80))
        f.write(_edf_ascii(start_dt.strftime("%d.%m.%y"), 8))
        f.write(_edf_ascii(start_dt.strftime("%H.%M.%S"), 8))
        f.write(_edf_ascii(str(header_bytes), 8))
        f.write(_edf_ascii("", 44))
        f.write(_edf_ascii(str(n_records), 8))
        f.write(_edf_ascii("1", 8))
        f.write(_edf_ascii(str(n_signals), 4))

        for field, width in [
            ("label", 16),
            ("transducer", 80),
            ("dimension", 8),
            ("physical_min", 8),
            ("physical_max", 8),
            ("digital_min", 8),
            ("digital_max", 8),
            ("prefilter", 80),
            ("sample_frequency", 8),
            ("reserved", 32),
        ]:
            for sig in signals:
                header = _identity_header_dict(
                    sig.label, sig.sample_frequency, sig.dimension, sig.transducer, sig.prefilter
                )
                value = header.get(field, "")
                f.write(_edf_ascii(value, width))

        for rec in range(n_records):
            for sig in signals:
                start = rec * sig.sample_frequency
                stop = start + sig.sample_frequency
                block = np.asarray(sig.samples[start:stop], dtype="<i2")
                if block.size != sig.sample_frequency:
                    raise EdfWriteError(f"Channel {sig.label} is shorter than expected at record {rec}.")
                f.write(block.tobytes(order="C"))


def write_edf_pyedflib(path: str, signals: Sequence[SignalSpec], metadata: StudyMetadata) -> None:
    if pyedflib is None:
        raise EdfWriteError("pyedflib is not installed")
    if not signals:
        raise EdfWriteError("No signals to write.")

    file_type = getattr(pyedflib, "FILETYPE_EDFPLUS", 1)
    writer = pyedflib.EdfWriter(path, len(signals), file_type=file_type)
    try:
        writer.setSignalHeaders(
            [
                _identity_header_dict(sig.label, sig.sample_frequency, sig.dimension, sig.transducer, sig.prefilter)
                for sig in signals
            ]
        )
        start_dt = metadata.export_start or metadata.recording_start
        if start_dt is not None:
            writer.setStartdatetime(start_dt)
        if metadata.patient_code:
            writer.setPatientCode(metadata.patient_code[:80])
        if metadata.device_model or metadata.device_serial:
            writer.setEquipment(f"{metadata.device_model} SN={metadata.device_serial}"[:80])
        rec_add = f"src={os.path.basename(metadata.source_path)}; offset={metadata.export_offset_est_s:.2f}s"
        writer.setRecordingAdditional(rec_add[:80])
        if metadata.probe_id:
            writer.setTechnician(f"Probe={metadata.probe_id}"[:80])
        writer.writeSamples([np.asarray(sig.samples, dtype=np.int16) for sig in signals], digital=True)
    finally:
        writer.close()


def write_edf(path: str, signals: Sequence[SignalSpec], metadata: StudyMetadata, backend: str) -> str:
    last_error: Optional[Exception] = None
    if backend in ("auto", "pyedflib") and pyedflib is not None:
        try:
            write_edf_pyedflib(path, signals, metadata)
            return "pyedflib"
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc
            if backend == "pyedflib":
                raise
            warnings.warn(f"pyedflib write failed; falling back to manual EDF writer: {exc}")

    try:
        write_edf_manual(path, signals, metadata)
        return "manual"
    except Exception as exc:
        if last_error is not None:
            raise EdfWriteError(
                f"pyedflib failed with {last_error!r}; manual EDF writer then failed with {exc!r}"
            ) from exc
        raise


def build_summary(
    metadata: StudyMetadata,
    decoded: Dict[str, object],
    mapping: ChannelMapping,
    signals: Sequence[SignalSpec],
    backend: str,
) -> Dict[str, object]:
    return {
        "source_path": metadata.source_path,
        "patient_code": metadata.patient_code,
        "recording_start": metadata.recording_start.isoformat() if metadata.recording_start else None,
        "export_start": metadata.export_start.isoformat() if metadata.export_start else None,
        "export_offset_est_s": metadata.export_offset_est_s,
        "manufacturer": metadata.manufacturer,
        "device_model": metadata.device_model,
        "device_serial": metadata.device_serial,
        "probe_id": metadata.probe_id,
        "software_version": metadata.software_version,
        "decoded_full_seconds": int(np.asarray(decoded["frames"]).shape[0]),
        "decoded_100hz_samples": int(np.asarray(decoded["frames"]).shape[0] * 100),
        "marker_counter_start_ms": int(np.asarray(decoded["counter_ms"])[0]),
        "marker_counter_end_ms": int(np.asarray(decoded["counter_ms"])[-1]),
        "layout": decoded.get("layout"),
        "channel_mapping": {
            "PPG_RED": mapping.ppg_red,
            "PPG_IR": mapping.ppg_ir,
            "PAT": mapping.pat,
            "Actigraphy": mapping.actigraphy,
            "ProbePress_high_rate_index": mapping.probe_pressure,
            "ProbePress_aux_high_nibble": mapping.probe_pressure_aux_high,
        },
        "confidence": mapping.confidence,
        "diagnostics": mapping.diagnostics,
        "edf_backend": backend,
        "edf_signals": [
            {
                "label": sig.label,
                "sample_frequency": sig.sample_frequency,
                "dimension": sig.dimension,
                "transducer": sig.transducer,
                "prefilter": sig.prefilter,
                "n_samples": int(len(sig.samples)),
            }
            for sig in signals
        ],
    }


def convert_zzp_to_edf(
    input_path: str,
    output_path: str,
    backend: str = "auto",
    include_internal_1hz: bool = False,
    include_pulse_rate: bool = True,
    json_summary_path: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, object]:
    metadata = StudyMetadata(source_path=input_path)

    with zipfile.ZipFile(input_path, "r") as zf:
        members = {name.lower(): name for name in zf.namelist()}
        try:
            sleep_name = members["sleep.dat"]
            patient_name = members["patient.dat"]
            log_name = members["log.dat"]
        except KeyError as exc:
            raise ZzpDecodeError(f"Archive is missing required member {exc.args[0]!r}") from exc

        sleep_dat = zf.read(sleep_name)
        patient_dat = zf.read(patient_name)
        log_dat = zf.read(log_name)

    metadata.patient_code = parse_patient_dat(patient_dat)
    decoded = decode_sleep_dat(sleep_dat, metadata, verbose=verbose)
    log_info = parse_log_dat(log_dat, metadata.recording_start)
    metadata.device_serial = log_info.get("device_serial", metadata.device_serial)
    metadata.probe_id = log_info.get("probe_id", metadata.probe_id)
    metadata.software_version = log_info.get("software_version", metadata.software_version)

    mapping = infer_channel_mapping(
        np.asarray(decoded["frames"], dtype=np.int16),
        np.asarray(decoded["b0"], dtype=np.int16),
        low_rate_channels=decoded.get("low_rate_channels"),
    )
    signals = build_signals(
        decoded, mapping, include_internal_1hz=include_internal_1hz, include_pulse_rate=include_pulse_rate
    )
    used_backend = write_edf(output_path, signals, metadata, backend=backend)

    summary = build_summary(metadata, decoded, mapping, signals, used_backend)
    if json_summary_path:
        with open(json_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    return summary


def _discover_input_files(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_dir():
        pattern = "**/*.zzp" if recursive else "*.zzp"
        files = sorted(path for path in input_path.glob(pattern) if path.is_file())
        if not files:
            raise FileNotFoundError(f"No .zzp files found under {input_path}")
        return files
    return [input_path]


def _ensure_parent_dir(path: Optional[Path]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_single_output_path(input_path: Path, output_arg: Optional[str]) -> Path:
    if output_arg is None:
        return input_path.with_suffix(".edf")

    output_path = Path(output_arg).expanduser()
    if output_path.exists() and output_path.is_dir():
        return output_path / input_path.with_suffix(".edf").name
    return output_path


def _resolve_batch_output_path(input_file: Path, input_root: Path, output_root: Path, suffix: str) -> Path:
    relative = input_file.relative_to(input_root)
    return (output_root / relative).with_suffix(suffix)


def _run_batch_conversion(args: argparse.Namespace, input_root: Path, files: Sequence[Path]) -> int:
    output_root = Path(args.output_edf).expanduser() if args.output_edf else input_root
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("When input_zzp is a directory, output_edf must be a directory path.")

    summary_root = Path(args.json_summary).expanduser() if args.json_summary else None
    if summary_root is not None and summary_root.exists() and not summary_root.is_dir():
        raise ValueError("When input_zzp is a directory, --json-summary must be a directory path.")

    converted = 0
    skipped = 0
    failures: List[Dict[str, str]] = []
    processed = 0
    started_at = time.time()
    write_progress(
        output_root,
        status="running",
        task="watchpat_zzp_to_edf",
        processed=0,
        total=len(files),
        success=0,
        failed=0,
        start_time=started_at,
    )
    iterator = files
    if tqdm is not None:
        iterator = tqdm(files, desc="Converting .zzp files", unit="file")

    for input_file in iterator:
        output_path = _resolve_batch_output_path(input_file, input_root, output_root, ".edf")
        summary_path = (
            _resolve_batch_output_path(input_file, input_root, summary_root, ".json")
            if summary_root is not None
            else None
        )

        if args.skip_existing and output_path.exists():
            skipped += 1
            processed += 1
            write_progress(
                output_root,
                status="running",
                task="watchpat_zzp_to_edf",
                processed=processed,
                total=len(files),
                success=converted,
                failed=len(failures),
                start_time=started_at,
                current_item=str(input_file),
                message=f"skipped={skipped}",
            )
            continue

        _ensure_parent_dir(output_path)
        _ensure_parent_dir(summary_path)

        try:
            convert_zzp_to_edf(
                input_path=str(input_file.resolve()),
                output_path=str(output_path.resolve()),
                backend=args.writer,
                include_internal_1hz=args.include_internal_1hz,
                include_pulse_rate=not args.no_pulse_rate,
                json_summary_path=str(summary_path.resolve()) if summary_path is not None else None,
                verbose=args.verbose,
            )
            converted += 1
        except Exception as exc:
            failures.append(
                {
                    "input_zzp": str(input_file.resolve()),
                    "output_edf": str(output_path.resolve()),
                    "error": str(exc),
                }
            )
        processed += 1
        write_progress(
            output_root,
            status="running",
            task="watchpat_zzp_to_edf",
            processed=processed,
            total=len(files),
            success=converted,
            failed=len(failures),
            start_time=started_at,
            current_item=str(input_file),
            message=f"converted={converted} skipped={skipped}",
        )

    print(
        json.dumps(
            {
                "batch_mode": True,
                "input_root": str(input_root.resolve()),
                "output_root": str(output_root.resolve()),
                "json_summary_root": str(summary_root.resolve()) if summary_root is not None else None,
                "total_files": len(files),
                "converted": converted,
                "skipped_existing": skipped,
                "failed": len(failures),
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
    )
    write_progress(
        output_root,
        status="completed" if not failures else "failed",
        task="watchpat_zzp_to_edf",
        processed=len(files),
        total=len(files),
        success=converted,
        failed=len(failures),
        start_time=started_at,
        message=f"converted={converted} skipped={skipped} failed={len(failures)}",
    )
    return 1 if failures else 0


def _make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert a WatchPAT .zzp archive into an EDF file.")
    p.add_argument("input_zzp", help="Input .zzp archive or a directory of .zzp archives")
    p.add_argument("output_edf", nargs="?", help="Output EDF path, or output directory when input_zzp is a directory")
    p.add_argument("--writer", choices=["auto", "pyedflib", "manual"], default="auto", help="EDF writer backend")
    p.add_argument("--no-pulse-rate", action="store_true", help="Skip 1 Hz PulseRate derivation from PPG_IR")
    p.add_argument(
        "--include-internal-1hz",
        action="store_true",
        help="Also export raw internal 1 Hz side-packet metrics (WP8/WP9/WPA/WPC0/WPD0)",
    )
    p.add_argument("--json-summary", help="Optional JSON summary path, or summary directory in batch mode")
    p.add_argument(
        "--recursive", action="store_true", help="Recursively discover .zzp files when input_zzp is a directory"
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip files whose output EDF already exists")
    p.add_argument("--verbose", action="store_true", help="Print progress to stderr")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _make_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input_zzp).expanduser()
    files = _discover_input_files(input_path, recursive=args.recursive)

    if input_path.is_dir():
        try:
            return _run_batch_conversion(args, input_path, files)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    output_path = _resolve_single_output_path(input_path, args.output_edf)
    summary_path = Path(args.json_summary).expanduser() if args.json_summary else None
    _ensure_parent_dir(output_path)
    _ensure_parent_dir(summary_path)

    try:
        summary = convert_zzp_to_edf(
            input_path=str(input_path.resolve()),
            output_path=str(output_path.resolve()),
            backend=args.writer,
            include_internal_1hz=args.include_internal_1hz,
            include_pulse_rate=not args.no_pulse_rate,
            json_summary_path=str(summary_path.resolve()) if summary_path is not None else None,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
