from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import typing as t

import numpy as np


@dataclass(frozen=True)
class KaldiWaveChannelSpec:
    name: str
    frames_per_epoch: int
    scp_path: Path


class KaldiWaveReaderPool:
    def __init__(self, root: str | Path, channel_specs: t.Mapping[str, KaldiWaveChannelSpec]) -> None:
        self.root = Path(root).expanduser()
        self.channel_specs = self._normalize_specs(channel_specs)
        self._pid = os.getpid()
        self._readers: dict[str, t.Any] = {}
        self._kaldi_native_io = None

    def _normalize_specs(
        self,
        channel_specs: t.Mapping[str, KaldiWaveChannelSpec],
    ) -> dict[str, KaldiWaveChannelSpec]:
        specs = {}
        for channel, spec in channel_specs.items():
            scp_path = Path(spec.scp_path).expanduser()
            if not scp_path.is_absolute():
                scp_path = self.root / scp_path
            specs[str(channel)] = KaldiWaveChannelSpec(
                name=str(spec.name),
                frames_per_epoch=int(spec.frames_per_epoch),
                scp_path=scp_path,
            )
        return specs

    def __getstate__(self) -> dict[str, t.Any]:
        state = dict(self.__dict__)
        state["_readers"] = {}
        state["_kaldi_native_io"] = None
        return state

    def __setstate__(self, state: dict[str, t.Any]) -> None:
        self.__dict__.update(state)
        self._pid = os.getpid()
        self._readers = {}
        self._kaldi_native_io = None

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers = {}

    def _import_kaldi_native_io(self):
        if self._kaldi_native_io is None:
            import kaldi_native_io

            self._kaldi_native_io = kaldi_native_io
        return self._kaldi_native_io

    def _ensure_current_process(self) -> None:
        pid = os.getpid()
        if pid != self._pid:
            self.close()
            self._pid = pid

    def _reader_for_channel(self, channel: str):
        self._ensure_current_process()
        if channel not in self._readers:
            spec = self.channel_specs[channel]
            kaldi_native_io = self._import_kaldi_native_io()
            self._readers[channel] = kaldi_native_io.RandomAccessFloatMatrixReader(f"scp:{spec.scp_path}")
        return self._readers[channel]

    def read_matrix(self, channel: str, key: str) -> np.ndarray:
        channel = str(channel)
        key = str(key)
        if channel not in self.channel_specs:
            available = sorted(self.channel_specs)
            raise KeyError(f"Unknown Kaldi channel {channel!r}. Available channels: {available}.")

        spec = self.channel_specs[channel]
        reader = self._reader_for_channel(channel)
        if key not in reader:
            raise KeyError(f"Missing Kaldi key {key!r} for channel {channel!r} in {spec.scp_path}.")

        arr = np.asarray(reader[key], dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"Kaldi matrix for channel {channel!r}, key {key!r} must be rank 2, got shape {arr.shape}."
            )
        if arr.shape[1] != spec.frames_per_epoch:
            raise ValueError(
                f"Kaldi matrix for channel {channel!r}, key {key!r} has frames_per_epoch={arr.shape[1]}, "
                f"expected {spec.frames_per_epoch}."
            )
        return arr.copy()


__all__ = ["KaldiWaveChannelSpec", "KaldiWaveReaderPool"]
