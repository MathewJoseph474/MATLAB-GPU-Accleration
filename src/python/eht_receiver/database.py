"""Lazy, layout-safe access to the MATLAB v7.3 packet database."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PacketBatch:
    """A CUDA-ready packet batch in contiguous packet-major layout."""

    source_indices: NDArray[np.int64]
    rx_i: NDArray[np.int16]
    rx_q: NDArray[np.int16]
    rx_scale: NDArray[np.float32]

    @property
    def packet_count(self) -> int:
        return int(self.source_indices.size)


class MatlabPacketDatabase:
    """Read packet arrays without loading the full database into RAM.

    MATLAB stores ``[samples, packets]`` columns contiguously.  A v7.3 MAT
    file exposes those dimensions reversed through h5py, so the normal case
    already appears as ``[packets, samples]``.  This class detects either
    representation and always returns C-contiguous packet-major arrays; their
    flat memory order matches the CUDA receiver's ``batch * sample_count``
    indexing.
    """

    def __init__(self, path: str | Path):
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "Reading the MATLAB v7.3 database requires h5py; "
                "install the Python project dependencies first."
            ) from exc

        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._file = h5py.File(self.path, "r")
        self._require_datasets("RxI", "RxQ", "RxScale", "TxBits")

        self._scales = np.asarray(self._file["RxScale"], dtype=np.float32).reshape(-1)
        self.unique_packets = int(self._scales.size)
        if self.unique_packets < 1:
            raise ValueError("RxScale contains no packets")

        self.sample_count = self._logical_width("RxI")
        self.payload_bits = self._logical_width("TxBits")
        if self._logical_width("RxQ") != self.sample_count:
            raise ValueError("RxI and RxQ dimensions do not match")

    def _require_datasets(self, *names: str) -> None:
        missing = [name for name in names if name not in self._file]
        if missing:
            raise ValueError(f"Missing database arrays: {', '.join(missing)}")

    def _packet_axis(self, name: str) -> int:
        shape = self._file[name].shape
        if len(shape) != 2:
            raise ValueError(f"{name} must be a two-dimensional array")
        matches = [axis for axis, size in enumerate(shape) if size == self.unique_packets]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot identify packet dimension for {name} with shape {shape}"
            )
        return matches[0]

    def _logical_width(self, name: str) -> int:
        axis = self._packet_axis(name)
        return int(self._file[name].shape[1 - axis])

    def _read_packet_rows(
        self, name: str, indices: NDArray[np.int64], dtype: np.dtype
    ) -> np.ndarray:
        dataset = self._file[name]
        axis = self._packet_axis(name)

        # h5py requires increasing fancy indices. Read each unique packet once,
        # then restore repetitions/order used by a logical benchmark workload.
        unique, inverse = np.unique(indices, return_inverse=True)
        if axis == 0:
            block = np.asarray(dataset[unique, :], dtype=dtype)
        else:
            block = np.asarray(dataset[:, unique], dtype=dtype).T
        return np.ascontiguousarray(block[inverse])

    def source_indices(self, num_packets: int) -> NDArray[np.int64]:
        if num_packets <= 0:
            raise ValueError("num_packets must be positive")
        return np.arange(num_packets, dtype=np.int64) % self.unique_packets

    def read_batch(self, indices: Sequence[int] | NDArray[np.integer]) -> PacketBatch:
        source = np.asarray(indices, dtype=np.int64).reshape(-1)
        if source.size == 0:
            raise ValueError("A batch must contain at least one packet")
        if np.any(source < 0) or np.any(source >= self.unique_packets):
            raise IndexError("Packet source index is outside the database")
        return PacketBatch(
            source_indices=source,
            rx_i=self._read_packet_rows("RxI", source, np.dtype(np.int16)),
            rx_q=self._read_packet_rows("RxQ", source, np.dtype(np.int16)),
            rx_scale=np.ascontiguousarray(self._scales[source]),
        )

    def read_bits(self, indices: Sequence[int] | NDArray[np.integer]) -> NDArray[np.int8]:
        source = np.asarray(indices, dtype=np.int64).reshape(-1)
        return self._read_packet_rows("TxBits", source, np.dtype(np.int8))

    def batches(self, num_packets: int, batch_size: int) -> Iterator[PacketBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        source = self.source_indices(num_packets)
        for first in range(0, num_packets, batch_size):
            yield self.read_batch(source[first : first + batch_size])

    def expected_checksums(self, indices: Sequence[int]) -> NDArray[np.uint64]:
        bits = self.read_bits(indices).astype(np.uint64, copy=False)
        weights = np.arange(1, self.payload_bits + 1, dtype=np.uint64)
        return bits @ weights

    def close(self) -> None:
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "MatlabPacketDatabase":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
