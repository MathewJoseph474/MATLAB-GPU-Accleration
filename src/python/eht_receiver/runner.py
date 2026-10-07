"""Backend-neutral asynchronous benchmark loop."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Protocol

import numpy as np

from .database import MatlabPacketDatabase, PacketBatch
from .metadata import ReceiverMetadata


class ReceiverBackend(Protocol):
    """Interface to be implemented by the forthcoming CUDA Python binding."""

    @property
    def buffer_count(self) -> int: ...

    def prepare(self, sample_count: int, batch_capacity: int) -> None: ...

    def submit(
        self, batch: PacketBatch, metadata: ReceiverMetadata, llr_scale: float
    ) -> int: ...

    def collect(self, ticket: int) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class BenchmarkResult:
    packets: int
    elapsed_seconds: float
    packets_per_second: float
    payload_mbps: float
    checksum: int
    expected_checksum: int
    stage_totals_ms: Mapping[str, float]


def run_gpu_benchmark(
    database: MatlabPacketDatabase,
    metadata: ReceiverMetadata,
    backend: ReceiverBackend,
    *,
    num_packets: int,
    batch_size: int = 1024,
    llr_scale: float = 3.5,
) -> BenchmarkResult:
    """Run the four-buffer algorithm used by the MATLAB benchmark."""
    if num_packets <= 0:
        raise ValueError("num_packets must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if database.sample_count < metadata.packet_length_samples:
        raise ValueError("Database waveforms are shorter than the receiver packet length")
    if database.payload_bits != metadata.payload_bits:
        raise ValueError("Database and metadata payload sizes differ")
    if llr_scale <= 0:
        raise ValueError("llr_scale must be positive")

    capacity = min(batch_size, num_packets)
    backend.prepare(database.sample_count, capacity)
    max_in_flight = int(backend.buffer_count)
    if max_in_flight <= 0:
        raise ValueError("backend.buffer_count must be positive")
    pending: list[tuple[int, PacketBatch]] = []
    checksum = 0
    stage_totals: dict[str, float] = {}

    def collect_oldest() -> None:
        nonlocal checksum
        ticket, _batch = pending.pop(0)
        stats = backend.collect(ticket)
        checksum += sum(int(value) for value in stats["checksums"])
        for name, value in dict(stats.get("stage_times_ms", {})).items():
            stage_totals[name] = stage_totals.get(name, 0.0) + float(value)

    start = perf_counter()
    for batch in database.batches(num_packets, capacity):
        if len(pending) == max_in_flight:
            collect_oldest()
        pending.append((backend.submit(batch, metadata, llr_scale), batch))
    while pending:
        collect_oldest()
    elapsed = perf_counter() - start

    source = database.source_indices(num_packets)
    expected = int(database.expected_checksums(source).sum(dtype=np.uint64))
    if checksum != expected:
        raise RuntimeError(
            f"CUDA checksum mismatch: recovered {checksum}, expected {expected}"
        )

    return BenchmarkResult(
        packets=num_packets,
        elapsed_seconds=elapsed,
        packets_per_second=num_packets / elapsed,
        payload_mbps=(num_packets * metadata.payload_bits) / elapsed / 1e6,
        checksum=checksum,
        expected_checksum=expected,
        stage_totals_ms=stage_totals,
    )
