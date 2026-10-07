from collections.abc import Mapping

import numpy as np

from eht_receiver.database import PacketBatch
from eht_receiver.runner import run_gpu_benchmark


class FakeDatabase:
    sample_count = 12
    payload_bits = 4
    unique_packets = 2

    def source_indices(self, count):
        return np.arange(count, dtype=np.int64) % self.unique_packets

    def read_bits(self, indices):
        values = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int8)
        return values[np.asarray(indices)]

    def expected_checksums(self, indices):
        return self.read_bits(indices).astype(np.uint64) @ np.arange(1, 5, dtype=np.uint64)

    def batches(self, count, batch_size):
        source = self.source_indices(count)
        for first in range(0, count, batch_size):
            idx = source[first : first + batch_size]
            shape = (idx.size, self.sample_count)
            yield PacketBatch(
                idx,
                np.zeros(shape, dtype=np.int16),
                np.zeros(shape, dtype=np.int16),
                np.ones(idx.size, dtype=np.float32),
            )


class FakeMetadata:
    packet_length_samples = 10
    payload_bits = 4


class FakeBackend:
    buffer_count = 2

    def __init__(self, database):
        self.database = database
        self.pending = {}
        self.next_ticket = 1
        self.peak = 0

    def prepare(self, sample_count, batch_capacity):
        assert sample_count == 12
        assert batch_capacity == 3

    def submit(self, batch, metadata, llr_scale):
        ticket = self.next_ticket
        self.next_ticket += 1
        self.pending[ticket] = batch.source_indices.copy()
        self.peak = max(self.peak, len(self.pending))
        return ticket

    def collect(self, ticket) -> Mapping[str, object]:
        source = self.pending.pop(ticket)
        return {
            "checksums": self.database.expected_checksums(source),
            "stage_times_ms": {"decode": 1.25},
        }


def test_async_runner_recycles_packets_and_validates_checksum():
    database = FakeDatabase()
    backend = FakeBackend(database)

    result = run_gpu_benchmark(
        database, FakeMetadata(), backend, num_packets=7, batch_size=3
    )

    assert result.packets == 7
    assert result.checksum == result.expected_checksum == 34
    assert result.stage_totals_ms == {"decode": 3.75}
    assert backend.peak == 2
