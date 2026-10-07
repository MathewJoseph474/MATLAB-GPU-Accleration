"""Fixed numeric inputs exported from MATLAB's WLAN Toolbox."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat


def _vector(values: object, dtype: np.dtype) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(-1))


@dataclass(frozen=True)
class ReceiverMetadata:
    schema_version: int
    index_base: int
    sample_rate_hz: float
    packet_length_samples: int
    payload_bits: int
    lltf_reference: NDArray[np.complex64]
    ltf_fft_indices: NDArray[np.int32]
    data_fft_indices: NDArray[np.int32]
    ltf_fft_start: int
    data_fft_starts: NDArray[np.int32]
    known_ltf: NDArray[np.complex64]
    pilot_indices: NDArray[np.int32]
    pilot_reference: NDArray[np.complex64]
    data_indices: NDArray[np.int32]

    @classmethod
    def load(cls, path: str | Path) -> "ReceiverMetadata":
        raw = loadmat(path, squeeze_me=True)

        def scalar(name: str, cast: type):
            if name not in raw:
                raise ValueError(f"Metadata is missing {name}")
            return cast(np.asarray(raw[name]).item())

        def array(name: str, dtype: np.dtype) -> np.ndarray:
            if name not in raw:
                raise ValueError(f"Metadata is missing {name}")
            return _vector(raw[name], dtype)

        metadata = cls(
            schema_version=scalar("schemaVersion", int),
            index_base=scalar("indexBase", int),
            sample_rate_hz=scalar("sampleRateHz", float),
            packet_length_samples=scalar("packetLengthSamples", int),
            payload_bits=scalar("payloadBits", int),
            lltf_reference=array("lltfReference", np.dtype(np.complex64)),
            ltf_fft_indices=array("ltfFFTIndices", np.dtype(np.int32)),
            data_fft_indices=array("dataFFTIndices", np.dtype(np.int32)),
            ltf_fft_start=scalar("ltfFFTStart", int),
            data_fft_starts=array("dataFFTStarts", np.dtype(np.int32)),
            known_ltf=array("knownLTF", np.dtype(np.complex64)),
            pilot_indices=array("pilotIndices", np.dtype(np.int32)),
            pilot_reference=array("pilotReference", np.dtype(np.complex64)),
            data_indices=array("dataIndices", np.dtype(np.int32)),
        )
        metadata.validate()
        return metadata

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported metadata schema {self.schema_version}")
        if self.index_base != 1:
            raise ValueError("CUDA receiver metadata must use MATLAB one-based indices")
        if self.lltf_reference.size != 2560:
            raise ValueError("L-LTF reference must contain 2560 samples")
        if self.packet_length_samples <= 0 or self.payload_bits <= 0:
            raise ValueError("Packet dimensions must be positive")
        if self.sample_rate_hz <= 0:
            raise ValueError("Sample rate must be positive")
