from pathlib import Path

import numpy as np
from scipy.io import savemat

from eht_receiver.metadata import ReceiverMetadata


def test_load_exported_metadata(tmp_path: Path) -> None:
    path = tmp_path / "metadata.mat"
    savemat(
        path,
        {
            "schemaVersion": np.int32(1),
            "indexBase": np.int32(1),
            "sampleRateHz": 320e6,
            "packetLengthSamples": np.int32(10000),
            "payloadBits": np.int32(32000),
            "lltfReference": np.ones(2560, dtype=np.complex64),
            "ltfFFTIndices": np.arange(4, dtype=np.int32) + 1,
            "dataFFTIndices": np.arange(8, dtype=np.int32) + 1,
            "ltfFFTStart": np.int32(65),
            "dataFFTStarts": np.array([65, 4225], dtype=np.int32),
            "knownLTF": np.ones(4, dtype=np.complex64),
            "pilotIndices": np.array([2, 4], dtype=np.int32),
            "pilotReference": np.ones(4, dtype=np.complex64),
            "dataIndices": np.array([1, 3], dtype=np.int32),
        },
    )

    metadata = ReceiverMetadata.load(path)

    assert metadata.payload_bits == 32000
    assert metadata.lltf_reference.dtype == np.complex64
    assert metadata.data_fft_starts.flags.c_contiguous
