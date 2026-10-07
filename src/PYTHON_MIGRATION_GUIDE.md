# MATLAB-to-Python Migration Guide

This guide explains how to use the Python migration currently available in
this repository and what remains before the complete CUDA receiver can run
without MATLAB.

## Current status

The following parts have been translated:

- Reading the existing MATLAB v7.3 packet database without loading all
  packets into memory.
- Converting MATLAB's stored array layout into contiguous packet-major arrays
  suitable for the CUDA receiver.
- Reusing the 128 source packets for larger benchmark workloads.
- Constructing GPU batches.
- Managing multiple asynchronous batches.
- Computing and validating recovered-payload checksums.
- Loading and validating fixed receiver metadata exported by MATLAB.

The CUDA kernels are still reached through the MATLAB MEX gateway. A Python
native extension must still be added before the complete GPU benchmark can be
launched directly from Python.

## 1. Install the prerequisites

You need:

- Python 3.10 or newer.
- MATLAB with WLAN Toolbox for the one-time metadata export.
- An NVIDIA CUDA-capable GPU.
- A CUDA toolkit compatible with the project.
- A C++17 compiler supported by the installed CUDA toolkit.

The existing packet database must be present at:

```text
database/eht_receiver_database_128.mat
```

## 2. Export the WLAN receiver metadata

MATLAB stores a `wlanEHTMUConfig` object in the database. Python cannot use
that proprietary MATLAB object directly. Run the exporter once to convert the
required WLAN values into ordinary numeric arrays.

From MATLAB, change to the project root and run:

```matlab
addpath(fullfile(pwd,'src'));
export_eht_receiver_python_metadata;
```

This creates:

```text
database/eht_receiver_python_metadata.mat
```

The exported file contains the L-LTF reference, FFT indices, FFT window
positions, known EHT-LTF values, pilot references, data indices, sample rate,
packet length, and payload size.

You only need to export it again when the WLAN configuration or packet format
changes.

## 3. Create a Python environment

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "src/python[test]"
```

The installation includes NumPy, SciPy, h5py, and pytest.

## 4. Check the packet database

With the virtual environment active, run:

```bash
eht-receiver inspect database/eht_receiver_database_128.mat
```

The command should print:

- Number of unique packets.
- Samples per packet.
- Payload bits per packet.

The loader exposes packets as arrays shaped `[packets, samples]`. This is
intentional: each packet remains contiguous in memory and matches the CUDA
backend's `packet * sample_count + sample` indexing.

## 5. Check the exported metadata

Run:

```bash
eht-receiver validate-metadata \
  database/eht_receiver_python_metadata.mat
```

The command validates the metadata schema and prints its main dimensions.

If this command reports that the metadata file is missing, complete step 2 in
MATLAB first.

## 6. Run the Python tests

From the project root:

```bash
PYTHONPATH=src/python python -m pytest src/python/tests
```

The current tests verify metadata loading, packet recycling, asynchronous
batch limits, timing aggregation, and checksum validation using a simulated
receiver backend.

## 7. Build the Python CUDA binding (remaining work)

The next development milestone is a native module named, for example,
`_eht_receiver_cuda`. It should be implemented with pybind11 or a similarly
small C-compatible wrapper.

The CUDA source should be divided into three layers:

1. CUDA kernels and receiver state that do not include MATLAB headers.
2. A small MATLAB MEX adapter that translates `mxArray` inputs and outputs.
3. A Python adapter that translates NumPy arrays and Python result objects.

The Python adapter must implement the `ReceiverBackend` interface declared in:

```text
src/python/eht_receiver/runner.py
```

It requires these operations:

```python
prepare(sample_count, batch_capacity)
submit(batch, metadata, llr_scale) -> ticket
collect(ticket) -> statistics
buffer_count
```

Input requirements for `submit` are:

- `rx_i`: C-contiguous `int16`, shaped `[batch, samples]`.
- `rx_q`: C-contiguous `int16`, shaped `[batch, samples]`.
- `rx_scale`: contiguous `float32`, one value per packet.
- Receiver reference arrays from `ReceiverMetadata`.
- A positive floating-point LLR scale.

The collected statistics must contain at least:

```python
{
    "checksums": [...],
    "stage_times_ms": {
        "packet_detection": 0.0,
        "coarse_cfo": 0.0,
        "timing_synchronization": 0.0,
        "fine_cfo": 0.0,
        "ofdm_demodulation": 0.0,
        "channel_estimation": 0.0,
        "equalization": 0.0,
        "qam_demapping": 0.0,
        "llr_generation": 0.0,
        "ldpc_decoding": 0.0,
        "descrambling": 0.0,
        "payload_recovery": 0.0,
    },
}
```

## 8. Validate the native binding

Before replacing the MATLAB benchmark, compare both implementations using the
same packets and configuration:

1. Run a small validation set through MATLAB and Python.
2. Compare every packet checksum.
3. Compare total bit errors and failed packets.
4. Test partial final batches, such as 1,025 packets with a batch size of
   1,024.
5. Test more batches than available asynchronous buffers.
6. Run CUDA Compute Sanitizer on the Python extension.
7. Compare 8K, 16K, and 32K workload results.

Do not compare only throughput. The checksum and bit-error results must match
before accepting performance measurements.

## What still requires MATLAB

Until packet generation itself is reimplemented, MATLAB and WLAN Toolbox are
still required to:

- Generate new EHT waveforms.
- Apply the TGax Channel Model D simulation.
- Change the EHT configuration.
- Regenerate fixed receiver metadata.
- Produce trusted reference results for parity testing.

Once the numeric metadata has been exported and the Python CUDA binding is
complete, MATLAB will not be required to run benchmarks against the existing
packet database.

## Important limitations

- Do not pass default NumPy arrays shaped `[samples, packets]` to the CUDA
  backend. Default NumPy arrays are row-major, unlike MATLAB arrays.
- Keep the database reader's `[packets, samples]` representation unless the
  CUDA indexing is changed at the same time.
- MATLAB indices in the exported metadata are one-based. The native backend
  must convert them in exactly the same places as the current MEX gateway.
- The Python orchestration exists now, but it cannot execute the real CUDA
  receiver until the native Python binding is implemented.
