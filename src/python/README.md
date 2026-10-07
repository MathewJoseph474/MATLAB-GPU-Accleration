# Python migration

This directory is the first runnable part of the MATLAB-to-Python migration.
It reads the existing MATLAB v7.3 packet database lazily, preserves the memory
layout expected by CUDA, recreates packet recycling and asynchronous batching,
and validates recovered-payload checksums.

## Setup

From `src/python`:

```bash
python -m pip install -e .
```

In MATLAB, export the numeric receiver constants once:

```matlab
export_eht_receiver_python_metadata
```

Then inspect both inputs:

```bash
eht-receiver inspect ../../database/eht_receiver_database_128.mat
eht-receiver validate-metadata ../../database/eht_receiver_python_metadata.mat
```

`eht_receiver.runner.run_gpu_benchmark` contains the benchmark loop and takes
any object implementing `ReceiverBackend`. The next migration step is to split
the CUDA kernels from the MEX gateway and provide that interface with pybind11.
MATLAB/WLAN Toolbox remains necessary only to regenerate packets or fixed
metadata; it is not part of the Python benchmark's intended runtime path.
