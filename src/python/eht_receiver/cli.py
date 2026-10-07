"""Command-line utilities for the incremental Python migration."""

from __future__ import annotations

import argparse

from .database import MatlabPacketDatabase
from .metadata import ReceiverMetadata


def main() -> None:
    parser = argparse.ArgumentParser(prog="eht-receiver")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect packet database")
    inspect_parser.add_argument("database")
    metadata_parser = subparsers.add_parser(
        "validate-metadata", help="validate MATLAB-exported receiver metadata"
    )
    metadata_parser.add_argument("metadata")
    args = parser.parse_args()

    if args.command == "inspect":
        with MatlabPacketDatabase(args.database) as database:
            print(f"packets: {database.unique_packets}")
            print(f"samples per packet: {database.sample_count}")
            print(f"payload bits: {database.payload_bits}")
    else:
        metadata = ReceiverMetadata.load(args.metadata)
        print(f"schema: {metadata.schema_version}")
        print(f"sample rate: {metadata.sample_rate_hz:g} Hz")
        print(f"packet samples: {metadata.packet_length_samples}")
        print(f"payload bits: {metadata.payload_bits}")


if __name__ == "__main__":
    main()
