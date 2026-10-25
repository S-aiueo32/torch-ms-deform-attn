"""Record the exact files in a Nix distribution for checked GPU preparation."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def record(archive, destination, variant):
    files = {}
    paths = set()
    with tarfile.open(archive) as bundle:
        for member in bundle:
            relative = PurePosixPath(member.name)
            if member.isdir() and relative == PurePosixPath("."):
                continue
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.parts[0] != variant
            ):
                raise ValueError(f"Invalid distribution path: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported archive member: {member.name}")
            if relative in paths:
                raise ValueError(f"Duplicate distribution path: {member.name}")
            paths.add(relative)
            digest = hashlib.sha256()
            with bundle.extractfile(member) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            # Preserve the tar member name, including a leading './', because
            # prepare_kernel_hub_nix.py verifies exact archive membership.
            files[member.name] = digest.hexdigest()
    if not files:
        raise ValueError("Empty distribution archive")
    Path(destination).write_text(json.dumps(files, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    record(args.archive, args.destination, args.variant)
