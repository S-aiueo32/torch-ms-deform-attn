"""Prepare the reviewed Nix artifact for GPU validation without recompiling it."""

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_RUN = "35280414318"
RECORD = ROOT / "docs/validation/kernel-hub-nix" / f"run-{BUILD_RUN}"
VARIANT = "torch211-cxx11-cu126-x86_64-linux"


def prepare(artifact, destination, revision):
    archive = artifact / "distribution.tar.gz"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != (RECORD / "distribution.sha256").read_text().split()[0]:
        raise ValueError("Nix distribution archive checksum mismatch")
    upstream = json.loads((RECORD / "UPSTREAM.json").read_text())
    # Require the reviewed binary's implementation and test sources unchanged.
    for name, expected in upstream["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Nix artifact source changed: {name}")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "kernel-hub/export.py"),
            str(destination),
            "--revision",
            revision,
        ],
        check=True,
    )
    expected = json.loads((RECORD / "distribution-files.json").read_text())
    files = {}
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            if member.isdir():
                continue
            if not member.isfile() or member.name not in expected:
                raise ValueError(f"Unexpected archive member: {member.name}")
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != VARIANT:
                raise ValueError("Invalid distribution path")
            data = bundle.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != expected[member.name]:
                raise ValueError(f"Distribution file checksum mismatch: {member.name}")
            target = destination / "build" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            files[relative.relative_to(VARIANT).as_posix()] = expected[member.name]
    if len(files) != len(expected):
        raise ValueError("Incomplete distribution archive")
    provenance = {
        "build_run": BUILD_RUN,
        "archive_sha256": digest,
        "artifact_source_sha": upstream["revision"],
        "export_revision": (RECORD / "export-revision.txt").read_text().strip(),
        "variant": VARIANT,
        "files": files,
    }
    (destination / "NIX_BUILD.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (destination / "nix-UPSTREAM.json").write_bytes((RECORD / "UPSTREAM.json").read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    prepare(args.artifact, args.destination, args.revision)
