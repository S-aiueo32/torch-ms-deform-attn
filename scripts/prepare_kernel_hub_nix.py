"""Prepare the reviewed Nix artifact for GPU validation without recompiling it."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANT = "torch211-cxx11-cu126-x86_64-linux"


def validate_build_run(build_run):
    if not re.fullmatch(r"[1-9][0-9]*", build_run):
        raise ValueError("Nix build run must be a positive numeric Actions run ID")


def download(artifact, build_run, release=""):
    validate_build_run(build_run)
    artifact.mkdir(parents=True, exist_ok=False)
    name = f"kernel-hub-nix-{build_run}"
    if release:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["gh", "release", "download", release, "--pattern", f"{name}.zip", "--dir", tmp],
                check=True,
            )
            with zipfile.ZipFile(Path(tmp) / f"{name}.zip") as bundle:
                bundle.extractall(artifact)
    else:
        subprocess.run(
            ["gh", "run", "download", build_run, "--name", name, "--dir", str(artifact)],
            check=True,
        )


def prepare(artifact, destination, revision, build_run):
    validate_build_run(build_run)
    archive = artifact / "distribution.tar.gz"
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != (artifact / "distribution.sha256").read_text().split()[0]:
        raise ValueError("Nix distribution archive checksum mismatch")
    upstream = json.loads((artifact / "UPSTREAM.json").read_text())
    # Implementation/build inputs must stay identical; model validation code
    # can evolve independently and is hashed in the current UPSTREAM manifest.
    for name, expected in upstream["source_sha256"].items():
        if name.startswith("kernel-hub/e2e/"):
            continue
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
    expected = json.loads((artifact / "distribution-files.json").read_text())
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
        "build_run": build_run,
        "archive_sha256": digest,
        "artifact_source_sha": upstream["revision"],
        "export_revision": (artifact / "export-revision.txt").read_text().strip(),
        "variant": VARIANT,
        "files": files,
    }
    (destination / "NIX_BUILD.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (destination / "nix-UPSTREAM.json").write_bytes((artifact / "UPSTREAM.json").read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--build-run", required=True)
    parser.add_argument(
        "--download", action="store_true", help="Download into a new artifact directory"
    )
    parser.add_argument("--release", default="", help="Download a saved ZIP from this release tag")
    args = parser.parse_args()
    if args.release and not args.download:
        parser.error("--release requires --download")
    if args.download:
        download(args.artifact, args.build_run, args.release)
    prepare(args.artifact, args.destination, args.revision, args.build_run)
