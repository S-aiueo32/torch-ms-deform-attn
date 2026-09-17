"""Export a self-contained builder tree; never maintain a second kernel copy."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Upstream adapter contract changed: {old!r}")
    return text.replace(old, new, 1)


def export(destination, revision=None):
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full Git commit SHA")
    destination = Path(destination).resolve()
    # Requiring a new directory prevents stale files and accidental overwrites.
    destination.mkdir(parents=True, exist_ok=False)
    sources = {}

    def write(source, target=None, transform=None):
        data = (ROOT / source).read_bytes()
        sources[source] = hashlib.sha256(data).hexdigest()
        target = destination / (target or source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(transform(data.decode()).encode() if transform else data)

    for name in ("LICENSE", "NOTICE", "csrc/dispatcher.h"):
        write(name)
    for source in sorted((ROOT / "csrc/cuda").glob("*")):
        if source.is_file():
            write(source.relative_to(ROOT).as_posix())
    for source in sorted((ROOT / "kernel-hub/torch-ext").rglob("*")):
        if source.is_file() and source.suffix in (".py", ".cpp", ".h"):
            write(
                source.relative_to(ROOT).as_posix(),
                source.relative_to(ROOT / "kernel-hub").as_posix(),
            )
    write("kernel-hub/build.toml", "build.toml")
    write("kernel-hub/flake.nix", "flake.nix")
    for source in sorted((ROOT / "kernel-hub/tests").glob("*.py")):
        write(source.relative_to(ROOT).as_posix(), f"tests/{source.name}")
    for name in ("rt_detr.py", "requirements.txt"):
        write(f"kernel-hub/e2e/{name}", f"e2e/{name}")

    def registrations(text):
        text = replace_once(
            text, "from . import _C", "from ._ops import add_op_namespace_prefix, ops as _C"
        )
        for name in ("forward", "backward"):
            text = replace_once(
                text, f'"torch_ms_deform_attn::{name}"', f'add_op_namespace_prefix("{name}")'
            )
        return text

    write(
        "src/torch_ms_deform_attn/_ops.py",
        "torch-ext/deformable_detr/_registrations.py",
        registrations,
    )
    write(
        "src/torch_ms_deform_attn/functional.py",
        "torch-ext/deformable_detr/functional.py",
        lambda text: replace_once(
            text,
            "from ._ops import forward as _forward",
            "from ._registrations import forward as _forward",
        ),
    )
    if revision is None:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    (destination / "UPSTREAM.json").write_text(
        json.dumps({"revision": revision, "source_sha256": sources}, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--revision", help="Full source SHA when exporting a git archive without .git"
    )
    args = parser.parse_args()
    export(args.destination, revision=args.revision)
