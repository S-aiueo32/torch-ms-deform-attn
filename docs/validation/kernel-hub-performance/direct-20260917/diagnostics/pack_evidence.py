#!/usr/bin/env python3
"""Archive direct profiling evidence, excluding binaries and credentials."""

import hashlib
import json
import tarfile
from pathlib import Path

root = Path("/workspace/ci")
results = root / "results"
singles = []
for path in sorted(results.glob("single-*/one-case.json")):
    data = json.loads(path.read_text())
    profile = data.get("profile", {})
    singles.append(
        {
            "source": str(path.relative_to(results)),
            "backend": data["config"]["backend"],
            "mode": data["config"]["mode"],
            "completed": data.get("completed"),
            "correctness": data.get("correctness"),
            "profile": profile,
            "error": data.get("error"),
        }
    )
assert len(singles) == 10
assert all(x["completed"] and x["correctness"]["passed"] and not x["error"] for x in singles)
(results / "single-profiles.json").write_text(json.dumps(singles, indent=2) + "\n")

files = []
for folder in ["run2", "run3", "host-probe", "binary", "ncu-case"]:
    files.extend(p for p in (results / folder).rglob("*") if p.is_file())
for folder in sorted(results.glob("single-*")):
    if folder.is_dir():
        files.extend(p for p in folder.rglob("*") if p.is_file())
files.extend(p for p in results.iterdir() if p.is_file() and p.suffix in {".json", ".txt", ".log"})
files.extend(
    p
    for p in (root / "source" / "diagnostics").iterdir()
    if p.is_file() and p.suffix in {".py", ".sh"}
)
files.extend(
    p
    for p in (root / "variants").rglob("*")
    if p.is_file() and p.suffix in {".cu", ".cuh", ".h", ".cpp", ".json", ".log", ".ninja"}
)
files = sorted(set(files))
manifest = {
    str(p.relative_to(root)): {
        "bytes": p.stat().st_size,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }
    for p in files
}
manifest_path = results / "evidence-manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
archive = results / "profiling-evidence.tar.gz"
with tarfile.open(archive, "w:gz") as bundle:
    for path in files + [manifest_path]:
        bundle.add(path, arcname=str(path.relative_to(root)), recursive=False)
print(
    json.dumps(
        {
            "archive": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "members": len(files) + 1,
        }
    )
)
for item in singles:
    p = item["profile"]
    kernels = {
        name: round(value["us_per_call"], 3)
        for name, value in p.get("cuda", {}).items()
        if "ms_deformable_" in name
    }
    print(
        json.dumps(
            {
                "backend": item["backend"],
                "mode": item["mode"],
                "complete": p.get("cuda_counts_complete"),
                "counts": p.get("msda_kernel_counts"),
                "cuda_us": p.get("cuda_total_us_per_iteration"),
                "kernels": kernels,
                "warning": p.get("warning"),
            }
        )
    )
